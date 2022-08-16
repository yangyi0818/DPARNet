#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# yangyi 2022.06

import torch
import torch.nn as nn
import torch.nn.functional as F
from fast_transformers.masking import FullMask, LengthMask, TriangularCausalMask
from fast_transformers.attention import SharedLinearAttention, LinearAttention

from asteroid.engine.optimizers import make_optimizer
from asteroid.losses import PITLossWrapper, pairwise_neg_snr

from feature import FeatureExtractor, AngleFeature
from mvdr_util import MVDR

def make_model_and_optimizer(conf):
    model = DPARNet1_DPARNet2()
    print(model)
    optimizer = make_optimizer(model.parameters(), **conf['optim'])
    return model, optimizer


class DPARNet1_DPARNet2(nn.Module):
    def __init__(self):
        super(DPARNet1_DPARNet2, self).__init__()

        self.mvdr = MVDR(False)

        self.DPARNet1 = DPARNet1(
                use_dense = False,
                use_att = [True, True, True, True, False],
                use_rnn = [True, True, False, False, False],
                width = 64,
                num_layers = 4,
                dropout_rate = 0.4,
                causal_conf = False,
                )

        self.DPARNet2 = DPARNet2(
                use_dense = False,
                use_att = [True, True, True, True, False],
                use_rnn = [True, False, False, False, False],
                width = 64,
                num_layers = 4,
                dropout_rate = 0.4,
                causal_conf = False,
                )

    def forward(self, mixture, do_eval = 0): # mixture [b m n]
        assert do_eval == 0 or do_eval == 1, "Eval type should be 0 (training) or 1 (multi-channel beamforming) !"

        out_DPARNet1, pha_src = self.DPARNet1(mixture)           # [b s m n] # [b m f t]
        out_DPARNet1, pha_src = out_DPARNet1.detach(), pha_src.detach()
        bf_DPARNet1 = self.mvdr(mixture, out_DPARNet1)           # [b s m n]
        out_DPARNet2_iter1 = self.DPARNet2(mixture, bf_DPARNet1) # [b s m n]

        bf_DPARNet2_iter1 = self.mvdr(mixture, out_DPARNet2_iter1)     # [b s m n]
        out_DPARNet2_iter2 = self.DPARNet2(mixture, bf_DPARNet2_iter1) # [b s m n]


        if do_eval == 0:
            #return out_DPARNet1, out_DPARNet2_iter1, pha_src
            return out_DPARNet1, out_DPARNet2_iter1, out_DPARNet2_iter2, pha_src

        else:
            bf_DPARNet2_iter1 = self.mvdr(mixture, out_DPARNet2_iter1)     # [b s m n]

            out_DPARNet2_iter2 = self.DPARNet2(mixture, bf_DPARNet2_iter1) # [b s m n]
            bf_DPARNet2_iter2 = self.mvdr(mixture, out_DPARNet2_iter2)     # [b s m n]

            out_DPARNet2_iter3 = self.DPARNet2(mixture, bf_DPARNet2_iter2) # [b s m n]
            bf_DPARNet2_iter3 = self.mvdr(mixture, out_DPARNet2_iter3)     # [b s m n]

            out_DPARNet2_iter4 = self.DPARNet2(mixture, bf_DPARNet2_iter3) # [b s m n]

            return out_DPARNet2_iter4[:,:,0], out_DPARNet2_iter4


def half_reshape(x, inverse):
    if(inverse):
        x = torch.cat([x[:,0:x.shape[1]//2,],x[:,x.shape[1]//2:x.shape[1]//2*2,]],-1)
        x = x[...,:-1]
        return x
    else:
        x = F.pad(x, (0,1,0,0), 'replicate')
        x = torch.cat([x[:,:,:,0:x.shape[-1]//2],x[:,:,:,x.shape[-1]//2:x.shape[-1]//2*2]],1)
        return x

class DPARNet1(nn.Module):
    def __init__(self,
                 use_dense,
                 use_att,
                 use_rnn,
                 num_channels=7,
                 num_spks=2,
                 frame_len=512,
                 frame_hop=128,
                 width=48,
                 num_layers=3,
                 dropout_rate=0.4,
                 causal_conf=False,
                ):
        super(DPARNet1, self).__init__()

        self.use_dense = use_dense
        self.use_att = use_att
        self.use_rnn = use_rnn

        self.num_channels = num_channels
        self.num_spks = num_spks
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.width = width
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.causal_conf = causal_conf
        self.num_bins = self.frame_len // 2 + 1

        self.extractor = FeatureExtractor(frame_len=self.frame_len, frame_hop=self.frame_hop, do_ipd=False)

        self.in_Conv = nn.Sequential(
                nn.Conv2d(in_channels=self.num_channels * 2 * 2, out_channels=self.width, kernel_size=(1, 1)),
                nn.LayerNorm(self.num_bins // 2 + 1),
                nn.PReLU(self.width),
                )

        self.in_Conv_att = nn.Sequential(nn.Conv2d(self.width, self.width // 2, kernel_size=(1, 1)), nn.PReLU())

        self.dualrnn_attention = nn.ModuleList()
        for i in range (self.num_layers):
            self.dualrnn_attention.append(DualRNN_Attention(dropout_rate=self.dropout_rate, d_model=self.width//2, use_att=self.use_att[i], use_rnn=self.use_rnn[i]))

        self.out_Conv_att = nn.Sequential(nn.Conv2d(self.width // 2, self.width, kernel_size=(1, 1)), nn.PReLU())

        self.out_Conv1 = nn.Sequential(nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 1)), nn.Tanh(),)
        self.out_Conv2 = nn.Sequential(nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 1)), nn.Sigmoid(),)

        self.out_Conv = nn.ConvTranspose2d(in_channels=self.width//2, out_channels=self.num_spks * self.num_channels, kernel_size=(1, 1))

        if (self.use_dense):
            self.in_DenseBlock = DenseBlock(init_ch=self.width, g1=8, g2=self.width)
            self.out_DenseBlock = DenseBlock(init_ch=self.width, g1=8, g2=self.width)
        else:
            self.in_DenseBlock = None
            self.out_DenseBlock = None

        self.sigmoid = nn.Sigmoid()


    def forward(self, mixture): # mixture: [b m n]

        real_spec_src, imag_spec_src = self.extractor.stft(mixture, cplx=True)       # [b m f t]
        com_spec_src = torch.cat((real_spec_src, imag_spec_src), 1).permute(0,1,3,2) # [b 2m t f=257]

        x = half_reshape(com_spec_src, False) # [b 4m t f=129]

        x = self.in_Conv(x)  # [b w=64 t f]
        if (self.use_dense):
            x = self.in_DenseBlock(x)
        x = self.in_Conv_att(x) # [b w=32 t f]

        for i in range (self.num_layers):
            x = self.dualrnn_attention[i](x) # [b w=64 t f]

        x = self.out_Conv_att(x) # [b w=64 t f]

        x = self.out_Conv1(x) * self.out_Conv2(x) # [b w=64 t f]

        if (self.use_dense):
            x = self.out_DenseBlock(x)

        x = half_reshape(x, True)  # [b w=32 t f=257]

        irm_est = self.sigmoid(self.out_Conv(x)).transpose(2,3) # [b sxm f t]
        irm_est = irm_est.chunk(self.num_spks, dim=1) # [b m f t] * s

        mag_src, pha_src = self.extractor.stft(mixture, cplx=False) # [b m f t]

        est_sig = []
        for id_spk in range (self.num_spks):
            for id_chan in range (self.num_channels):
                mag_est = irm_est[id_spk][:,id_chan] * mag_src[:,id_chan]
                est_sig.append(self.extractor.istft(mag_est, pha_src[:,id_chan], cplx=False))

        output = torch.stack(est_sig, 1) # [b mxs n]
        output = torch.stack(output.chunk(self.num_spks, dim=1), 1) # [b s m n]
        output = torch.nn.functional.pad(output,[0,mixture.shape[-1]-output.shape[-1]])

        return output, pha_src ## [b s m n] [b m f t]


class DPARNet2(nn.Module):
    def __init__(self,
                 use_dense,
                 use_att,
                 use_rnn,
                 num_channels=7,
                 num_spks=2,
                 frame_len=512,
                 frame_hop=128,
                 width=48,
                 num_layers=3,
                 dropout_rate=0.4,
                 causal_conf=False,
                ):
        super(DPARNet2, self).__init__()

        self.use_dense = use_dense
        self.use_att = use_att
        self.use_rnn = use_rnn

        self.num_channels = num_channels
        self.num_spks = num_spks
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.width = width
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.causal_conf = causal_conf
        self.num_bins = self.frame_len // 2 + 1

        self.extractor = FeatureExtractor(frame_len=self.frame_len, frame_hop=self.frame_hop, do_ipd=False)

        self.in_Conv = nn.Sequential(
                nn.Conv2d(in_channels=self.num_channels * 2 * 2 * (self.num_spks+1), out_channels=self.width, kernel_size=(1, 1)),
                nn.LayerNorm(self.num_bins // 2 + 1),
                nn.PReLU(self.width),
                )

        self.in_Conv_att = nn.Sequential(nn.Conv2d(self.width, self.width // 2, kernel_size=(1, 1)), nn.PReLU())

        self.dualrnn_attention = nn.ModuleList()
        for i in range (self.num_layers):
            self.dualrnn_attention.append(DualRNN_Attention(dropout_rate=self.dropout_rate, d_model=self.width//2, use_att=self.use_att[i], use_rnn=self.use_rnn[i]))

        self.out_Conv1 = nn.Sequential(nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 1)), nn.Tanh(),)
        self.out_Conv2 = nn.Sequential(nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 1)), nn.Sigmoid(),)

        self.out_Conv_att = nn.Sequential(nn.Conv2d(self.width // 2, self.width, kernel_size=(1, 1)), nn.PReLU())

        self.out_Conv = nn.ConvTranspose2d(in_channels=self.width//2, out_channels=self.num_channels * self.num_spks * 2, kernel_size=(1, 1))

        if (self.use_dense):
            self.in_DenseBlock = DenseBlock(init_ch=self.width, g1=8, g2=self.width)
            self.out_DenseBlock = DenseBlock(init_ch=self.width, g1=8, g2=self.width)
        else:
            self.in_DenseBlock = None
            self.out_DenseBlock = None


    def forward(self, mixture, bf_DPARnet1): # mixture: [b m n] bf_DPARnet1: [b s m n]
        
        B = mixture.size(0)
        real_spec_src, imag_spec_src = self.extractor.stft(mixture, cplx=True) # [b m f t]
        real_spec_dparnet1, imag_spec_dparnet1 = self.extractor.stft(bf_DPARnet1.view(B, self.num_spks*self.num_channels, -1), cplx=True) # [b sxm f t]
        com_spec_src = torch.cat((real_spec_src, imag_spec_src, real_spec_dparnet1, imag_spec_dparnet1), 1).permute(0,1,3,2) # [b 2xmx(s+1) t f=257]

        x = half_reshape(com_spec_src, False) # [b 4xmx(s+1) t f=129]

        x = self.in_Conv(x)  # [b w=64 t f]
        if (self.use_dense):
            x = self.in_DenseBlock(x)

        x = self.in_Conv_att(x) # [b w=32 t f]

        for i in range (self.num_layers):
            x = self.dualrnn_attention[i](x) # [b w t f]

        x = self.out_Conv_att(x) # [b w=64 t f]

        x = self.out_Conv1(x) * self.out_Conv2(x) # [b w=64 t f]

        if (self.use_dense):
            x = self.out_DenseBlock(x)

        x = half_reshape(x, True)  # [b w=32 t f=257]

        cmask_est = self.out_Conv(x).transpose(2,3) # [b 2xsxm f t]
        cmask_est = cmask_est.chunk(self.num_spks, dim=1) # [b 2m f t] * 2

        est_sig = []
        for id_spk in range (self.num_spks):
            for id_chan in range (self.num_channels):
                rmask_est = cmask_est[id_spk][:,0+2*id_chan] # [b f t]
                imask_est = cmask_est[id_spk][:,1+2*id_chan]

                real_spec_est = rmask_est * real_spec_src[:,id_chan] - imask_est * imag_spec_src[:,id_chan]
                imag_spec_est = rmask_est * imag_spec_src[:,id_chan] + imask_est * real_spec_src[:,id_chan]
                est_sig.append(self.extractor.istft(real_spec_est, imag_spec_est, cplx=True)) # [b n] * m * s

        output = torch.stack(est_sig, 1) # [b mxs n]
        output = torch.stack(output.chunk(self.num_spks,1), 1) # [b s m n]
        output = torch.nn.functional.pad(output,[0,mixture.shape[-1]-output.shape[-1]])
        
        return output


# NOTE
class DualRNN_Attention(nn.Module):

    def __init__(self, dropout_rate, d_model, nhead=4, use_att=False, use_rnn=False):
        super(DualRNN_Attention,self).__init__()
        self.dropout_rate = dropout_rate
        self.d_model = d_model
        self.nhead = nhead
        self.use_att = use_att
        self.use_rnn = use_rnn

        self.bn = nn.BatchNorm1d(1)

        if (self.use_att):
            self.shared_att1 = SharedLinearAttention(self.d_model)
            self.linear_att1 = nn.Linear(self.d_model, self.d_model*3)
            self.shared_att2 = SharedLinearAttention(self.d_model)
            self.linear_att2 = nn.Linear(self.d_model, self.d_model*3)
            self.ln_att = nn.LayerNorm(self.d_model)
        else:
            self.shared_att1 = None
            self.linear_att1 = None
            self.shared_att2 = None
            self.linear_att2 = None
            self.ln_att = None

        if (self.use_rnn):
            self.rnn1 = nn.LSTM(input_size=self.d_model, hidden_size=self.d_model * 2, num_layers=1, bias=False, bidirectional=True, batch_first=True)
            self.rnn2 = nn.LSTM(input_size=self.d_model, hidden_size=self.d_model * 2, num_layers=1, bias=False, bidirectional=True, batch_first=True)
            self.dropout = nn.Dropout(p=self.dropout_rate)
            self.linear_rnn1 = nn.Linear(in_features = self.d_model * 4, out_features = self.d_model)
            self.linear_rnn2 = nn.Linear(in_features = self.d_model * 4, out_features = self.d_model)
            #self.relu = nn.ReLU()
            self.ln_rnn = nn.LayerNorm(self.d_model)
        else:
            self.rnn1 = None
            self.rnn2 = None
            self.dropout = None
            self.linear_rnn1 = None
            self.linear_rnn2 = None
            #self.relu = None
            self.ln_rnn = None

    def forward(self, x): # [b w t f=129]
        B, W, T, F = x.size()
        att_in1 = x.permute(0,2,3,1).contiguous().view(B*T, F, -1)

        if (not self.use_att):
            att_out1 = att_in1
        else:
            q, k, v = self.linear_att1(att_in1).view(B*T, F, self.nhead, -1).chunk(3,-1)
            m1, m2, m3 = FullMask(q.shape[1], k.shape[1], device=x.device),FullMask(q.shape[0], q.shape[1], device=x.device),FullMask(k.shape[0], k.shape[1], device=x.device)
            att_out1 = self.shared_att1(q, k, v, m1, m2, m3, causal=False).view(B*T, F, -1)
            att_out1 = self.ln_att(att_in1 + att_out1)

        rnn_in1 = att_out1 # [bxt f w]

        if (not self.use_rnn):
            rnn_out1 = rnn_in1
        else:
            rnn_out1, _ = self.rnn1(rnn_in1)
            rnn_out1 = self.linear_rnn1(self.dropout(rnn_out1))
            rnn_out1 = self.ln_rnn(att_in1 + rnn_out1)

        rnn_out1 = rnn_out1.view(B, T, F, -1).permute(0,3,1,2) # [b w t f]

        rnn_out1 = (self.bn(rnn_out1.reshape(B,1,-1))).reshape(*rnn_out1.shape) # [b w t f]
        rnn_out1 = rnn_out1 + x

        att_in2 = rnn_out1.permute(0,3,2,1).contiguous().view(B*F, T, -1)

        if (not self.use_att):
            att_out2 = att_in2
        else:
            q, k, v = self.linear_att2(att_in2).view(B*F, T, self.nhead, -1).chunk(3,-1)
            m1, m2, m3 = FullMask(q.shape[1], k.shape[1], device=x.device),FullMask(q.shape[0], q.shape[1], device=x.device),FullMask(k.shape[0], k.shape[1], device=x.device)
            att_out2 = self.shared_att2(q, k, v, m1, m2, m3, causal=False).view(B*F, T, -1)
            att_out2 = self.ln_att(att_in2 + att_out2)

        rnn_in2 = att_out2 # [bxf t w]

        if (not self.use_rnn):
            rnn_out2 = rnn_in2
        else:
            rnn_out2, _ = self.rnn2(rnn_in2)
            rnn_out2 = self.linear_rnn2(self.dropout(rnn_out2))
            rnn_out2 = self.ln_rnn(att_in2 + rnn_out2)

        rnn_out2 = rnn_out2.view(B, F, T, -1).permute(0,3,2,1) # [b w t f]

        rnn_out2 = (self.bn(rnn_out2.reshape(B,1,-1))).reshape(*rnn_out2.shape) # [b w t f]
        rnn_out2 = rnn_out2 + rnn_out1

        return rnn_out2


class DenseBlock(nn.Module):

    def __init__(self, init_ch, g1, g2):
        super(DenseBlock,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(init_ch, g1, kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ELU(),
            nn.InstanceNorm2d(g1,affine=False)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(init_ch+g1, g1, kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ELU(),
            nn.InstanceNorm2d(g1,affine=False)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(init_ch+2*g1, g1, kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ELU(),
            nn.InstanceNorm2d(g1,affine=False)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(init_ch+3*g1, g1, kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ELU(),
            nn.InstanceNorm2d(g1,affine=False)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(init_ch+4*g1, g2, kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ELU(),
            nn.InstanceNorm2d(g2,affine=False)
        )

    def forward(self,x):
        y0 = self.conv1(x)

        y0_x = torch.cat((x,y0),dim=1)
        y1 = self.conv2(y0_x)

        y1_0_x = torch.cat((x,y0,y1),dim=1)
        y2 = self.conv3(y1_0_x)

        y2_1_0_x = torch.cat((x,y0,y1,y2),dim=1)
        y3 = self.conv4(y2_1_0_x)

        y3_2_1_0_x = torch.cat((x,y0,y1,y2,y3),dim=1)
        y4 = self.conv5(y3_2_1_0_x)

        return y4


# NOTE
class com_sisdr_loss1(nn.Module):
    def __init__(self, frame_len=512, frame_hop=128, num_channel=7):
        super().__init__()
        self.extractor = FeatureExtractor(frame_len=frame_len, frame_hop=frame_hop)
        self.sig_loss = PITLossWrapper(pairwise_neg_snr, pit_from='pw_mtx')

    # est_targets [b s m n] pha_mix [b m f t] targets [b s m n] SP [b]
    #def forward(self, est_targets1, est_targets2_1, pha_mix, targets, SP): 
    def forward(self, est_targets1, est_targets2_1, est_targets2_2, pha_mix, targets, SP):
        B, S, M, N = est_targets1.size()

        mag_src_spk1, pha_src_spk1 = self.extractor.stft(targets[:,0]) # [b m f t]
        mag_src_spk2, pha_src_spk2 = self.extractor.stft(targets[:,1]) # [b m f t]
        _, _, F, T = mag_src_spk1.size()
        mag_src_spk1, pha_src_spk1 = mag_src_spk1.view(-1,F,T), pha_src_spk1.view(-1,F,T)
        mag_src_spk2, pha_src_spk2 = mag_src_spk2.view(-1,F,T), pha_src_spk2.view(-1,F,T)
        pha_mix = pha_mix.view(-1,F,T)

        targets_recover1 = self.extractor.istft(mag_src_spk1 * torch.cos(pha_mix - pha_src_spk1), pha_src_spk1).view(B,M,N)
        targets_recover2 = self.extractor.istft(mag_src_spk2 * torch.cos(pha_mix - pha_src_spk2), pha_src_spk2).view(B,M,N)

        targets_recover = torch.stack((targets_recover1, targets_recover2), 1) # [b s m n]

        est_targets1, targets_recover = est_targets1.permute(0,2,1,3).contiguous().view(B*M, S, N), targets_recover.permute(0,2,1,3).contiguous().view(B*M, S, N)
        est_targets2_1, targets = est_targets2_1.permute(0,2,1,3).contiguous().view(B*M, S, N), targets.permute(0,2,1,3).contiguous().view(B*M, S, N)
        est_targets2_2 = est_targets2_2.permute(0,2,1,3).contiguous().view(B*M, S, N)
        SP = SP.repeat(M,1).transpose(0,1).contiguous().view(B*M)

        if (sum(SP)==0):
            snr_loss1 = self.sig_loss(est_targets1, targets_recover)
            snr_loss2_1 = self.sig_loss(est_targets2_1, targets)
            snr_loss2_2 = self.sig_loss(est_targets2_2, targets)
        elif (sum(SP)==SP.shape[0]):
            snr_loss1 = 0.05 * self.sig_loss(est_targets1[:,[0]], targets_recover[:,[0]])
            snr_loss2_1 = 0.05 * self.sig_loss(est_targets2_1[:,[0]], targets[:,[0]])
            snr_loss2_2 = 0.05 * self.sig_loss(est_targets2_2[:,[0]], targets[:,[0]])
        else:
            snr_loss1 = 0.05 * self.sig_loss(est_targets1[SP == 1][:,[0]], targets_recover[SP == 1][:,[0]]) + self.sig_loss(est_targets1[SP == 0], targets_recover[SP == 0])
            snr_loss2_1 = 0.05 * self.sig_loss(est_targets2_1[SP == 1][:,[0]], targets[SP == 1][:,[0]]) + self.sig_loss(est_targets2_1[SP == 0], targets[SP == 0])
            snr_loss2_2 = 0.05 * self.sig_loss(est_targets2_2[SP == 1][:,[0]], targets[SP == 1][:,[0]]) + self.sig_loss(est_targets2_2[SP == 0], targets[SP == 0])

        #loss = snr_loss2_1.mean()
        #loss_dict = dict(sig_loss=loss, snr_loss1=snr_loss1.mean(), snr_loss2_1=snr_loss2_1.mean())
        loss = (snr_loss2_1.mean() + snr_loss2_2.mean()) / 2
        loss_dict = dict(sig_loss=loss, snr_loss1=snr_loss1.mean(), snr_loss2_1=snr_loss2_1.mean(), snr_loss2_2=snr_loss2_2.mean())

        return loss, loss_dict