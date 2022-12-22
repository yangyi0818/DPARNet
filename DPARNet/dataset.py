import torch
from torch.utils import data
import numpy as np
import os
import soundfile as sf
import math
import random
import shutil

from sms_wsj.database.create_rirs import config, scenarios, rirs
from sms_wsj.reverb.reverb_utils import convolve

EPS=1e-8

def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = torch.mean(wav_tensor, dim=-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)

def rms(y):
    return np.sqrt(np.mean(np.abs(y) ** 2, axis=0, keepdims=False))

def get_amplitude_scaling_factor(s, n, snr, method='rms'):
    original_sn_rms_ratio = rms(s) / rms(n)
    target_sn_rms_ratio =  10. ** (float(snr) / 20.)    # snr = 20 * lg(rms(s) / rms(n))
    signal_scaling_factor = target_sn_rms_ratio / original_sn_rms_ratio
    
class Dataset(data.Dataset):

    def __init__(
        self,
        reverb_matrixs_dir,
        rirNO = 5,
        trainingNO = 5000,
        segment = 6,
        channel = [0,1,2,3,4,5,6],
        overlap = [0.1, 0.2, 0.3, 0.4, 0.5],
        raw_dir = '/path/to/LibriSpeech/filelist-all/',
        noise_dir = '/path/to/noise/',
        sample_rate = 16000,
        use_aneconic = False,
        channel_permute = False,
        normalize = False,
    ):
        super(Dataset, self).__init__()
        self.reverb_matrixs_dir = reverb_matrixs_dir
        self.rirNO = rirNO
        self.trainingNO = trainingNO
        self.segment = segment
        self.channel = channel
        self.overlap = overlap
        self.raw_dir = raw_dir
        self.noise_list = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if '.wav' in f]
        self.sample_rate = sample_rate
        self.use_aneconic = use_aneconic
        self.channel_permute = channel_permute
        self.normalize = normalize

    def __len__(self):
        return self.trainingNO
      
      
    def add_reverb(self,raw_dir1,raw_dir2,raw_dir3,h_use):
        with open(raw_dir1,'r') as fin1:
            with open(raw_dir2,'r') as fin2:
                with open(raw_dir3,'r') as fin3:
                    wav1 = fin1.readlines()
                    wav2 = fin2.readlines()
                    wav3 = fin3.readlines()
                    mix_location = np.random.choice(['front','end','both'], size=1, replace=False)
                    choose_wav = True
                    while(choose_wav):
                        i = np.random.randint(0,len(wav1))
                        j = np.random.randint(0,len(wav2))
                        k = np.random.randint(0,len(wav3))
                        w1,fs = sf.read(os.path.join('/path/to/LibriSpeech', wav1[i].rstrip("\n")), dtype="float32")
                        w2,fs = sf.read(os.path.join('/path/to/LibriSpeech', wav2[j].rstrip("\n")), dtype="float32")
                        w3,fs = sf.read(os.path.join('/path/to/LibriSpeech', wav3[k].rstrip("\n")), dtype="float32")

                        if mix_location == 'front' or mix_location == 'end':
                            overlap = np.random.choice(self.overlap)
                            if (overlap == 0.0):
                                single_speaker = 1
                            else:
                                single_speaker = 0
                            seg_len1 = int(fs * self.segment)
                            seg_len2 = int(fs * overlap * self.segment)
                            if (w1.shape[0] > seg_len1 + 1 and w2.shape[0] > seg_len2 + 1):
                                choose_wav = False

                            mix_name = 'overlap' + str(overlap) + '_' + os.path.basename(raw_dir1)[:-4] + '-' + os.path.basename(raw_dir2)[:-4] + '.wav'

                        elif mix_location == 'both':
                            overlap1 = np.random.choice([0.1, 0.2, 0.3, 0.4])
                            overlap2 = np.random.choice([0.1, 0.2, 0.3, 0.4])
                            seg_len1 = int(fs * self.segment)
                            seg_len2 = int(fs * overlap1 * self.segment)
                            seg_len3 = int(fs * overlap2 * self.segment)
                            single_speaker = 0
                            if (w1.shape[0] > seg_len1 + 1 and w2.shape[0] > seg_len2 + 1 and w3.shape[0] > seg_len3 + 1):
                                choose_wav = False

                            mix_name='overlap' + str(overlap1) + '_' + str(overlap2) + '_' + os.path.basename(raw_dir1)[:-4] + '-' + os.path.basename(raw_dir2)[:-4] + '.wav'
                            
                    rand_start1 = np.random.randint(0, w1.shape[0] - seg_len1)
                    rand_start2 = np.random.randint(0, w2.shape[0] - seg_len2)
                    stop1 = int(rand_start1 + seg_len1)
                    stop2 = int(rand_start2 + seg_len2)

                    if mix_location == 'both':
                        rand_start3 = np.random.randint(0, w3.shape[0] - seg_len3)
                        stop3 = int(rand_start3 + seg_len3)
                        
                    if (self.use_aneconic):
                        #print('Using aneconic...')
                        h_ely = h_use.copy()
                        for oneid in range(h_ely.shape[0]):
                            start_inx=(np.abs(h_ely[oneid,0])>(np.abs(h_ely[oneid,0]).max()/10.0)).argmax()
                            end_inx=start_inx+self.sample_rate//1000*50
                            h_ely[oneid,:,end_inx:]=0.0

                        w1_con = convolve(w1, h_ely[0,:,:]).T
                        w2_con = convolve(w2, h_ely[1,:,:]).T
                        w3_con = convolve(w3, h_ely[2,:,:]).T

                    else:
                        w1_con = convolve(w1, h_use[0,:,:]).T
                        w2_con = convolve(w2, h_use[1,:,:]).T
                        w3_con = convolve(w3, h_use[2,:,:]).T

                    # dynamic SIR
                    SIR1 = random.uniform(-5,5)
                    scalar1=get_amplitude_scaling_factor(w1_con, w2_con, snr = SIR1)
                    w2_con = w2_con / scalar1

                    SIR2 = random.uniform(-5,5)
                    scalar2=get_amplitude_scaling_factor(w1_con, w3_con, snr = SIR2)
                    w3_con = w3_con / scalar2

                    if (mix_location == 'front'):
                        mix_reverb = np.concatenate([w1_con[rand_start1:rand_start1 + seg_len2] + w2_con[rand_start2:stop2], \
                                                     w1_con[rand_start1 + seg_len2:stop1]], axis=0)

                        s1_reverb = w1_con[rand_start1:stop1]
                        s2_reverb = np.concatenate([w2_con[rand_start2:stop2], np.zeros_like(w1_con[rand_start1 + seg_len2:stop1])], axis=0)

                    if (mix_location == 'end'):
                        mix_reverb = np.concatenate([w1_con[rand_start1:rand_start1 + seg_len1 - seg_len2], \
                                                     w1_con[rand_start1 + seg_len1 - seg_len2:rand_start1 + seg_len1] + w2_con[rand_start2:stop2]], axis=0)

                        s1_reverb = w1_con[rand_start1:stop1]
                        s2_reverb = np.concatenate([np.zeros_like(w1_con[rand_start1:rand_start1 + seg_len1 - seg_len2]), w2_con[rand_start2:stop2]], axis=0)

                    if (mix_location == 'both'):
                        mix_reverb = np.concatenate([w1_con[rand_start1:rand_start1 + seg_len2] + w2_con[rand_start2:stop2], \
                                                     w1_con[rand_start1 + seg_len2:stop1 - seg_len3], \
                                                     w1_con[stop1 - seg_len3:stop1] + w3_con[rand_start3:stop3]], axis=0)
                        s1_reverb = w1_con[rand_start1:stop1]
                        s2_reverb = np.concatenate([w2_con[rand_start2:stop2], \
                                                    np.zeros_like(w1_con[rand_start1 + seg_len2:stop1 - seg_len3]), \
                                                    w3_con[rand_start3:stop3]], axis=0)

        return mix_reverb, s1_reverb, s2_reverb, mix_name, single_speaker
    
    def add_noise(self, mix_reverb):
        # dynamic SNR
        SNR = random.uniform(5,20)
        if(random.uniform(0,1)<0.1):
            w_n = np.random.randn(*mix_reverb.shape)
        else:
            w_n = sf.read(random.choice(self.noise_list), dtype="float32")[0]
            start_inx = random.randint(0,w_n.shape[0]-mix_reverb.shape[0]-1)
            w_n = w_n[start_inx:start_inx+mix_reverb.shape[0],0:mix_reverb.shape[-1]]
        scalar = get_amplitude_scaling_factor(mix_reverb[:,0], w_n[:,0], snr = SNR)

        mix_noise = mix_reverb + w_n / scalar
        return mix_noise
    
    def __getitem__(self,idx):
        raw_list = os.listdir(self.raw_dir)
        SpeakerNo = len(raw_list)

        speaker1 = np.random.randint(0,SpeakerNo)
        speaker2 = np.random.randint(0,SpeakerNo)
        speaker3 = np.random.randint(0,SpeakerNo)
        while (speaker1 == speaker2):
            speaker2 = np.random.randint(0,SpeakerNo)
        while (speaker3 == speaker1 or speaker3 == speaker2):
            speaker3 = np.random.randint(0,SpeakerNo)
        raw_dir1 = self.raw_dir+raw_list[speaker1]
        raw_dir2 = self.raw_dir+raw_list[speaker2]
        raw_dir3 = self.raw_dir+raw_list[speaker3]

        choose_rir = np.random.randint(0,self.rirNO)
        rand_rir = np.load(self.reverb_matrixs_dir + str(choose_rir).zfill(5) + '.npz')
        h_use, _source_positions, _sensor_positions = rand_rir['h'], rand_rir['source_positions'], rand_rir['sensor_positions']

        # step1:add reverb to utterance
        mix_reverb, s1_reverb, s2_reverb, mix_name, single_speaker = self.add_reverb(raw_dir1,raw_dir2,raw_dir3,h_use)

        # step2:add noise
        mix_noise = self.add_noise(mix_reverb)
        mix_noise = mix_noise.transpose()[self.channel]

        # choose reference channel
        source_arrays = []
        if (self.channel_permute):
            #print('Using channel permutation...')
            ref_channel = np.random.randint(0, len(self.channel))
            # s1_reverb [n c]
            source_arrays.append(np.concatenate((s1_reverb.T[ref_channel:], s1_reverb.T[:ref_channel]), axis=0))
            source_arrays.append(np.concatenate((s2_reverb.T[ref_channel:], s2_reverb.T[:ref_channel]), axis=0))
            mixture = np.concatenate((mix_noise[ref_channel:], mix_noise[:ref_channel]), axis=0)

        else:
            source_arrays.append(s1_reverb.T[self.channel])
            source_arrays.append(s2_reverb.T[self.channel])
            mixture = mix_noise
            
        # [s c n]
        sources = torch.from_numpy(np.stack(source_arrays, axis=0).astype(np.float32))
        # [c n]
        mixture = torch.from_numpy(np.array(mixture).astype(np.float32))

        # 2022.04.06
        # normalization
        if (self.normalize):
            print('Using normalization...')
            # [c n]
            m_std = mixture.std(-1, keepdim=True)
            # [c n]
            mixture = normalize_tensor_wav(mixture, eps=EPS, std=m_std)
            # [s n]
            sources = normalize_tensor_wav(sources, eps=EPS, std=m_std[[ref_channel]])

        # mixture [c n] sources [s c n]
        return mixture, sources, single_speaker
    
    
if __name__ == "__main__":
    from tqdm import tqdm

    base_dir = 'path/to/testset'
    mix_reverb = os.path.join(base_dir, 'mix_reverb')
    s1_reverb = os.path.join(base_dir, 's1_reverb')
    s2_reverb = os.path.join(base_dir, 's2_reverb')

    dir_list = [mix_reverb, s1_reverb, s2_reverb,]
    for item in dir_list:
        try:
            os.makedirs(item)
        except OSError:
            pass

    d = Dataset(
            reverb_matrixs_dir = '/path/to/reverb-set/',
            rirNO = 10000,
            trainingNO = 1,
            segment = 6,
            channel = [0,1,2,3,4,5,6],
            )

    pbar = tqdm(range(10))
    for i in pbar:
        mix, src, _, _ = d[i]
        sf.write(os.path.join(mix_reverb,'{}.wav'.format(i)),mix[0].numpy(),16000)
        sf.write(os.path.join(s1_reverb,'{}.wav'.format(i)),src[0,0,:].numpy().transpose(),16000)
        sf.write(os.path.join(s2_reverb,'{}.wav'.format(i)),src[1,0,:].numpy().transpose(),16000)

    print('Done.')
