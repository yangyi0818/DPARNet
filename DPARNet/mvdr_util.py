import json
import os
import torch
from torch import nn
from sklearn.cluster import KMeans
from asteroid import torch_utils
import asteroid.filterbanks as fb
from asteroid.engine.optimizers import make_optimizer
from asteroid.masknn import norms, activations
from asteroid.utils.torch_utils import pad_x_to_y
from asteroid.losses import PITLossWrapper, pairwise_neg_snr
from torch_complex.tensor import ComplexTensor
from torch_complex import functional as FC
from distutils.version import LooseVersion

stft_dict={
    'n_filters': 4096,
    'kernel_size': 4096,
    'stride': 1024,
}

class STFT(nn.Module):
    def __init__(self,stft_dict=stft_dict):
        super().__init__()
        self.stft_dict=stft_dict
        enc, dec = fb.make_enc_dec('stft', **stft_dict)
        self.enc = enc
        self.dec = dec

    def stft(self,x):
        # x should be  ... , t
        tf = self.enc(x.contiguous())
        # ..., F, T
        return tf

    def istft(self,x,y=None):
        # x ...,f,t
        x=self.dec(x)
        if(y is not None):
            x=torch_utils.pad_x_to_y(x,y)
        return x
      
def get_causal_power_spectral_density_matrix(observation, normalize=False, causal=False):
    '''
    psd = np.einsum('...dft,...eft->...deft', observation, observation.conj()) # (..., sensors, sensors, freq, frames)
    if normalize:
        psd = np.cumsum(psd, axis=-1)/np.arange(1,psd.shape[-1]+1,dtype=np.complex64)
    if(psd.shape[-1]%causal_step==0):
        return psd[...,causal_step-1::causal_step]
    else:
        return np.concatenate([psd[...,causal_step-1::causal_step], psd[...,[-1]]],-1)
    '''
    obsr, obsi = observation.chunk(2,-2) # S C F T
    psdr = torch.einsum('saft,sbft->sabft',obsr,obsr) + torch.einsum('saft,sbft->sabft',obsi,obsi)
    psdi = -torch.einsum('saft,sbft->sabft',obsr,obsi) + torch.einsum('saft,sbft->sbaft',obsr,obsi)
    if causal:
        psd = torch.cat([psdr,psdi],-2).cumsum(-1) # S C C F T
        if(normalize):
            psd = psd/torch.arange(1,psd.shape[-1]+1,1,dtype=psd.dtype, device=psd.device)[None,None,None,None,:]
    else:
        psd = torch.cat([psdr,psdi],-2).sum(-1,keepdim=True) # S C C F T
        if(normalize):
            pad = psd/psdr.shape[-1]
    return psd
  
def get_mvdr_vector(
    psd_s: ComplexTensor,
    psd_n: ComplexTensor,
    reference_vector = 0,
    use_torch_solver: bool = True,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
) -> ComplexTensor:
    """Return the MVDR (Minimum Variance Distortionless Response) vector:

        h = (Npsd^-1 @ Spsd) / (Tr(Npsd^-1 @ Spsd)) @ u

    Reference:
        On optimal frequency-domain multichannel linear filtering
        for noise reduction; M. Souden et al., 2010;
        https://ieeexplore.ieee.org/document/5089420

    Args:
        psd_s (ComplexTensor): speech covariance matrix (..., F, C, C)
        psd_n (ComplexTensor): observation/noise covariance matrix (..., F, C, C)
        reference_vector (torch.Tensor): (..., C)
        use_torch_solver (bool): Whether to use `solve` instead of `inverse`
        diagonal_loading (bool): Whether to add a tiny term to the diagonal of psd_n
        diag_eps (float):
        eps (float):
    Returns:
        beamform_vector (ComplexTensor): (..., F, C)
    """  # noqa: D400
    if diagonal_loading:
        psd_n = tik_reg(psd_n, reg=diag_eps, eps=eps)
    '''
    if use_torch_solver and is_torch_1_1_plus:
        # torch.solve is required, which is only available after pytorch 1.1.0+
        numerator = FC.solve(psd_s, psd_n)[0]
    else:
    '''
    numerator = FC.matmul(psd_n.inverse2(), psd_s)
    # ws: (..., C, C) / (...,) -> (..., C, C)
    ws = numerator / (FC.trace(numerator)[..., None, None] + eps)
    # h: (..., F, C_1, C_2) x (..., C_2) -> (..., F, C_1)
    beamform_vector = ws
    return beamform_vector
  
def tik_reg(mat: ComplexTensor, reg: float = 1e-8, eps: float = 1e-8) -> ComplexTensor:
    """Perform Tikhonov regularization (only modifying real part).

    Args:
        mat (ComplexTensor): input matrix (..., C, C)
        reg (float): regularization factor
        eps (float)
    Returns:
        ret (ComplexTensor): regularized matrix (..., C, C)
    """
    # Add eps
    C = mat.size(-1)
    eye = torch.eye(C, dtype=mat.dtype, device=mat.device)
    shape = [1 for _ in range(mat.dim() - 2)] + [C, C]
    eye = eye.view(*shape).repeat(*mat.shape[:-2], 1, 1)
    with torch.no_grad():
        epsilon = FC.trace(mat).real[..., None, None] * reg
        # in case that correlation_matrix is all-zero
        epsilon = epsilon + eps
    mat = mat + epsilon * eye
    return mat
  
class MVDR(nn.Module):
    def __init__(self, causal):
        super().__init__()
        self.stft_model = STFT()
        self.causal = causal
        self.permute = PITLossWrapper(pairwise_neg_snr, pit_from='pw_mtx')

    def forward(self, x, s, do_permute=True):
        """
        x: mix  b x c x n
        s: est  b x s x c x n
        """
        n_batch, n_src, n_chan, n_samp = s.shape
        if (do_permute):
            s = self.permute_sig(s)
        x = x.unsqueeze(1).repeat(1,n_src,1,1).view(n_batch*n_src, n_chan, n_samp)
        s = s.view(n_batch*n_src, n_chan, n_samp)

        X = self.stft_model.stft(x) # B*S C F T
        S = self.stft_model.stft(s) # B*S C F T
        N = X - S
        
        n_freq, n_frame = S.shape[-2:]

        # print('N ', N.shape)
        Sscm = get_causal_power_spectral_density_matrix(S, normalize=True, causal=self.causal) # B*S C C F T 
        Nscm = get_causal_power_spectral_density_matrix(N, normalize=True, causal=self.causal) # B*S C C F T

        # print('N maxtrix ', N.shape)
        Sscm = ComplexTensor(*Sscm.chunk(2,-2)).permute(0,4,3,1,2) # B*S T F C C
        Nscm = ComplexTensor(*Nscm.chunk(2,-2)).permute(0,4,3,1,2)
        est_filt = get_mvdr_vector(Sscm, Nscm) # B*S T F C C
        est_filt = torch.cat([est_filt.real,est_filt.imag],2) # B*S T F C C
        est_filt = est_filt.permute(0,3,4,2,1) # B*S C C F T
        # print('est_filt ', est_filt.shape)

        est_S = self.apply_bf(est_filt,X) # B*S C F T
        est_s = torch_utils.pad_x_to_y(self.stft_model.istft(est_S), s).view(n_batch, n_src, n_chan, n_samp) # b*s c t
        s = s.view(n_batch, n_src, n_chan, n_samp)

        return est_s
      
    def apply_bf(self,f,X):
        '''
            f B C C F T
            X B C F T
        '''
        X_real, X_imag = X.unsqueeze(2).chunk(2,-2) # B C 1 F T
        f_real, f_imag = f.chunk(2,-2)
        f_imag = -1.0 * f_imag
        # enhX_real = (X_real * (f_real + torch.ones_like(f_real))).sum(1) - (X_imag * f_imag).sum(1) # B C F T
        # enhX_imag = (X_real * f_imag).sum(1) + (X_imag * (f_real + torch.ones_like(f_real))).sum(1)
        enhX_real = (X_real * f_real).sum(1) - (X_imag * f_imag).sum(1) # B C F T
        enhX_imag = (X_real * f_imag).sum(1) + (X_imag * f_real).sum(1)
        enhX = torch.cat([enhX_real, enhX_imag],2)
        return enhX
      
    def permute_sig(self, est_sources):
        # b s c t
        reest_sources = [est_sources[:,:,0,:],]
        for chan in range(1,est_sources.shape[2]):
            if(self.causal):
                est_sources_rest = torch.zeros_like(est_sources[:,:,chan,:])
                if(est_sources.shape[-1]<self.stft_dict['kernel_size']):
                    reest_sources.append(self.permute(est_sources[:,:,chan,:], est_sources[:,:,0,:], return_est=True)[1])
                else:
                    est_sources_rest[:,:,0:self.stft_dict['kernel_size']] = self.permute(est_sources[:,:,chan,0:self.stft_dict['kernel_size']], \
                        est_sources[:,:,0,0:self.stft_dict['kernel_size']], return_est=True)[1]
                    for starti in range(self.stft_dict['kernel_size'], est_sources.shape[-1], self.stft_dict['stride']):
                        endi = min(starti+self.stft_dict['stride'],est_sources.shape[-1])
                        est_sources_rest[:,:,starti:endi] = self.permute(est_sources[:,:,chan,0:endi], \
                            est_sources[:,:,0,0:endi], return_est=True)[1][:,:,starti:endi]
                    reest_sources.append(est_sources_rest)
            else:
                reest_sources.append(self.permute(est_sources[:,:,chan,:], est_sources[:,:,0,:], return_est=True)[1])
        return torch.stack(reest_sources,2)
      
      
