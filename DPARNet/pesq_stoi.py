import sys
import os
import numpy as np
from tqdm import tqdm
import soundfile as sf

from pesq import pesq
from pystoi import stoi
from mir_eval.separation import bss_eval_sources

def eval_pesq_stoi1(ref, i):
    mix_wav = os.listdir(os.path.join(ref, 'mix_reverb'))
    src_mix, _ = sf.read(os.path.join(ref, 'mix_reverb', mix_wav[i]))
    if len(src_mix.shape) == 2:
        src_mix = src_mix[:,0]

    src_ref1, fs = sf.read(os.path.join(ref, 's1_reverb', mix_wav[i]))
    pesq_score1 = pesq(fs, src_ref1, src_mix, 'wb')
    stoi_score1 = stoi(src_ref1, src_mix, fs, extended=False)

    src_ref2, fs = sf.read(os.path.join(ref, 's2_reverb', mix_wav[i]))
    pesq_score2 = pesq(fs, src_ref2, src_mix, 'wb')
    stoi_score2 = stoi(src_ref2, src_mix, fs, extended=False)

    pesq_score_, stoi_score_ = pesq_score1+pesq_score2, stoi_score1+stoi_score2

    return pesq_score_, stoi_score_
  
def eval_pesq_stoi2(ref, separated, i):
    sep_wav = os.listdir(separated)
    src_est, fs = sf.read(os.path.join(separated, sep_wav[i]))
    if len(src_est.shape) == 2:
        src_est = src_est[:,0]

    src_ref1, _ = sf.read(os.path.join(ref, 's1_reverb', sep_wav[i].rsplit('_', 1)[0]+'.wav'))
    src_ref2, _ = sf.read(os.path.join(ref, 's2_reverb', sep_wav[i].rsplit('_', 1)[0]+'.wav'))

    pesq_score_ = max(pesq(fs, src_ref1, src_est, 'wb'), pesq(fs, src_ref2, src_est, 'wb'))
    stoi_score_ = max(stoi(src_ref1, src_est, fs, extended=False), stoi(src_ref2, src_est, fs, extended=False))

    return pesq_score_, stoi_score_
  
def eval_sdri(ref, separated, i):
    mix_wav = os.listdir(os.path.join(ref, 'mix_reverb'))
    src_mix, _ = sf.read(os.path.join(ref, 'mix_reverb', mix_wav[i]))
    if len(src_mix.shape) == 2:
        src_mix = src_mix[:,0]

    src_ref1, _ = sf.read(os.path.join(ref, 's1_reverb', mix_wav[i]))
    src_ref2, _ = sf.read(os.path.join(ref, 's2_reverb', mix_wav[i]))

    src_est1, _ = sf.read(os.path.join(separated,  mix_wav[i].rsplit('.', 1)[0]+'_0.wav'))
    src_est2, _ = sf.read(os.path.join(separated,  mix_wav[i].rsplit('.', 1)[0]+'_1.wav'))

    src_mix = np.stack([src_mix, src_mix], axis=0)
    src_ref = np.stack([src_ref1, src_ref2], axis=0)
    src_est = np.stack([src_est1, src_est2], axis=0)

    sdr, sir, sar, popt = bss_eval_sources(src_ref, src_est)
    sdr0, sir0, sar0, popt0 = bss_eval_sources(src_ref, src_mix)

    avg_SDRi = ((sdr[0]-sdr0[0]) + (sdr[1]-sdr0[1])) / 2
    sdri1, sdri2 = sdr[0]-sdr0[0], sdr[1]-sdr0[1]

    return sdri1, sdri2
  
if __name__ == "__main__":
    pesq_score, stoi_score, sdri_score = 0, 0, 0

    separated, ref = sys.argv[1], sys.argv[2]

    # 原始语音的pesq和stoi
    """
    pbar = tqdm(range(len(os.listdir(os.path.join(ref, 'mix_reverb')))))
    for i in pbar:
        pesq_score_, stoi_score_ = eval_pesq_stoi1(ref, i)
        pesq_score += pesq_score_
        stoi_score += stoi_score_
        
        pbar.set_description("pesq: {:.3f}, stoi: {:.3f}".format(pesq_score/2/(i+1), stoi_score/2/(i+1)))
    """

    pbar = tqdm(range(len(os.listdir(separated))))
    for i in pbar:
        pesq_score_, stoi_score_ = eval_pesq_stoi2(ref, separated, i)
        pesq_score += pesq_score_
        stoi_score += stoi_score_

        #sdri1, sdri2 = eval_sdri(ref, separated, i)
        #sdri += sdri1
        #sdri += sdri2

        pbar.set_description("pesq: {:.3f}, stoi: {:.3f}".format(pesq_score/(i+1), stoi_score/(i+1)))
        #pbar.set_description("pesq: {:.3f}, stoi: {:.3f}, sdri: {:.3f}".format(pesq_score/(i+1), stoi_score/(i+1), sdri_score/2/(i+1)))
