import os
import random
import soundfile as sf
import torch
import yaml
import json
import argparse
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from pprint import pprint

from asteroid import torch_utils
from asteroid.metrics import get_metrics
from asteroid.losses import pairwise_neg_sisdr
from asteroid.losses.pit_wrapper import PITLossWrapper
from DPARNet import make_model_and_optimizer
from asteroid.utils import tensors_to_device
from mvdr_util import MVDR

from dataset_css import XiandaoDataset

parser = argparse.ArgumentParser()
parser.add_argument("--normalize", type=int, required=True, help="")
parser.add_argument("--test_dir_simu", type=str, required=True, help="Test directory")
parser.add_argument("--test_dir_css", type=str, required=True, help="Test directory")
parser.add_argument("--save_wav_simu", type=int, default=0, help="Whether to save wav files")
parser.add_argument("--save_wav_css", type=int, default=0, help="Whether to save wav files")
parser.add_argument("--save_dir_simu", type=str, required=True, help="Output directory")
parser.add_argument("--save_dir_css", type=str, required=True, help="Output directory")
parser.add_argument("--use_gpu", type=int, default=0, help="Whether to use the GPU for model execution")
parser.add_argument("--do_mvdr", type=int, default=0, help="Whether to use mvdr")
parser.add_argument("--exp_dir", default="exp/tmp", help="Experiment root")

compute_metrics = ["si_sdr"]

def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = torch.mean(wav_tensor, dim=-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


def load_best_model(model, exp_dir):
    # Create the model from recipe-local function
    try:
        # Last best model summary
        with open(os.path.join(exp_dir, 'best_k_models.json'), "r") as f:
            best_k = json.load(f)
        best_model_path = min(best_k, key=best_k.get)
    except FileNotFoundError:
        # Get last checkpoint
        all_ckpt = os.listdir(os.path.join(exp_dir, 'checkpoints/'))
        all_ckpt=[(ckpt,int("".join(filter(str.isdigit,ckpt)))) for ckpt in all_ckpt]
        all_ckpt.sort(key=lambda x:x[1])
        best_model_path = os.path.join(exp_dir, 'checkpoints', all_ckpt[-1][0])
    print( 'LOADING from ',best_model_path)
    # Load checkpoint
    checkpoint = torch.load(best_model_path, map_location='cpu')
    for k in list(checkpoint['state_dict'].keys()):
        if('loss_func' in k):
            del checkpoint['state_dict'][k]
    # Load state_dict into model.
    model = torch_utils.load_state_dict_in(checkpoint['state_dict'], model)
    model = model.eval()
    return model
  
class sisdr_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sig_loss = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')

    def forward(self, est_targets, targets):
        sig_loss, reordered_sources = self.sig_loss(est_targets, targets, return_est=True)

        return sig_loss.mean(), reordered_sources

def main1(model, conf):
    causal = False
    mvdr = MVDR(causal)
    if conf["use_gpu"]:
        model.cuda()
        mvdr.cuda()
    model_device = next(model.parameters()).device

    normalize = conf['normalize']
    test_dir_simu = conf['test_dir_simu']
    save_dir_simu = conf['save_dir_simu']
    dlist = os.listdir(test_dir_simu)
    pbar = tqdm(range(len(dlist)))
    series_list = []
    torch.no_grad().__enter__()
    for idx in pbar:
        test_wav = np.load(test_dir_simu + dlist[idx])
        mix, sources, name, single_speaker = tensors_to_device([torch.from_numpy(test_wav['mix']), torch.from_numpy(test_wav['src']), \
                                                                str(test_wav['n']), test_wav['single_speaker']], device=model_device)

        mix = mix.permute(1,0) # [m n]

        if (normalize):
            m_std = mix.std(1, keepdim=True)
            mix = normalize_tensor_wav(mix, eps=1e-8, std=m_std)
            sources = normalize_tensor_wav(sources, eps=1e-8, std=m_std[[0]]) # [s n]

        est_sources_7ch = model(mix[None])   # [b s m n]
        est_sources = est_sources_7ch[:,:,0] # [b s n]
        
        loss, reordered_sources = sisdr_loss()(est_sources, sources[None])

        sources_np = sources.cpu().data.numpy()

        if conf["do_mvdr"]:
            est_sources = mvdr(mix[None], est_sources_7ch) # b s m n
            est_sources_np = est_sources.squeeze(0).cpu().data.numpy() # s m n
            est_sources_np = est_sources_np[:,0]
        else:
            est_sources_np = reordered_sources.squeeze(0).cpu().data.numpy()

        if (single_speaker == 1):
            sources_np = sources_np[[0],:]
            est_sources_np = est_sources_np[[0],:]

        mix = mix[0,:]
        mix_np = mix[None].cpu().data.numpy()
        
        # save wave
        if not os.path.exists(os.path.join(save_dir_simu, name.split('_')[0])):
            os.makedirs(os.path.join(save_dir_simu, name.split('_')[0]))
        if idx<1000:
            est_s1 = est_sources_np[0]
            est_s1 *= np.max(np.abs(mix_np.squeeze()))/np.max(np.abs(est_s1))
            est_sources_np *= np.max(np.abs(mix_np.squeeze()))/np.max(np.abs(est_sources_np))
            sf.write(os.path.join(save_dir_simu, name.split('_')[0], name[:-4]+'_0.wav'), est_s1.squeeze() / est_s1.max(), conf["sample_rate"])
            est_s2 = est_sources_np[1]
            est_s2 *= np.max(np.abs(mix_np.squeeze()))/np.max(np.abs(est_s2))
            sf.write(os.path.join(save_dir_simu, name.split('_')[0], name[:-4]+'_1.wav'), est_s2.squeeze() / est_s2.max(), conf["sample_rate"])

        utt_metrics = get_metrics(
            mix_np,
            sources_np,
            est_sources_np,
            sample_rate=conf["sample_rate"],
            metrics_list=compute_metrics,
        )
        utt_metrics["mix_path"] = name
        series_list.append(pd.Series(utt_metrics))
        pbar.set_description("si_sdr : {}".format(pd.DataFrame(series_list)['si_sdr'].mean()))
    # Save all metrics to the experiment folder.
    all_metrics_df = pd.DataFrame(series_list)
    all_metrics_df.to_csv(os.path.join(conf["exp_dir"], "all_metrics_permute.csv"))

    # Print and save summary metrics
    final_results = {}
    for metric_name in compute_metrics:
        input_metric_name = "input_" + metric_name
        ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
        final_results[metric_name] = all_metrics_df[metric_name].mean()
        final_results[metric_name + "_imp"] = ldf.mean()
    print("Overall metrics :")
    pprint(final_results)
    

def main2(model, conf):
    causal = False
    mvdr = MVDR(causal)
    if conf["use_gpu"]:
        model.cuda()
        mvdr.cuda()
    model_device = next(model.parameters()).device
    normalize = conf['normalize']
    test_dir_css = conf['test_dir_css']
    save_dir_css = conf['save_dir_css']
    test_set = XiandaoDataset(
        conf["test_dir_css"],
        'sep_clean',
        sample_rate=conf["sample_rate"],
        nondefault_nsrc=2,
        segment=None,
        channel=[0,1,2,3,4,5,6]
    )  # Uses all segment length

    series_list = []
    torch.no_grad().__enter__()
    for idx in tqdm(range(len(test_set))):
        # Forward the network on the mixture.
        mix, sources, name = tensors_to_device(test_set[idx], device=model_device)
        name = name[:-4]

        # normalization
        mix = mix.permute(1,0) # [m n]
        if (normalize):
            m_std = mix.std(1, keepdim=True)
            mix = normalize_tensor_wav(mix, eps=1e-8, std=m_std)

        est_sources, est_sources_7ch = model(mix[None], do_eval=1)

        mix_7ch = mix
        if (normalize):
            mix = mix[0,:]
        else:
            mix = mix[:,0]
            
        mix_np = mix[None].cpu().data.numpy()

        if conf["do_mvdr"]:
            est_sources = mvdr(mix_7ch[None], est_sources_7ch) # b s c n
            est_sources_np = est_sources.squeeze(0).cpu().data.numpy()
            est_sources_s1_np = est_sources_np[0,0]
            est_sources_s2_np = est_sources_np[1,0]
        else:
            est_sources_np = est_sources.squeeze(0).cpu().data.numpy()
            est_sources_s1_np = est_sources_np[0]
            est_sources_s2_np = est_sources_np[1]

        # save wave
        est_sources_np = est_sources.squeeze(0).cpu().data.numpy()
        est_wav_s1 = est_sources_s1_np * np.max(np.abs(mix_np.squeeze()))/np.max(np.abs(est_sources_s1_np))
        est_wav_s2 = est_sources_s2_np * np.max(np.abs(mix_np.squeeze()))/np.max(np.abs(est_sources_s2_np))
        if not os.path.exists(save_dir_css + os.path.dirname(name)):
            os.makedirs(save_dir_css + os.path.dirname(name))
        sf.write(save_dir_css+name+'_0.wav', est_wav_s1.squeeze() / est_wav_s1.max(), conf["sample_rate"])
        sf.write(save_dir_css+name+'_1.wav', est_wav_s2.squeeze() / est_wav_s2.max(), conf["sample_rate"])
        
        
def main(conf):
    model, _ = make_model_and_optimizer(train_conf)
    model = load_best_model(model, conf['exp_dir'])
    save_dir_simu = conf['save_dir_simu']
    save_dir_css = conf['save_dir_css']

    if (conf['save_wav_simu']):
        main1(model, conf)
    if (conf['save_wav_css']):
        main2(model, conf)


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    # Load training config
    conf_path = os.path.join(args.exp_dir, "conf.yml")
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)
    arg_dic["sample_rate"] = train_conf["data"]["sample_rate"]
    arg_dic["train_conf"] = train_conf

    main(arg_dic)
