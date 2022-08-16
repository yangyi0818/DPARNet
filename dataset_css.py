import torch
from torch.utils import data
import json
import os
import numpy as np
import soundfile as sf

EPS = 1e-8

DATASET = "xiandao2020"
sep_clean = {"mixture": "mix_noise", "sources": "each_spk", "infos": [], "default_nsrc": 2}

xiandao2020_TASKS = {"sep_clean": sep_clean,}

def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    #return (wav_tensor - mean) / (std + eps)
    wav_tensor =wav_tensor - mean
    return wav_tensor / (wav_tensor.max() + EPS)


class XiandaoDataset(data.Dataset):

    dataset_name = "xiandao2020"

    def __init__(
        self,
        json_dir,
        task,
        sample_rate=16000,
        segment=5,
        nondefault_nsrc=None,
        normalize_audio=False,
        channel = [0]
    ):
        super(XiandaoDataset, self).__init__()
        if task not in xiandao2020_TASKS.keys():
            raise ValueError(
                "Unexpected task {}, expected one of " "{}".format(task, xiandao2020_TASKS.keys())
            )
        # Task setting
        self.json_dir = json_dir
        self.task = task
        self.task_dict = xiandao2020_TASKS[task]
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio
        self.seg_len = None if segment is None else int(segment * sample_rate)
        self.channel = channel
        if not nondefault_nsrc:
            self.n_src = self.task_dict["default_nsrc"]
        else:
            assert nondefault_nsrc >= self.task_dict["default_nsrc"]
            self.n_src = nondefault_nsrc
        self.like_test = self.seg_len is None
        # Load json files
        mix_json = os.path.join(json_dir, self.task_dict["mixture"] + ".json")
        sources_json = os.path.join(json_dir, self.task_dict["sources"] + ".json")

        with open(mix_json, "r") as f:
            mix_infos = json.load(f)
        with open(sources_json, "r") as f:
            sources_infos = json.load(f)
            
        # Filter out short utterances only when segment is specified
        orig_len = len(mix_infos)
        drop_utt, drop_len = 0, 0
        if not self.like_test:
            for i in range(len(mix_infos) - 1, -1, -1):  # Go backward
                if mix_infos[i][1] < self.seg_len:
                    drop_utt += 1
                    drop_len += mix_infos[i][1]
                    del mix_infos[i]
                    for src_inf in sources_infos:
                        del src_inf[i]

        print(
            "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                drop_utt, drop_len / sample_rate / 36000, orig_len, self.seg_len
            )
        )
        self.mix = mix_infos
        # Handle the case n_src > default_nsrc
        while len(sources_infos) < self.n_src:
            sources_infos.append([None for _ in range(len(self.mix))])
        self.sources = sources_infos

    def __add__(self, xiandao):
        if self.n_src != xiandao.n_src:
            raise ValueError(
                "Only datasets having the same number of sources"
                "can be added together. Received "
                "{} and {}".format(self.n_src, xiandao.n_src)
            )
        if self.seg_len != xiandao.seg_len:
            self.seg_len = min(self.seg_len, xiandao.seg_len)
            print(
                "Segment length mismatched between the two Dataset"
                "passed one the smallest to the sum."
            )
            
        self.mix = self.mix + xiandao.mix
        self.sources = [a + b for a, b in zip(self.sources, xiandao.sources)]

    def __len__(self):
        return len(self.mix)

    def __getitem__(self, idx):

        # Random start
        if self.mix[idx][1] == self.seg_len or self.like_test:
            rand_start = 0
        else:
            rand_start = np.random.randint(0, self.mix[idx][1] - self.seg_len)
        if self.like_test:
            stop = None
        else:
            stop = rand_start + self.seg_len

        # Load mixture
        x, _ = sf.read(self.mix[idx][0], start=rand_start, stop=stop, dtype="float32")
        x = x.transpose()[self.channel]

        base_name = os.path.basename(self.mix[idx][0])
        dir_name = os.path.basename(os.path.dirname(self.mix[idx][0]))
        name = dir_name + '/' + base_name
        seg_len = torch.as_tensor([len(x)])

        # Load sources
        s, _ = sf.read(self.sources[idx][0], start=rand_start, stop=stop, dtype="float32")
        s = s.T

        sources = torch.from_numpy(np.array(s).astype(np.float32))
        mixture = torch.from_numpy(np.array(x).astype(np.float32)).permute(1,0)

        if self.normalize_audio:
            m_std = mixture.std(-1, keepdim=True)
            mixture = normalize_tensor_wav(mixture, eps=EPS, std=m_std)
            sources = normalize_tensor_wav(sources, eps=EPS, std=m_std)

        return mixture, sources, name
