import os
import sys
import argparse
import json
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import torch
import random

import time
from sms_wsj.database.create_rirs import config, scenarios, rirs
from sms_wsj.reverb.reverb_utils import convolve

parser = argparse.ArgumentParser()
parser.add_argument("--generate_NO", type=int, default=300, help="generate new rirs NO")

def _worker_init_fn_(worker_id):
    torch_seed = torch.initial_seed()

    random.seed(torch_seed + worker_id)
    if torch_seed >= 2**32:
        torch_seed = torch_seed % 2**32
    np.random.seed(torch_seed + worker_id)
    
def generate_rir(i):
    src_position = '3d'    # choose between '2d','3d','Gaussian'
    _worker_init_fn_(i)
    reverb_matrixs_dir = '/path/to/reverb-set/'
    geometry, sound_decay_time_range, sample_rate, filter_length = config()
    room_dimensions, source_positions, sensor_positions, sound_decay_time = scenarios(geometry, sound_decay_time_range,src_position,)
    h = rirs(sample_rate, filter_length, room_dimensions, source_positions, sensor_positions, sound_decay_time)
    np.savez(reverb_matrixs_dir + str(i+10000).zfill(5) + '.npz', h=h, source_positions=source_positions, sensor_positions=sensor_positions,)
