import os
import sys
import json
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import torch
import random

import time
from sms_wsj.database.create_rirs import config, scenarios, rirs
from sms_wsj.reverb.reverb_utils import convolve

    
def generate_rir(i):
    _worker_init_fn_(i)
    reverb_matrixs_dir = '/path/to/reverb-set/'
    geometry, sound_decay_time_range, sample_rate, filter_length = config()
    room_dimensions, source_positions, sensor_positions, sound_decay_time = scenarios(geometry, sound_decay_time_range,)
    h = rirs(sample_rate, filter_length, room_dimensions, source_positions, sensor_positions, sound_decay_time)
    np.savez(reverb_matrixs_dir + str(i).zfill(5) + '.npz', h=h, source_positions=source_positions, sensor_positions=sensor_positions,)
   

def main(conf):
    reverb_matrixs_dir = '/path/to/reverb-set/'
    generate_NO = 10000

    # generate new rirs
    if not os.path.exists(reverb_matrixs_dir):
        os.makedirs(reverb_matrixs_dir)
    else:
        if (input('target dir already esists, continue? [y/n]  ') == 'n'):
            print('Exit. Nothing happends.')
            sys.exit()
    print('Generating reverb matrixs into ', reverb_matrixs_dir, '......')
    '''
    # single process
    pbar = tqdm(range(generate_NO))
    for i in pbar:
    generate_rir(i, reverb_matrixs_dir)
    '''
    # multi process
    time_start=time.time()
    pool = Pool(processes=32)
    args = []
    for i in range (generate_NO):
        args.append(i)
    pool.map(generate_rir, args)
    pool.close()
    pool.join()
    time_end=time.time()
    print('totally cost ', round((time_end-time_start)/60), 'minutes')
