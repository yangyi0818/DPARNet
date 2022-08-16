##!/bin/bash

# Exit on error
set -e
set -o pipefail

export PYTHONPATH=/path/to/asteroid:/path/to/sms_wsj
python=

# General
stage=1
expdir=exp
id=$CUDA_VISIBLE_DEVICES

train_dir=
val_dir=
test_dir_simu=
test_dir_css=

. utils/parse_options.sh

# Training
use_aneconic=1
channel_permute=1
normalize=0
batch_size=6
num_workers=16
optimizer=adam
lr=0.001
epochs=100

# Evaluation
eval_use_gpu=1
save_simu=1
save_css=1
do_mvdr=1
save_dir_simu=
save_dir_css=

mkdir -p $expdir
echo "Results from the following experiment will be stored in $expdir"

if [[ $stage -le 1 ]]; then
  echo "Stage 1: Training"
  mkdir -p logs
  CUDA_VISIBLE_DEVICES=$id $python -u executor1/train.py \
                --use_aneconic $use_aneconic \
                --channel_permute $channel_permute \
                --normalize $normalize \
                --train_dirs $train_dir \
                --val_dirs $val_dir \
                --sample_rate 16000 \
                --lr $lr \
                --epochs $epochs \
                --batch_size $batch_size \
                --num_workers $num_workers \
                --exp_dir ${expdir}/ | tee logs/train.log
        cp logs/train.log $expdir/train.log
        echo "Stage 1 - training: Done."
fi

if [[ $stage -le 2 ]]; then
        echo "Stage 2 : Evaluation."
        mkdir -p logs
        echo "Saving mode is ${save_simu} in ${save_dir_simu}"
        echo "Saving mode is ${save_css} in ${save_dir_css}"
        echo "MVDR state is ${do_mvdr}"
        [[ ! -d $save_dir_simu ]] && mkdir -p $save_dir_simu
        [[ ! -d $save_dir_css ]] && mkdir -p ${save_dir_css}/utterances
        CUDA_VISIBLE_DEVICES=$id $python -u executor1/eval.py \
                --normalize $normalize \
                --test_dir_simu $test_dir_simu \
                --test_dir_css $test_dir_css \
                --use_gpu $eval_use_gpu \
                --do_mvdr $do_mvdr \
                --save_wav_simu $save_simu \
                --save_wav_css $save_css \
                --save_dir_simu $save_dir_simu \
                --save_dir_css ${save_dir_css}/utterances/ \
                --exp_dir ${expdir} | tee logs/eval.log
        cp logs/eval.log $expdir/eval_simu.log
        echo "Stage 2 - evaluation: Done."
fi

if [[ $stage -le 3 ]]; then
        echo "Stage 3 : PESQ and STOI"
        $python pesq_stoi.py ${save_dir_simu} ${test_dir_simu}
        echo "Stage 3 - PESQ and STOI: Done."
fi

if [[ $stage -le 4 ]]; then
        echo "Stage 4 : ASR"
        cd /path/to/scripts
        ./run_decode_wrapper.sh ${save_dir_css} $id
fi


