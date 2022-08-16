# DPARNet
**Light-weight speech separation based on dual-path attention and recurrent neural network**

**基于双路注意力循环网络的轻量化语音分离 [1]**

## Contents 
* [跳到1. 这是一级标题](#1-DPARNet)

## Introduction
DPARNet, which is an improvement of DPTFSNet [2], is composed of encoder, separation network and decoder. To alleviate the computation burden, sub-band processing approach is leveraged in the encoder. Dual-path attention mechanism and recurrent network structure are introduced in the separation network to model the speech signals in each sub-band, which facilitate extraction of deep feature information and rich spectrum details.

The parameters and computation cost of DPARNet model is only 0.15M and 15.2G/6s.

Inspired by [3], we also introduce Beam-Guided DPARNet, which makes full use of spatial information.

# Results
WER (%) on LibriCSS

|Model|Year|0S|0L|OV10|OV20|OV30|OV40|


# Citation
[1] Yang Y, Hu Q, Zhang P Y. Light-weight speech separation based on dual-path attention and recurrent neural network 

杨弋，胡琦，张鹏远. 基于双路注意力循环网络的轻量化语音分离

[2] Dang F, Chen H T, Zhang P Y. DPT-FSNet: Dual-path Transformer Based Full-band and Sub-band Fusion Network for Speech Enhancement. Proc. IEEE
Int. Conf. Acoust. Speech Signal Process., 2022: 6857—6861

# Note
This paper has been submitted to *Chinese Journal of Acoustics*.

Please feel free to contact us if you have any questions.
