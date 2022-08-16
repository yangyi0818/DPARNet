# DPARNet
**Light-weight speech separation based on dual-path attention and recurrent neural network**

**基于双路注意力循环网络的轻量化语音分离 [1]**

## Contents 
* **[DPARNet](#dparnet)**
  * **[Contents](#contents)**
  * **[Introduction](#introduction)**
  * **[Dataset](#dataset)**
  * **[Requirement](#requirement)**

## Introduction
**DPARNet, which is an improvement of DPTFSNet [2], is composed of encoder, separation network and decoder. To alleviate the computation burden, sub-band processing approach is leveraged in the encoder. Dual-path attention mechanism and recurrent network structure are introduced in the separation network to model the speech signals in each sub-band, which facilitate extraction of deep feature information and rich spectrum details.**

**The parameters and computation cost of DPARNet model is only 0.15M and 15.2G/6s.**

**Inspired by [3], we also introduce Beam-Guided DPARNet, which makes full use of spatial information.**

## Dataset
**We use [sms_wsj][sms_wsj] to generate reverb set. 'sms_wsj/reverb/scenario.py' and 'sms_wsj/database/create_rirs.py' should be replaced by scripts in 'sms_wsj_replace' folder.**

**use 'python generate_rir.py' to generate training and valadation data**

**We use LibriCSS dataset as test set, which can be found [here][libricss].**

## Requirement
**Our script use [asteroid][asteroid] as the basic framework.** 

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

[libricss]: https://github.com/chenzhuo1011/libri_css
[asteroid]: https://github.com/asteroid-team/asteroid
[sms_wsj]: https://github.com/fgnt/sms_wsj


