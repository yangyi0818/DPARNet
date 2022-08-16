# DPARNet
**Light-weight speech separation based on dual-path attention and recurrent neural network**

**基于双路注意力循环网络的轻量化语音分离 [1]**

## Contents 
* **[DPARNet](#dparnet)**
  * **[Contents](#contents)**
  * **[Introduction](#introduction)**
  * **[Dataset](#dataset)**
  * **[Train](#train)**
  * **[Test](#test)**
  * **[Requirement](#requirement)**
  * **[Results](#results)**
  * **[Citation](#citation)**

## Introduction
**DPARNet, which is an improvement of DPTFSNet [2], is composed of encoder, separation network and decoder. To alleviate the computation burden, sub-band processing approach is leveraged in the encoder. Dual-path attention mechanism and recurrent network structure are introduced in the separation network to model the speech signals in each sub-band, which facilitate extraction of deep feature information and rich spectrum details.**

**The parameters and computation cost of DPARNet model is only 0.15M and 15.2G/6s.**

**Inspired by [3], we also introduce Beam-Guided DPARNet, which makes full use of spatial information.**

## Dataset
**We use [sms_wsj][sms_wsj] to generate reverb set. ```sms_wsj/reverb/scenario.py``` and ```sms_wsj/database/create_rirs.py``` should be replaced by scripts in 'sms_wsj_replace' folder.**

**use ```python generate_rir.py``` to generate training and valadation data**

**We use LibriCSS dataset as test set, which can be found [here][libricss].**

## Train
**We recommend running to train end-to-end :**

```./run.sh --id 0,1,2,3```

or :

```./run.sh --id 0,1,2,3 --stage 1```

## Test
```./run.sh --id 0 --stage 2```

## Requirement
**Our script use [asteroid][asteroid] as the basic framework.** 

## Results
**WER (%) on LibriCSS**

|**Model|Year|0S|0L|OV10|OV20|OV30|OV40|
| :-----| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|Raw[4]|2020|11.8|11.7|18.8|27.2|35.6|43.3|
|BLSTM[5]|2021|7.0|7.5|10.8|13.4|16.5|18.8|12.3|
|Conformer-large[5]|2021|7.2|7.5|9.6|11.3|13.7|15.1|
|DPT-FSNet[2]| 2022 |7.1| 7.3 |7.6| 8.9| 10.8| 11.3|
|Beam-Guided TasNet[3] (single stage)| 2022| 7.3 |7.3 |7.8 |8.9 |10.6| 11.1 |
|Beam-Guided TasNet[3] (two stages)|2022| 7.1 |7.1 |7.1 |8.0| 9.2| 9.7 |
|proposed DPARNet |- |7.2| 7.2| 7.4 |8.6 |10.3| 10.9|
|Beam-Guided DPARNet| -| 7.3 |6.9 |7.2| 7.7 |9.0| 9.4 **|


## Citation
**[1] Yang Y, Hu Q, Zhang P Y. Light-weight speech separation based on dual-path attention and recurrent neural network** 

**杨弋，胡琦，张鹏远. 基于双路注意力循环网络的轻量化语音分离**

**[2] Dang F, Chen H T, Zhang P Y. DPT-FSNet: Dual-path Transformer Based Full-band and Sub-band Fusion Network for Speech Enhancement. Proc. IEEE
Int. Conf. Acoust. Speech Signal Process., 2022: 6857—6861**

**[3] Chen H T, Zhang P Y. Beam-Guided TasNet: An Iterative Speech Separation Framework with Multi-Channel Output, 2021: arXiv preprint arXiv:
2102.02998**

**[4] Chen Z, Yoshioka T, Lu L et al. Continuous speech separation: dataset and analysis. Proc. IEEE Int. Conf. Acoust. Speech Signal Process., 2020:
7284—7288**

**[5] Chen S Y, Wu Y, Chen Z et al. Continuous Speech Separation with Conformer. Proc. IEEE Int. Conf. Acoust. Speech Signal Process., 2021; 5749—5753**

**This paper has been submitted to *Chinese Journal of Acoustics*.**

**Please feel free to contact us if you have any questions.**

[libricss]: https://github.com/chenzhuo1011/libri_css
[asteroid]: https://github.com/asteroid-team/asteroid
[sms_wsj]: https://github.com/fgnt/sms_wsj


