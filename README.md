# DPARNet
**Light-weight speech separation based on dual-path attention and recurrent neural network**

**基于双路注意力循环网络的轻量化语音分离**

**This work has been accepted by *声学学报 (Chinese Journal of Acoustics)*.** 

## Contents 
* **[DPARNet](#dparnet)**
  * **[Contents](#contents)**
  * **[Introduction](#introduction)**
  * **[Dataset](#dataset)**
  * **[Requirement](#requirement)**
  * **[Train](#train)**
  * **[Test](#test)**
  * **[Results](#results)**
  * **[Citation](#citation)**
  * **[References](#references)**

## Introduction
**DPARNet, which is an improvement of DPTFSNet [1], is composed of encoder, separation network and decoder. To alleviate the computation burden, sub-band processing approach is leveraged in the encoder. Dual-path attention mechanism and recurrent network structure are introduced in the separation network to model the speech signals in each sub-band, which facilitate extraction of deep feature information and rich spectrum details.**

**The parameters and computation cost of DPARNet model is only 0.15M and 15.2G/6s.**

**Inspired by [2], we also introduce Beam-Guided DPARNet, which makes full use of spatial information.**

## Dataset
**We use [sms_wsj][sms_wsj] to generate room impulse responses (RIRs) set. ```sms_wsj/reverb/scenario.py``` and ```sms_wsj/database/create_rirs.py``` should be replaced by scripts in 'sms_wsj_replace' folder.**

**use ```python generate_rir.py``` to generate training and valadation data**

**We use [LibriCSS][libricss] dataset as test set.**

## Requirement
**Our script use [asteroid][asteroid] toolkit as the basic environment.**

## Train
**We recommend running to train end-to-end :**

**```./run.sh --id 0,1,2,3```**

**or :**

**```./run.sh --id 0,1,2,3 --stage 1```**

## Test
**```./run.sh --id 0 --stage 2```** 

## Results
**WER (%) on LibriCSS, model parameters (MiB) and computation (G/6s speech)**

|**Model**                   |**Year**|**0S**  |**0L**  |**OV10**|**OV20**|**OV30**|**OV40**|**parameters**|**computation**|
| :-----                     | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----:       | :----:        |
|**Raw[3]**                  |2020    |11.8    |11.7    |18.8    |27.2    |35.6    |43.3    | -            | -             |
|**BLSTM[4]**                |2021    |**7.0** |7.5     |10.8    |13.4    |16.5    |18.8    |21.8          |17.1           |
|**PW-NBDF[5]**              |2021    |7.3     |7.3     | 8.3    |10.6    |13.4    |15.8    |18.9          |20.1           |
|**Conformer-large[4]**      |2021    |7.2     |7.5     |9.6     |11.3    |13.7    |15.1    |58.7          |43.6           |
|**DPT-FSNet[1]**            |2022    |7.1     |7.3     |7.6     |8.9     |10.8    |11.3    |0.50          |49.1           |
|**Beam-Guided DPT-FSNet[2]**|2022    |7.1     |7.1     |**7.1** |8.0     |9.2     |9.7     |1.0           |50.1           |
|**Proposed DPARNet**        |-       |7.2     |7.2     |7.4     |8.6     |10.3    |10.9    |0.15          |15.2           | 
|**Beam-Guided DPARNet**     |-       |7.3     |**6.9** |7.2     |**7.7** |**9.0** |**9.4** |0.41          |41.1           |


## Citation
**This paper has been accepted yet not published. The citation will be added here when it is published.**

## Referenecs

**[1] Dang F, Chen H T, Zhang P Y. DPT-FSNet: Dual-path Transformer Based Full-band and Sub-band Fusion Network for Speech Enhancement. Proc. IEEE
Int. Conf. Acoust. Speech Signal Process., 2022: 6857—6861**

**[2] Chen H T, Zhang P Y. Beam-Guided TasNet: An Iterative Speech Separation Framework with Multi-Channel Output, 2021: arXiv preprint arXiv:
2102.02998**

**[3] Chen Z, Yoshioka T, Lu L et al. Continuous speech separation: dataset and analysis. Proc. IEEE Int. Conf. Acoust. Speech Signal Process., 2020:
7284—7288**

**[4] Chen S Y, Wu Y, Chen Z et al. Continuous Speech Separation with Conformer. Proc. IEEE Int. Conf. Acoust. Speech Signal Process., 2021; 5749—5753**

**[5] Zhang S Y, Li X F. Microphone Array Generalization for Multichannel Narrowband Deep Speech Enhancement. Proc. Interspeech, 2021: 666—670**

**Please feel free to contact us if you have any questions.**

[libricss]: https://github.com/chenzhuo1011/libri_css
[asteroid]: https://github.com/asteroid-team/asteroid
[sms_wsj]: https://github.com/fgnt/sms_wsj
