
# MSPRT-TANDEM and LSEL
- See tutorial for (M)SPRT-TANDEM: https://github.com/Akinori-F-Ebihara/SPRT-TANDEM_tutorial
- See PyTorch version: https://github.com/Akinori-F-Ebihara/SPRT-TANDEM-PyTorch

## Introduction
This repository contains MSPRT-TANDEM, including the Log-Sum-Exp Loss (LSEL), proposed in ["The Power of Log-Sum-Exp: Sequential Density Ratio Estimation for Spped-Accuracy Optimization"](http://proceedings.mlr.press/v139/miyagawa21a.html) [ICML2021]. MSPRT-TANDEM performs early multiclass classification of time-series. MSPRT-TANDEM sequentially estimates log-likelihood ratios of multihypotheses, or classes, for fast and accurate sequential data classification, with the help of a novel loss function, the LSEL. We prove in our paper that the LSEL is consistent, has the hard class weighting effect, and guess-averse. 


## Requirements
- Python 3.5
- TensorFlow 2.0.0
- CUDA 10.0
- cuDNN 7.6.0
- Jupyter Notebook 4.4.0
- Optuna 0.14.0
 

## Quick Start
1. Create NMNIST-H and NMNSIT-100f.

    1-0. `cd ./data-directory`
    
    1-1. `python make_nmnist-h.py` with `train_or_test` = "train"
    
    1-2. `python make_nmnist-h.py` with `train_or_test` = "test"
    
    1-3. `python make_nmnist-100f.py` with `train_or_test` = "train"

    1-4. `python make_nmnist-100f.py` with `train_or_test` = "test"

2. Extract features.
    2-1. `save_feature_tfrecords.ipynb` with `save_data` = "train" (NMNSIT-H)

    2-1. `save_feature_tfrecords.ipynb` with `save_data` = "test" (NMNSIT-H)

    2-1. `save_feature_tfrecords.ipynb` with `save_data` = "train" (NMNSIT-100f)

    2-1. `save_feature_tfrecords.ipynb` with `save_data` = "test" (NMNSIT-100f)

3. Plot the speed-accuracy tradeoff (SAT) curve with `plot_SATC_casual.ipynb`.


## Result Images
### Speed-accuracy tradeoff (SAT) curve on NMNIST-H
![](./images/github1.png)

### MSPRT-TANDEM vs. NP test on NMNIST-H
![](./images/github2.png)

### SAT curve with standard deviation of hitting times on NMNIST-H
![](./images/github3.png)

### Trajectories of log-likelihood ratio of NMNIST-H
![](./images/github4.png)


## What You Can Do
1. Create and extract the bottleneck features of NMNIST-H and NMNIST-100f as TFRecord files.
    - Relevant files and directories:
        - /data-directory/make_nmnist-h.py
        - /data-directory/make_nmnist-100f.py
        - /data-directory/trained_models
        - save_feature_tfrecords.ipynb
2. Create and extract the bottleneck features of the clipped UCF101 and HMDB51 used in our paper as TFRecord files.
    - Relevant directories:
        - /data-directory/create_UCF101
        - /data-directory/create_HMDB51
3. Train a ResNet on NMNIST-H or NMNSIT-100f as a feature extractor.
    - Relevant files:
        - train_fe_nmnist-h.py
        - /configs/config_fe_nmnist-h.yaml
        - show_trial_params.ipynb
4. Train an LSTM on NMNIST-H, NMNIST-100f, UCF101, or HMDB51 as a temporal integrator.
    - Relevant files:
        - train_X_Y.py (X = ti or dre. Y = nmnist-h or UCF101.)
        - trains_X_Y.sh 
        - /configs/config_X_Y.yaml
        - show_trial_params.ipynb
5. Plot SAT curves.
    - Relevant files:
        - plot_SATC_casual.ipynb
        - plot_SATC_casual_lite.ipynb

## Directories
- `/configs`
    - Config files for training codes.
- `/datasets`
    - TFRecords loader and preprocessing methods.
- `/models`
    - Backbone ResNet, LSTM, and loss functions. The LSEL is here (/models/losses.py).
- `/utils`
    - MSPRT algorithm and miscellaneous stuff.
- `/data-directory`
    - The train logs will be stored here.
    - Creation files of NMNIST-H, NMNSIT-100f, UCF101, and HMDB51.
- `/images`
    - For README file.

## Citation
___Please cite our paper if you use the whole or a part of our codes.___
```
ICML
@InProceedings{MSPRT-TANDEM,
  title = 	 {The Power of Log-Sum-Exp: Sequential Density Ratio Matrix Estimation for Speed-Accuracy Optimization},
  author =       {Miyagawa, Taiki and Ebihara, Akinori F},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {7792--7804},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/miyagawa21a/miyagawa21a.pdf},
  url = 	 {http://proceedings.mlr.press/v139/miyagawa21a.html},
  abstract = 	 {We propose a model for multiclass classification of time series to make a prediction as early and as accurate as possible. The matrix sequential probability ratio test (MSPRT) is known to be asymptotically optimal for this setting, but contains a critical assumption that hinders broad real-world applications; the MSPRT requires the underlying probability density. To address this problem, we propose to solve density ratio matrix estimation (DRME), a novel type of density ratio estimation that consists of estimating matrices of multiple density ratios with constraints and thus is more challenging than the conventional density ratio estimation. We propose a log-sum-exp-type loss function (LSEL) for solving DRME and prove the following: (i) the LSEL provides the true density ratio matrix as the sample size of the training set increases (consistency); (ii) it assigns larger gradients to harder classes (hard class weighting effect); and (iii) it provides discriminative scores even on class-imbalanced datasets (guess-aversion). Our overall architecture for early classification, MSPRT-TANDEM, statistically significantly outperforms baseline models on four datasets including action recognition, especially in the early stage of sequential observations. Our code and datasets are publicly available.}
}

(arXiv; cite the ICML version.)
@article{MSPRT-TANDEM,
  title={The Power of Log-Sum-Exp: Sequential Density Ratio Matrix Estimation for Speed-Accuracy Optimization},
  author={Miyagawa, Taiki and Ebihara, Akinori F},
  journal={arXiv preprint arXiv:2105.13636},
  year={2021}
}
```
