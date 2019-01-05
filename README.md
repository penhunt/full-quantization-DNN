# Blended coarse gradient descent for full quantization

This repository implements Blended Coarse Gradient Descent in the paper "Blended coarse gradient descent for full quantization of deep neural networks".

<p align="center">
  <img src="https://github.com/penhunt/full-quantization-DNN/blob/master/resnet18_4a1w_top1.png" width="600" alt="accessibility text">
</p>

## References
```
@article{yin2018blended,
  title={Blended coarse gradient descent for full quantization of deep neural networks},
  author={Yin, Penghang and Zhang, Shuai and Lyu, Jiancheng and Osher, Stanley and Qi, Yingyong and Xin, Jack},
  journal={arXiv preprint arXiv:1808.05240},
  year={2018}
}
```
```
@inproceedings{
yin2018understanding,
title={Understanding Straight-Through Estimator in Training Activation Quantized Neural Nets},
author={Penghang Yin and Jiancheng Lyu and Shuai Zhang and Stanley J. Osher and Yingyong Qi and Jack Xin},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=Skh4jRcKQ},
}
```

## Run the demo
```
python main_cifar.py --method bc --rho 1e-5
```
i.e., blending parameter = 1e-5
