# getNNUnet

# System requirements

## Operating System
getNNUnet has been tested on Linux (Ubuntu 22.04)! It should work out of the box!

## Hardware requirements
We support GPU (recommended) and CPU

# Installation instructions
We strongly recommend that you install nnU-Net in a virtual environment! Pip or anaconda (e.g. pip or conda install) are both recommened.

Use a recent version of Python! 3.9 or newer is guaranteed to work!

**nnU-Net v2 can coexist with nnU-Net v1! Both can be installed at the same time.**

1) Install [PyTorch](https://pytorch.org/get-started/locally/) as described on their website (conda/pip). Please 
install the latest version with support for your hardware (cuda, mps, cpu).
**DO NOT JUST `pip install nnunetv2` WITHOUT PROPERLY INSTALLING PYTORCH FIRST**. For maximum speed, consider 
[compiling pytorch yourself](https://github.com/pytorch/pytorch#from-source) (experienced users only!). 

2) Install nnU-Net depending on your use case:
    1) For use as **standardized baseline**, **out-of-the-box segmentation algorithm** or for running 
     **inference with pretrained models**:

       ```pip install nnunetv2```
