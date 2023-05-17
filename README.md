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

1) Install nnU-Net, please following the instruction shown in the following website: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md

```pip install nnunetv2```

2) Install ONNX libary either with pip or anaconda using the following: 

```pip install onnx```
or 
```conda install -c conda-forge onnx```

3) Download and import getNNUnet.py and called getNNUNet2ONNX(...), example can be found in the code

4) Note that you can easily convert an.onnx model to pytorch by using "onnx2pytorch" library:

```pip install onnx2pytorch```

For example:
```
import onnx
from onnx2pytorch import ConvertModel

onnx_model = onnx.load(path_to_onnx_model)
pytorch_model = ConvertModel(onnx_model)
```

# Limitations:

* Only work with 3D 
* We assumed that your model are trained using CUDA
