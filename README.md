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

4) Note that you can easily convert a .onnx model to pytorch by using "onnx2torch" library:

```pip install onnx2torch```
or
```conda install -c conda-forge onnx2torch```

For example:
```
import torch
from onnx2torch import convert

# Path to ONNX model
onnx_model_path = '/some/path/mobile_net_v2.onnx'
# You can pass the path to the onnx model to convert it or...
torch_model_1 = convert(onnx_model_path)

# Or you can load a regular onnx model and pass it to the converter
onnx_model = onnx.load(onnx_model_path)
torch_model_2 = convert(onnx_model)
```
# Save nnUnet model to onnx (using getNNUnet)

Please used "getNNUnet.py" and import getNNUNet2ONNX

# Save modified nnUNET pretrained model to custom architecture (with example)

Please used "modiNNUNET.py" and import modifyNNUNET


# Limitations:

* Only work with 3D 
* We assumed that your model are trained using CUDA
