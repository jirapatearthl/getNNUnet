#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 16:53:34 2023

@author: Jirapat Likitlersuang, PhD
"""

import onnx
import torch
from torchinfo import summary
from onnx2torch import convert
import numpy as np


def modifyNNUNET(onnxmodel, inputSize, outputFilenames, sliceLayer, newLayer, toFreeze = True):
    """
    :param onnxModel: file location where the .onnx model is saved
    :param inputSize: size of the input 
    :param outputFilenames: output path to saved modify model in .onnx
    :param sliceLayer: slice to trim 
    :param newLayer: layer to stack after the slice layer 
    :return:
    """
    
    onnxPath = onnxmodel###'/mnt/InternalHDD/User/likitler/ENE_Project/Segmentation/HeadNeck/data4Seg_T/ONNX_MODEL/nnunetModel.onnx'
    onnx_model = onnx.load(onnxPath)

    print(onnx.helper.printable_graph(onnx_model.graph))

    pytorch_model = convert(onnx_model)
    model=pytorch_model 
    print(summary(model, input_size=inputSize, device="cuda"))

    child_counter = 0
    for child in model.children():
       child_counter += 1
    for child in model.children():
       for param in child.parameters():
          break
       break
    coderEnd = sliceLayer
    child_counter = 0
    for child in model.children():
       if child_counter < coderEnd:
          for param in child.parameters():
              param.requires_grad = not toFreeze
       else:
           print("child ",child_counter," was not frozen")
       child_counter += 1

    print("MODEL")
    nnU_model = torch.nn.Sequential(*list(model.children())[:coderEnd])
    print(nnU_model)
    del pytorch_model
    del model
    #addU_model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(256000, 5000), torch.nn.ReLU(),  torch.nn.Linear(5000, 1000), torch.nn.Linear(1000, 1))
    ###addU_model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(552960, 32768), torch.nn.ReLU(), torch.nn.Linear(32768, 4096), torch.nn.Linear(4096, 256), torch.nn.Linear(256, 1), torch.nn.Sigmoid())
    ###addU_model = torch.nn.Sequential(torch.nn.AdaptiveMaxPool3d((6, 6, 6)), torch.nn.Flatten(), torch.nn.Linear(552960,4096), torch.nn.ReLU(), torch.nn.Linear(4096, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1), torch.nn.Sigmoid())
    
    
    addU_model = newLayer
    new_model = torch.nn.Sequential(nnU_model, addU_model)
    del nnU_model
    del addU_model
    
    
    print(summary(new_model, input_size=inputSize, device="cuda"))
    #print(summary(new_model, input_size=(2, 1, 64, 160, 160), device="cuda"))
    
    fileNameNewNameNow = outputFilenames###"/mnt/InternalHDD/User/likitler/ENE_Project/HN_DL_SCANNATIVE_PT/MODEL/Pytorch/nnTransferModel.onnx"
    
    dynamic_axes = {'input' : {0 : 'batch_size'}, 
                            'output' : {0 : 'batch_size'}}
    #dummpySample = torch.randn([2, 1, 64, 160, 160]).to("cuda") 
    dummpySample = torch.randn(inputSize).to("cuda") 
    torch.onnx.export(new_model, dummpySample, fileNameNewNameNow, verbose=False, input_names=['input'], output_names=["output"], export_params=True,
                      opset_version=11, operator_export_type=torch.onnx.OperatorExportTypes.ONNX, dynamic_axes=dynamic_axes)
    
modelPathIN = '/mnt/InternalHDD/User/likitler/ENE_Project/Segmentation/HeadNeck/data4Seg_T/ONNX_MODEL/nnunetModel.onnx'
inputSizeIN = (2, 1, 96, 192, 192)
freezeWeightIN = True
outputFilenamesIN ="/mnt/InternalHDD/User/likitler/ENE_Project/HN_DL_SCANNATIVE_PT/MODEL/Pytorch/nnTransferModel.onnx"
sliceLayerIN = 37
addModelIN = torch.nn.Sequential(torch.nn.AdaptiveMaxPool3d((6, 6, 6)), torch.nn.Flatten(), torch.nn.Linear(69120,4096), torch.nn.ReLU(), torch.nn.Linear(4096, 512), torch.nn.ReLU(), torch.nn.Linear(512, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1), torch.nn.Sigmoid())
modifyNNUNET(modelPathIN, inputSizeIN, outputFilenamesIN, sliceLayerIN,addModelIN, freezeWeightIN) 