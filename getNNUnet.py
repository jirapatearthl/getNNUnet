#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 13:24:12 2022

@author: Jirapat Likitlersuang, PhD
"""
import onnx
import os
import torch
from nnunet.training.model_restore import load_model_and_checkpoint_files


def getNNUNet2ONNX(model, output_filenames, folds=0, mixed_precision=True, checkpoint_name="model_final_checkpoint"):
    """
    :param segmentation_export_kwargs:
    :param model: folder where the model is saved, must contain fold_x subfolders
    :param output_filenames: output path to saved
    :param folds: default = 0 for using fold_0
    :return:
    """
    
    print("emptying cuda cache")
    torch.cuda.empty_cache()
    print("loading parameters for folds,", folds)
    foldNum=int(folds)
    foldVal="fold_"+str(folds)
    trainer, params = load_model_and_checkpoint_files(model, folds, mixed_precision=mixed_precision, checkpoint_name=checkpoint_name)
    trainer.load_checkpoint_ram(params[foldNum], False)
    checkpoint = os.path.join(model, foldVal, checkpoint_name+".model" )
    modelNet = torch.load(checkpoint)
    print("Reading Model...")
    print(folds)
    
    net = trainer.network
    net.load_state_dict(modelNet['state_dict'])
    net.eval()

    fileNameFileNow = output_filenames
    fileNameNewNow = os.path.join(fileNameFileNow, "ONNX_MODEL")
    if not os.path.exists(fileNameNewNow):
        os.makedirs(fileNameNewNow)
    fileNameNewNameNow = os.path.join(fileNameNewNow, "nnunetModel.onnx")
    #print((trainer.batch_size, tuple(list(trainer.patch_size))[0], tuple(list(trainer.patch_size))[1], tuple(list(trainer.patch_size))[2]))
    dummpySample = torch.zeros([1, 1, tuple(list(trainer.patch_size))[0], tuple(list(trainer.patch_size))[1], tuple(list(trainer.patch_size))[2]]).to("cuda") 
    
    torch.onnx.export(net, dummpySample, fileNameNewNameNow, verbose=False, input_names=["input"], output_names=["output"], opset_version=11, operator_export_type=torch.onnx.OperatorExportTypes.ONNX)
    
    


##Example
##To save .onnx (as pytorch)
TestInputModelPath="/mnt/InternalHDD/User/likitler/ENE_Project/Segmentation/HeadNeck/nnUNet/nnUNet_trained_models/nnUNet/3d_fullres/Task502_tot_p_n/nnUNetTrainerV2__nnUNetPlansv2.1"
TestOutputPath="/mnt/InternalHDD/User/likitler/ENE_Project/Segmentation/HeadNeck/data4Seg_T"
getNNUNet2ONNX(TestInputModelPath, TestOutputPath, 0)

#To load .onnx (as pytorch)
onnxPath = '/mnt/InternalHDD/User/likitler/ENE_Project/Segmentation/HeadNeck/data4Seg_T/ONNX_MODEL/nnunetModel.onnx'
onnx_model = onnx.load(onnxPath)
onnx.checker.check_model(onnx_model)
print(onnx.helper.printable_graph(onnx_model.graph))
