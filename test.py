#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 13:05:44 2022

@author: hossein
"""
import torch
torch.cuda.empty_cache()
from torchreid import models
from torchreid import utils
from transformers import MBConvBlock, ScaledDotProductAttention
import torch.nn as nn
from utils import get_n_params

input=torch.randn(1,3,256,128)


model = models.build_model(
    name='osnet_x1_0',
    num_classes=751,
    loss='softmax',
    pretrained=False
)

weight_path = './checkpoints/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth'
utils.load_pretrained_weights(model, weight_path)
global_avgpool = nn.AdaptiveAvgPool2d(1)
out_conv_4 = model.layer_extractor(input, 'out_conv4') 

s2 = MBConvBlock(ksize=3,input_filters=512, output_filters=128, image_size=(16,8))

#####
mlp2=nn.Sequential(MBConvBlock(ksize=3,input_filters=512, output_filters=128, image_size=(16,8)),
    nn.Conv2d(128, 128, kernel_size=1),
    nn.ReLU(),
    nn.Conv2d(128, 128, kernel_size=1)
)
#####


num_params_mlp2 = get_n_params(mlp2) # 318336

out_mlp2 = mlp2(out_conv_4) # (1, 256, 16, 8) (1, 128, 16, 8)

#####
s3 = ScaledDotProductAttention(128, 16, 16, 8)
#####


out_mlp2_reshaped = out_mlp2.reshape(1, 128, -1).permute(0,2,1) #B,N, (1, 128, 128)
out_s3 = s3(out_mlp2_reshaped,out_mlp2_reshaped,out_mlp2_reshaped) # (1, 128, 128)


#####
mlp3=nn.Sequential(
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 128)
)
#####




out_mlp3 = mlp3(out_s3) # (1, 128, 128)


maxpool1d = nn.MaxPool1d(kernel_size=2, stride=2)
out_maxpool1 = maxpool1d(out_mlp3.permute(0,2,1)).permute(0,2,1) # (1, 64, 128)


#####
s4 = ScaledDotProductAttention(128, 16, 16,8)
#####




out_s4 = s4(out_maxpool1, out_maxpool1, out_maxpool1) # (1, 64, 128)

#####
mlp4=nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32)
)
#####



out_mlp4 = mlp4(out_s4) # (1, 64, 32)

out_maxpool2 = maxpool1d(out_mlp4.permute(0,2,1))

reshape_final = out_maxpool2.reshape(1, -1)

mlp5=nn.Sequential(
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 4)
)

out_final = mlp5(reshape_final)

# out_new_conv = mlp2(out_conv_4)
# y = out_new_conv.reshape(1, 64, -1).permute(0,2,1) #B,N,

# out_dot = s3(y, y, y)
# out_mlp = mlp3(out_dot)
