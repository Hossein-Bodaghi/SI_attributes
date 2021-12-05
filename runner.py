#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 14:22:06 2021

@author: hossein
"""

from delivery import data_delivery
from torchvision import transforms
from loaders import CA_Loader
from models import mb_build_model
import torch
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('calculation is on:',device)
torch.cuda.empty_cache()

#%%
main_path = '/home/hossein/deep-person-reid/my_osnet/Market-1501-v15.09.15/gt_bbox/'
path_attr = '/home/hossein/SI_attribute/attributes/new_total_attr.npy'

attr = data_delivery(main_path,
                  path_attr=path_attr,
                  path_start=None,
                  only_id=False,
                  double = False,
                  need_collection=True,
                  need_attr=True,
                  mode = 'CA_Market')

train_idx_path = './attributes/train_idx_full.pth' 
test_idx_path = './attributes/test_idx_full.pth'
train_idx = torch.load(train_idx_path)
test_idx = torch.load(test_idx_path)

for key , value in attr.items():
  try: print(key , 'size is: \t {} \n'.format((value.size())))
  except TypeError:
    print(key,'\n')
#%%    
train_transform = transforms.Compose([
                            transforms.RandomRotation(degrees=10),
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(saturation=[1,3])
                            ])

train_data = CA_Loader(img_path=main_path,
                          attr=attr,
                          resolution=(256,128),
                          transform=train_transform,
                          indexes=train_idx,
                          need_attr =False,
                          need_collection=True,
                          need_id = False,
                          two_transforms = False)

test_data = CA_Loader(img_path=main_path,
                          attr=attr,
                          resolution=(256, 128),
                          indexes=test_idx,
                          need_attr = False,
                          need_collection=True,
                          need_id = False,
                          two_transforms = False,                          
                          ) 

batch_size = 32
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_data,batch_size=100,shuffle=False)
#%%
torch.cuda.empty_cache()
from torchreid import models
from torchreid import utils

model = models.build_model(
    name='osnet_x1_0',
    num_classes=751,
    loss='softmax',
    pretrained=False
)

weight_path = '/home/hossein/Downloads/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth'
utils.load_pretrained_weights(model, weight_path)

attr_net = mb_build_model(model=model,
                 feature_dim = 512,
                 attr_dim = 46)

attr_net = attr_net.to(device)
