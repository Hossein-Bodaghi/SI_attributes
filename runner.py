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
import torch.nn as nn 
from trainings import dict_training_multi_branch

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

attr_net = mb_build_model(model = model,
                 main_cov_size = 512,
                 attr_dim = 128,
                 dropout_p = 0.3,
                 sep_conv_size = 128,
                 sep_fc = True,
                 sep_clf = True)

attr_net = attr_net.to(device)
#%%
img = test_data[0]['img']
img = torch.unsqueeze(img, dim=0)

img1 = test_data[1]['img']
img1 = torch.unsqueeze(img1, dim=0)

img2 = torch.cat([img, img1], dim=0)

img2 = img2.to(device)

out = attr_net(img2)
#%%
criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.BCEWithLogitsLoss()

params = attr_net.parameters()

lr = 3.5e-4
optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.99), eps=1e-08)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 17], gamma=0.1)
#%%
save_path = './results/'
dict_training_multi_branch(num_epoch = 30,
                     attr_net = attr_net,
                     train_loader = train_loader,
                     test_loader = test_loader,
                     optimizer = optimizer,
                     scheduler = scheduler,
                     cce_loss = criterion1,
                     bce_loss = criterion2,
                     save_path = save_path,                    
                     device = device,
                     version = 'V_Cl_MFC_01',
                     resume=False,
                     loss_train = None,
                     loss_test=None,
                     train_attr_F1=None,
                     test_attr_F1=None,
                     train_attr_acc=None,
                     test_attr_acc=None,  
                     stoped_epoch=None)