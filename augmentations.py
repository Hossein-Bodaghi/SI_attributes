#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 10:02:19 2021

@author: hossein
"""

from delivery import data_delivery
from torchvision import transforms
import torch
from loaders import get_image
from utils import plot, augmentor
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('calculation is on:',device)
torch.cuda.empty_cache()

#%%
main_path = './datasets/Market1501/Market-1501-v15.09.15/gt_bbox/'
path_attr = './attributes/CA_Market.npy'

attr = data_delivery(main_path,
                  path_attr=path_attr,
                  need_parts=True,
                  need_attr=True,
                  dataset = 'CA_Market')

#%%    
train_transform = transforms.Compose([
                            transforms.RandomRotation(degrees=10),
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(saturation=[1,3])
                            ])

torch.manual_seed(0)



img_paths = [os.path.join(main_path, attr['img_names'][i]) for i in torch.randint(0, 25259, (10,1))]
orig_imgs = [get_image(addr,256, 128) for addr in img_paths]

augmented = [augmentor(orig_img, train_transform) for orig_img in orig_imgs]
plot(augmented, orig_imgs)

    




