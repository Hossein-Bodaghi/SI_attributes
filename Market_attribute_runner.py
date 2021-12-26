#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 15:34:23 2021

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
from utils import get_n_params

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('calculation is on:',device)
torch.cuda.empty_cache()

#%%
main_path = './datasets/Market1501/Market-1501-v15.09.15/gt_bbox/'
path_attr = './attributes/Market_attribute_with_id.npy'

attr = data_delivery(main_path,
                  path_attr=path_attr,
                  need_id=True,
                  need_collection=True,
                  need_attr=True,
                  dataset = 'Market_attribute')


