#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 13:44:15 2021

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

train_img_path = './datasets/Dukemtmc/bounding_box_train'
test_img_path = './datasets/Dukemtmc/bounding_box_test'
path_attr_train = './attributes/Duke_attribute_train_with_id.npy'
path_attr_test = './attributes/Duke_attribute_test_with_id.npy'

attr_train = data_delivery(train_img_path,
                  path_attr=path_attr_train,
                  need_id=False,
                  need_collection=True,
                  need_attr=True,
                  dataset = 'Duke_attribute')

attr_test = data_delivery(test_img_path,
                  path_attr=path_attr_test,
                  need_id=False,
                  need_collection=True,
                  need_attr=True,
                  dataset = 'Duke_attribute')

#%%