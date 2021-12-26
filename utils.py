#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 14:35:31 2021

@author: hossein
"""

import numpy as np
import os
import torch

market_main_path = './datasets/Market1501/Market-1501-v15.09.15/gt_bbox/'
path_ca_market = '/home/hossein/SI_attributes/attributes/CA_Market_with_id.npy'
path_market_attribute = '/home/hossein/SI_attributes/attributes/Market_attribute_with_id.npy'

duke_path_train = '/home/hossein/SI_attributes/datasets/Dukemtmc/bounding_box_train'
duke_path_test = '/home/hossein/SI_attributes/datasets/Dukemtmc/bounding_box_test'
train_duke_path = '/home/hossein/SI_attributes/attributes/Duke_attribute_train_with_id.npy'
test_duke_path = '/home/hossein/SI_attributes/attributes/Duke_attribute_test_with_id.npy'

def load_attributes(path_attr):
    attr_vec_np = np.load(path_attr)# loading attributes
        # attributes
    attr_vec_np = attr_vec_np.astype(np.int32)
    return torch.from_numpy(attr_vec_np)

def load_image_names(main_path):
    img_names = os.listdir(main_path)
    img_names.sort()    
    return np.array(img_names)

def unique(list1):
    # initialize a null list
    unique_list = []    
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return len(unique_list)

def one_hot_id(id_):
    num_ids = unique(id_)
    id_ = torch.from_numpy(np.array(id_))# becuase list doesnt take a list of indexes it should be slice or inegers.
    id1 = torch.zeros((len(id_),num_ids))
    
    sample = id_[0]
    i = 0
    for j in range(len(id1)):
        if sample == id_[j]:
           id1[j, i] = 1
        else:
            i += 1
            sample = id_[j]
            id1[j, i] = 1     
    return id1

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


