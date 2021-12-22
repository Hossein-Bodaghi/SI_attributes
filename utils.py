#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 14:35:31 2021

@author: hossein
"""

import numpy as np

path_ca_market = '/home/hossein/SI_attributes/attributes/CA_Market.npy'

def load_attributes(path_attr, ratio=1):
    
    attr_vec_np = np.load(path_attr)# loading attributes
        # attributes
    attr_vec_np = attr_vec_np.astype(np.int32)
    if ratio:
        attr_vec_np = np.append(attr_vec_np,attr_vec_np,axis=0)  
        
    return attr_vec_np

ca_attr = load_attributes(path_ca_market, ratio=None)