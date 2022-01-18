#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 19:18:24 2022

@author: hossein
"""
from scipy import io
import numpy as np
import os
 

mat = io.loadmat('/home/hossein/SI_attributes/datasets/RAPv2/RAP_annotation/RAP_annotation.mat')
attr1 = mat['RAP_annotation']
a = np.ndarray.tolist(attr1)[0][0]
attributes = np.array(np.ndarray.tolist(a[1]))
names = np.squeeze(np.array(np.ndarray.tolist(a[2])))




b = np.ndarray.tolist(a[2])[0]
strings = [name for name in a] # a list contains 28 numpy array with the size of 750 (train) for market 
c = np.ndarray.tolist(a[1])[0][0]

