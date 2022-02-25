#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 15:41:29 2022

@author: hossein
"""
import numpy as np
# from SI_attributes.utils import load_attributes, load_image_names


attr_names_old =['gender','cap','hairless','short hair','long hair',
           'knot','h_w','h_r','h_o','h_y','h_green','h_b',
           'h_gray','h_p','h_black','Tshirt/shirt','coat',
           'top','simple/patterned','b_w','b_r',
           'b_o','b_y','b_green','b_b',
           'b_gray','b_p','b_black','backpack',
           'hand bag','no bag','pants',
           'short','skirt','l_w','l_r','l_o','l_y','l_green','l_b',
           'l_gray','l_p','l_black','shoes','sandal',
           'hidden','f_w','f_r','f_o','f_y','f_green','f_b',
           'f_gray','f_p','f_black']

attr_names_new = ['gender','cap','hairless','short hair','long hair',
           'knot', 'h_colorful', 'h_black','Tshirt_shs', 'shirt_ls','coat',
           'top','simple/patterned','b_w','b_r',
           'b_y','b_green','b_b',
           'b_gray','b_p','b_black','backpack', 'shoulder bag',
           'hand bag','no bag','pants',
           'short','skirt','l_w','l_r','l_br','l_y','l_green','l_b',
           'l_gray','l_p','l_black','shoes','sandal',
           'hidden','no color','f_w', 'f_colorful','f_black', 'young', 
           'teenager', 'adult', 'old']

attr_names_modified =  ['gender','cap','hairless','short hair','long hair',
           'knot', 'h_colorful', 'h_black','Tshirt/shirt', 'coat',
           'top','simple/patterned','b_w','b_r',
           'b_y','b_green','b_b',
           'b_gray','b_p','b_black','backpack', 'bag','no bag','pants',
           'short','skirt','l_w','l_r','l_br','l_y','l_green','l_b',
           'l_gray','l_p','l_black','shoes','sandal',
           'hidden','no color','f_w', 'f_colorful','f_black','young', 
           'teenager', 'adult', 'old']

attr_names_modified_new =  ['gender','cap','hairless','short hair','long hair',
           'knot', 'h_colorful/h_black','Tshirt/shirt', 'coat',
           'top','simple/patterned','b_w','b_r',
           'b_y','b_green','b_b',
           'b_gray','b_p','b_black','backpack', 'bag','no bag','pants',
           'short','skirt','l_w','l_r','l_br','l_y','l_green','l_b',
           'l_gray','l_p','l_black','shoes','sandal',
           'hidden','no color','f_w', 'f_colorful','f_black','young', 
           'teenager', 'adult', 'old']
path_old = '/home/hossein/anaconda3/envs/torchreid/deep-person-reid/my_osnet/attributes/total_attr.npy'
path_new = '/home/hossein/SI_attributes/attributes/CA_Market_with_id.npy'
path_modified = '/home/hossein/CA_Market_with_id.npy'
attr_old = np.load(path_old) # numpy array
attr_new = np.load(path_new) # numpy array
attr_modified = np.load(path_modified) # numpy array
attr_modified_new = np.copy(attr_modified)
attr_modified_new = np.delete(attr_modified_new,[7], axis=1)




