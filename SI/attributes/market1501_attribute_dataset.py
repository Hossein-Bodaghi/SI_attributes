#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 13:56:26 2021

@author: hossein
"""


from scipy import io
import numpy as np
import os
 
#%%
    
def market_duke_attr(path, key = 'market_attribute'):
    '''
    key = 'gallery' or 'market_attribute' or 'duke_attribute' 
    
    for market1501 output:
        0) young 1)teenager 2)adult 3)old 
        4)backpack 5)shoulder-bag 6)hand-bag 7)down-black 8)down-blue 9)down-brown 
        10)down-gray 11)down-green 12)down-pink 13)down-purple 14)down-white 
        15)down-yellow 16)up-black 17)up-blue 18)up-green 19)up-gray 
        20)up-purple 21)up-red 22)up-white 23)up-yellow 24)lower-body type 
        25)lower-body length 26)slleve-length 27)hair-length 28)hat 29)gender 
        30)ID
    
    for DukeMTMC:
        0) gender 1)upper-body_length 2)boots 3)hat 4)backpack 5)bag 
        6)hand_bag 7)age 8)upblack 9)upwhite 10)upred 11)uppurple 12)upgray 
        13)upblue 14)upgreen 15)upbrown 16)downblack 17)downwhite 18)downred 
        19)downgray 20)downblue 21)downgreen 22)downbrown
    '''
    
    mat = io.loadmat(path)
    attr1 = mat[key]
    a = np.ndarray.tolist(attr1)[0][0]
    
    b = np.ndarray.tolist(a[0])[0][0] # a list contains 28 numpy array with the size of 750 (train) for market 
    c = np.ndarray.tolist(a[1])[0][0] # a list contains 28 numpy array with the size of 751 (test) for market
    
    tr_attr = np.squeeze(np.array(c)) # (28, 751) the last row is a ndarray object for id
    te_attr = np.squeeze(np.array(b)) # (28, 750) the last row is a ndarray object for id
    # ids start from 1 to 1501
    
    for i in range(np.shape(tr_attr)[1]):
        tr_attr[-1,i] = int(tr_attr[-1,i])  # ndarray object to integer

    for i in range(np.shape(te_attr)[1]):
        te_attr[-1,i] = int(te_attr[-1,i])  # ndarray object to integer
        
    if key == 'duke_attribute':
        tr_attr = tr_attr.T # (1110, 24) actually test
        te_attr = te_attr.T  # (702, 24) actually train       
        t_attr = np.append(te_attr, tr_attr, axis=0) # (1812, 24) first train then test 
        t_attr = np.where(t_attr == 0, 1, t_attr)
        t_attr[:,:-1] = t_attr[:,:-1] - 1
        return t_attr
        
    elif key == 'market_attribute':    
        tr_attr = tr_attr.T # (750, 28)
        te_attr = te_attr.T  # (751, 28)
        t_attr = np.append(tr_attr, te_attr, axis=0) # (1501, 28) first train then test 
        attributes = np.zeros((1501, 31), dtype = int) # becuase age in first row contains {1, 2, 3, 4} and is not binary
        
        for i in range(len(attributes)):
            for j in range(28):
                if j == 0:
                    if t_attr[i, j] == 1:
                        attributes[i, 0] = 1
                        
                    elif t_attr[i, j] == 2:
                        attributes[i, 1] = 1
                        
                    elif t_attr[i, j] == 3:
                        attributes[i, 2] = 1
                        
                    elif t_attr[i, j] == 4:
                        attributes[i, 3] = 1
                elif j == 27:
                    attributes[i, j+3] = t_attr[i, j]
                    
                else:
                    attributes[i, j+3] = t_attr[i, j] - 1
        # train and test orders are not the same
        attributes2 = np.zeros((1501, 31), dtype = int)
        attributes2[:751,:] = attributes[:751,:] # choose train as origin 
        attributes2[751:,:7] = attributes[751:,:7] # the first six attributes orders are same
        attributes2[751:,-1] = attributes[751:,-1] # the ID (last column) 
        attributes2[751:,7] = attributes[751:,21] # down-black
        attributes2[751:,8] = attributes[751:,27] # down-blue
        attributes2[751:,9] = attributes[751:,29] # down-brown
        attributes2[751:,10] = attributes[751:,26] # down-gray
        attributes2[751:,11] = attributes[751:,28] # down-green
        attributes2[751:,12] = attributes[751:,23] # down-pink
        attributes2[751:,13] = attributes[751:,24] # down-purple
        attributes2[751:,14] = attributes[751:,22] # down-white
        attributes2[751:,15] = attributes[751:,25] # down-yellow
        attributes2[751:,16] = attributes[751:,13] # up-black
        attributes2[751:,17] = attributes[751:,19] # up-blue
        attributes2[751:,18] = attributes[751:,20] # up-green
        attributes2[751:,19] = attributes[751:,18] # up-gray
        attributes2[751:,20] = attributes[751:,16] # up-purple
        attributes2[751:,21] = attributes[751:,15] # up-red
        attributes2[751:,22] = attributes[751:,14] # up-white
        attributes2[751:,23] = attributes[751:,17] # up-yellow
        attributes2[751:,24] = attributes[751:,7] # lower-type
        attributes2[751:,25] = attributes[751:,8] # lowe-length
        attributes2[751:,26] = attributes[751:,9] # lower-length
        attributes2[751:,27] = attributes[751:,10] # hair-length
        attributes2[751:,28] = attributes[751:,11] # hat
        attributes2[751:,29] = attributes[751:,12] # gender
        return attributes2

#%%

main_path = '/home/hossein/deep-person-reid/my_osnet/Market-1501-v15.09.15/gt_bbox/'
attr_path = '/home/hossein/deep-person-reid/datasets/dukemtmc/DukeMTMC-attribute-master/duke_attribute.mat'
attr = market_duke_attr(attr_path, key='duke_attribute') 

#%%
'''
prepare our data for gt_bbox folder
'''

main_path = '/home/hossein/deep-person-reid/my_osnet/Market-1501-v15.09.15/gt_bbox/'
attr_path = '/home/hossein/deep-person-reid/market1501_label/Market-1501_Attribute-master/market_attribute.mat'
attr = market_duke_attr(attr_path) # (1501, 31)

img_names = os.listdir(main_path)
img_names.sort()

attributes = np.zeros((len(img_names), 30))

for i, name in enumerate(img_names):
    b = name.split('_')
    for j in range(len(attr)):
        if attr[j, 30] == int(b[0]):
            attributes[i] = attr[j, :30]
save_path = '/home/hossein/deep-person-reid/market1501_label/Market-1501_Attribute-master/gt_bbox_market_attribute.npy'    
np.save(save_path, attributes)

