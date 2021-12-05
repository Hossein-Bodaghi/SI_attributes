#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 17:07:01 2021

@author: hossein
"""

import numpy as np
import os

main_path = '/home/hossein/deep-person-reid/my_osnet/Market-1501-v15.09.15/gt_bbox/'

img_names = os.listdir(main_path)
img_names.sort()

id_ = []
for name in img_names:
    b = name.split('_')
    id_.append(int(b[0]))
    
market_attr_path = '/home/hossein/deep-person-reid/market1501_label/Market-1501_Attribute-master/gt_bbox_market_attribute.npy' 
market_attrs = np.load(market_attr_path)
attr_path = '/home/hossein/anaconda3/envs/torchreid/deep-person-reid/my_osnet/attributes/total_attr.npy'
attributes =  np.load(attr_path)
atr_new = np.zeros((np.shape(attributes)[0], 48), dtype=int)

atr_new[:, :6] = attributes[:, :6] # ['gender','cap','hairless','short hair','long hair','knot']

# handle hair colors from 9 colors to 2 color (black, colourful) 
# ['h_w','h_r','h_o','h_y','h_green','h_b','h_gray','h_p','h_black'] to [h_colorful, h_black]
for i in range(np.shape(attributes)[0]):
    for j in range(6,14):
        if attributes[i, j] == 1:
            atr_new[i, 6] = 1
atr_new[:, 7] = attributes[:, 14]


# for all our old attributes that they are tshirt and shirt, we devide them to short sleeve and long sleeve
# ['Tshirt/shirt','coat','top','simple/patterned'] to ['Tshirt_shs', 'shirt_ls','coat','top','simple/patterned']
for i in range(np.shape(attributes)[0]):
    if attributes[i, 15] == 1:
        if market_attrs[i, 26] == 1:
            atr_new[i, 8] = 1
        else:
            atr_new[i, 9] = 1
atr_new[:, 10:13] = attributes[:, 16:19]  

# upper body color we eliminated the orange color
# ['b_w','b_r','b_o','b_y','b_green','b_b','b_gray','b_p','b_black'] to ['b_w','b_r','b_y','b_green','b_b','b_gray','b_p','b_black']
atr_new[:, 13:15] = attributes[:, 19:21]
atr_new[:, 15:21] = attributes[:, 22:28]
oranges = np.where(attributes[:, 21])[0]
for idx in oranges:
    if id_[idx]==52 or id_[idx]==100 or id_[idx]==105 or id_[idx]==520: # red
        atr_new[idx, 14] = 1
        
    elif id_[idx]==622 or id_[idx]==729 or id_[idx]==779 or id_[idx]==818: # red
        atr_new[idx, 14] = 1
        
    elif id_[idx]==921 or id_[idx]==1098 or id_[idx]==1099 or id_[idx]==1137: # red
        atr_new[idx, 14] = 1 
        
    elif id_[idx]==1189 or id_[idx]==1201: # red
        atr_new[idx, 14] = 1
    
    elif id_[idx]==80 or id_[idx]==140: # gray
        atr_new[idx, 18] = 1          

    elif id_[idx]==427 or id_[idx]==527 or id_[idx]==1341: # yellow
        atr_new[idx, 15] = 1    

    elif id_[idx]==238 or id_[idx]==937: # green
       atr_new[idx, 16] = 1    
       
       
# we devided our hand bag to hand bag and shoulder bag
# ['backpack', 'hand bag', 'no bag'] to ['backpack', shoulder bag, 'hand bag', 'no bag']
for i in range(np.shape(attributes)[0]):
    if attributes[i, 28] == 1:
        if market_attrs[i, 5] == 1:
            atr_new[i, 22] = 1
        else:
            atr_new[i, 23] = 1            
atr_new[:, 21] = attributes[:, 28]   
atr_new[:, 24] = attributes[:, 30]       

# lower body color we have changed orange with brown
# ['l_w','l_r','l_o','l_y','l_green','l_b','l_gray','l_p','l_black'] to ['l_w','l_r','l_br','l_y','l_green','l_b','l_gray','l_p','l_black']
atr_new[:, 28:37] = attributes[:, 34:43]
browns = [45, 69, 88, 138, 228, 259, 289, 376,
          489, 615, 1371, 1376, 1467, 205, 269, 294,
          308, 341, 348, 372, 385, 405, 522, 582,
          613, 648, 683, 751, 753, 763, 898, 968, 978,
          989, 1017, 1111, 1254, 1268, 1335, 1428, 1443,
          1489]
browns.sort()  
greens = [68, 242]
yellows = [191, 1137, 1341, 1354]
whites = [262, 779]
grays = [142, 994]
reds = [630, 1189]
for i in range(np.shape(attributes)[0]):
    for idd in browns:
        if id_[i] == idd:
            atr_new[i, 28:37] = 0
            atr_new[i, 30] = 1
    for idd in greens:
        if id_[i] == idd:
            atr_new[i, 28:37] = 0
            atr_new[i, 32] = 1    
    for idd in yellows:
        if id_[i] == idd:
            atr_new[i, 28:37] = 0
            atr_new[i, 31] = 1        
    for idd in whites:
        if id_[i] == idd:
            atr_new[i, 28:37] = 0
            atr_new[i, 28] = 1   
    for idd in grays:
        if id_[i] == idd:
            atr_new[i, 28:37] = 0
            atr_new[i, 34] = 1 
    for idd in reds:
        if id_[i] == idd:
            atr_new[i, 28:37] = 0
            atr_new[i, 29] = 1             

# leg part without any change
atr_new[:, 25:28] = attributes[:, 31:34]        

# foot
atr_new[:, 37:40] = attributes[:, 43:46] 

# shoes color
# a no color attribute is added 
# ['f_w','f_r','f_o','f_y','f_green','f_b','f_gray','f_p','f_black'] to ['no color','f_w', 'f_colorful','f_black']
# handle no color
for i in range(np.shape(attributes)[0]):
    for j in range(44,46):
        if attributes[i, j] == 1:
            atr_new[i, 40] = 1 # foot non 
            
atr_new[:, 41] = attributes[:, 46] # f_white

# shoes colors from 9 colors to 4 color 
for i in range(np.shape(attributes)[0]):
    for j in range(47,54):
        if attributes[i, j] == 1:
            atr_new[i, 42] = 1 # foot colourful   
atr_new[:, 43] = attributes[:, -1] # f_black]

# age added to new data, it is borrowed from original market attributes
atr_new[:, 44:48] = market_attrs[:, 0:4]


sum_attr = np.sum(attributes, axis=0)
sum_attr_sum = np.sum(atr_new, axis=0)


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
print('new attribtes \n')
for i in range(len(attr_names_new)):
    print(i , ')', attr_names_new[i], '-->', int(sum_attr_sum[i]))
    
attr_names_old = ['gender','cap','hairless','short hair','long hair',
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

print('\n the old attributes \n')
for i in range(len(attr_names_old)):
    print(i , ')', attr_names_old[i], '-->', int(sum_attr[i]))

#%%

def attr_id_finder(attributes, id_, attr_idx=21):
    ids = []
    oranges = np.where(attributes[:, attr_idx])
    idd = id_[oranges[0][0]]
    m = 0
    for idx in oranges[0]:
        if idd == id_[idx]:        
            if m == 0:
                print(idd)
                ids.append(idd)
            else: pass
            m += 1
        else:
            m = 1
            idd = id_[idx]
            print(idd)
            ids.append(idd)
    return ids

body_oranges = attr_id_finder(attributes, id_, attr_idx=21)  
leg_oranges = attr_id_finder(market_attrs, id_, attr_idx=9)  
#%%
np.save('/home/hossein/anaconda3/envs/torchreid/deep-person-reid/my_osnet/attributes/new_total_attr.npy',
        atr_new)
