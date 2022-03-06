#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 16:24:42 2022

@author: hossein
"""

import numpy as np

old_path = '/home/hossein/SI_attributes/attributes/CA_Duke_train_with_id.npy'
ca_duke_attr = np.load(old_path)








old_names = ['gender','hairless',"short hair","longhair(straight)","knot","unvisible(hair)",
            "burnette","blonde", "black",'no-color',
            'cap',"snowcap","hoodiecap","no cap","unvisible(cap)",
            "c_white","c_blue","c_green","c_red","c_brown","c_yellow","c_gray","c_black",'no-color', 
            "T-shirt/shirt","jacket/sweatshirt","overcoat","hoodie",
            "b_white","b_purple","b_pink","b_blue","b_green","b_red","b_brown","b_yellow","b_gray","b_black", 
            "backpack","bag/handbag",'no bags',
            "umbrella(open)","umbrella(closed)","no umbrella",
            "beard","shaved","hidden",
            "pants","shorts","skirt","unvisible",
            "l_white","l_blue","l_green","l_red","l_brown","l_yellow","l_gray","l_black",'no-color',
            'formal shoes',"sneakers","high boots",'hidden',
            "f_white","f_colorful","f_brown","f_gray","f_black",'no-color',
            "sunglasses","headphone","gloves","scarf","tie",
            "front/back",
            "white", "black", "unkown"
            ]

new_names = ['gender',
             'hairless',"short hair","longhair(straight)","knot","unvisible(hair)",
            "burnette","blonde", "black",'no-color',
            'cap',"snowcap","hoodiecap","no cap","unvisible(cap)",
            "c_white","c_blue","c_green","c_red","c_brown","c_gray","c_black",'no-color', 
            "T-shirt/shirt","jacket/sweatshirt","overcoat","hoodie",
            "b_white","b_blue","b_green","b_red","b_brown","b_yellow","b_gray","b_black", 
            "backpack","bag/handbag",'no bags',
            "beard","shaved","hidden",
            "pants","shorts","skirt","unvisible",
            "l_white","l_blue","l_green","l_red","l_brown","l_gray","l_black",'no-color',
            'formal shoes',"sneakers","high boots",'hidden',
            "f_white","f_colorful","f_brown","f_gray","f_black",'no-color',
            "sunglasses","headphone","gloves","scarf","tie","umbrella(open)","umbrella(closed)",
            "front/back",
            "white", "black", "unkown"
            ]

for i, name in enumerate(new_names):
    print(i, ') -->', name)

for i, name in enumerate(old_names):
    print(i, ') -->', name)
    
    
new_attr = np.zeros((len(ca_duke_attr), 75))

new_attr[:, :19] = ca_duke_attr[:, :19]
new_attr[:, 20:28] = ca_duke_attr[:, 21:29]
new_attr[:, 28:30] = ca_duke_attr[:, 31:33]
new_attr[:, 31:38] = ca_duke_attr[:, 34:41]
new_attr[:, 68:70] = ca_duke_attr[:, 41:43]
new_attr[:, 38:49] = ca_duke_attr[:, 44:55]
new_attr[:, 50:68] = ca_duke_attr[:, 57:75]
new_attr[:, 70:75] = ca_duke_attr[:, 75:80]

for i in range(len(ca_duke_attr)):    
    if ca_duke_attr[i, 19] == 1 or ca_duke_attr[i, 20] == 1:
        new_attr[i, 19] = 1

    if ca_duke_attr[i, 29] == 1 or ca_duke_attr[i, 30] == 1 or ca_duke_attr[i, 33] == 1:
        new_attr[i, 30] = 1
        
    if ca_duke_attr[i, 55] == 1 or ca_duke_attr[i, 56] == 1:
        new_attr[i, 49] = 1
        
# color handeling gray and other color at the same time 
idd = 1
ids = [1]
for i in range(len(ca_duke_attr)): 
    m = 0
    if idd != ca_duke_attr[i,-1]:
        idd = ca_duke_attr[i,-1]
        m += 1
        ids.append(ca_duke_attr[i,-1])

check = {}
for j in ids:
    b = np.where(new_attr[:,-1]==j)
    a = new_attr[b]
    sum_colors = np.sum(a[:,27:33])
    sum_colors += np.sum(a[:,34])
    sum_gray = np.sum(a[:,33])
    if sum_gray != 0 and sum_colors != 0:
        check.update({str(j):(int(sum_gray),int(sum_colors))})
        
def color_changer(new_attr, idd, c):
    b = np.where(new_attr[:,-1]==idd)
    new_attr[b,27:35] = 0
    new_attr[b,c] = 1
    return new_attr  

new_attr = color_changer(new_attr, 57, 29)
new_attr = color_changer(new_attr, 121, 33)
new_attr = color_changer(new_attr, 131, 33)
new_attr = color_changer(new_attr, 203, 27)
new_attr = color_changer(new_attr, 248, 33)
new_attr = color_changer(new_attr, 281, 34)
new_attr = color_changer(new_attr, 282, 33)
new_attr = color_changer(new_attr, 291, 34)
new_attr = color_changer(new_attr, 357, 33)
new_attr = color_changer(new_attr, 521, 33)
new_attr = color_changer(new_attr, 735, 29)
new_attr = color_changer(new_attr, 3371, 34)
new_attr = color_changer(new_attr, 4096, 34)
new_attr = color_changer(new_attr, 4192, 33)


np.save('/home/hossein/CA_Duke_train_with_id.npy', new_attr)
