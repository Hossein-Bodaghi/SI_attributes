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
idd = np.reshape(np.array(id_), (len(id_),1))
attr_with_id = np.append(atr_new, idd, axis=1)
market_attr_with_id = np.append(market_attrs, idd, axis=1)
np.save('./CA_Market_with_id.npy',
        attr_with_id)
np.save('./Market_attribute_with_id.npy',
        market_attr_with_id)


#%% CA-Duke new attributes


"""
old attributes : ["male/female",

    "hairless","short hair","longhair(straight)","knot","unvisible(hair)",
    "burnette","blonde","gray/white","black",
    'cap',"snowcap","hoodiecap","no cap", "unvisible(cap)",
    "c_white","c_purple","c_pink","c_blue","c_green","c_red","c_brown","c_yellow","c_gray","c_black",  
    "T-shirt/shirt","jacket/sweatshirt","overcoat","hoodie",
    "b_white","b_purple","b_pink","b_blue","b_green","b_red","b_brown","b_yellow","b_gray","b_black",  
    "backpack","bag/handbag",'no bags',
    "umbrella(open)","umbrella(closed)","no umbrella",
    "beard","shaved","hidden",
    "pants","shorts","skirt","unvisible",
    "l_white","l_purple","l_pink","l_blue","l_green","l_red","l_brown","l_yellow","l_gray","l_black",
    'formal shoes',"sneakers","high boots",'hidden',
    "f_white","f_purple","f_pink","f_blue","f_green","f_red","f_brown","f_yellow","f_gray","f_black",
    "sunglasses","headphone","gloves","scarf","tie",
    "front/back",
    "white", "black", "unkown"
    ]
new attributes : [0'gender', 1'hairless', 2"short hair",3"longhair(straight)",4"knot",5"unvisible(hair)",
                  6"burnette",7"blonde", 8"black",9'no-color',
                  10'cap',11"snowcap",12"hoodiecap",13"no cap", 14"unvisible(cap)",
                  15"c_white",16"c_blue",17"c_green",18"c_red",19"c_brown",20"c_yellow",21"c_gray",22"c_black",23'no-color', 
                  24"T-shirt/shirt",25"jacket/sweatshirt",26"overcoat",27"hoodie",
                  28"b_white",29"b_purple",30"b_pink",31"b_blue",32"b_green",33"b_red",34"b_brown",35"b_yellow",36"b_gray",37"b_black", 
                  38"backpack",39"bag/handbag",40'no bags',
                  41"umbrella(open)",42"umbrella(closed)",43"no umbrella",
                  44"beard",45"shaved",46"hidden",
                  47"pants",48"shorts",49"skirt",50"unvisible",
                  51"l_white",52"l_blue",53"l_green",54"l_red",55"l_brown",56"l_yellow",57"l_gray",58"l_black",59'no-color',
                  60'formal shoes',61"sneakers",62"high boots",63'hidden',
                  64"f_white",65"f_colorful",66"f_brown",67"f_gray",68"f_black",69'no-color',
                  70"sunglasses",71"headphone",72"gloves",73"scarf",74"tie",
                  75"front/back",
                  76"white", 77"black", 78"unkown"
                  ]                  
"""

import numpy as np
import os
import pickle

# #main_path = '/home/taarlab/anaconda3/envs/torchreid/deep-person-reid/my_osnet/DUKMTMC/Dataset/dukemtmc/DukeMTMC-reID/DukeMTMC-reID/bounding_box_train/'
# main_path = '/home/taarlab/anaconda3/envs/torchreid/deep-person-reid/my_osnet/DUKMTMC/Dataset/dukemtmc/DukeMTMC-reID/DukeMTMC-reID/bounding_box_test/'

# all_attr_path_tr = '/home/taarlab/SI_attributes/attributes/CA_Duke/trainlabel_final.pkl'
# attributes_dict = np.load(all_attr_path_tr,allow_pickle=True)

# attributes2 = np.array([attributes_dict[pth] for pth in attributes_dict])

# test_attr_path = '/home/taarlab/SI_attributes/attributes/CA_Duke/labeled test/final_attr_org.npy'
# attributes = np.load(test_attr_path)    
#attr_path = '/home/taarlab/SI_attributes/attributes/CA_Duke/final_attr_org.npy'
#ttributes =  np.load(attr_path)

path_attr = '/home/hossein/Downloads/generated/final_attr_org.npy'
path_stop = '/home/hossein/Downloads/generated/final_stop.npy'
stop_idx = np.load(path_stop)
attributes = np.load(path_attr)[:stop_idx]

main_path = '/home/hossein/SI_attributes/datasets/Dukemtmc/bounding_box_test'
img_names = os.listdir(main_path)
img_names.sort()
img_name = img_names[:len(attributes)]


id_ = []
for name in img_name:
    b = name.split('_')
    id_.append(int(b[0]))


atr_new = np.zeros((np.shape(attributes)[0], 79), dtype=int)

atr_new[:, :7] = attributes[:, :7] # ['gender','hairless', "short hair","longhair(straight)","knot","unvisible(hair)","burnette"]

#handle hair color, mix blonde and white/gray to each other, named blonde
for i in range(np.shape(attributes)[0]):
    for j in range(7,9):
        if attributes[i, j] == 1:
            atr_new[i, 7] = 1

atr_new[:, 8] = attributes[:, 9]  #black hair

# add no-color atrribute for hair when it's invisible
atr_new[:, 9] = attributes[:, 5] 

atr_new[:, 10:16] = attributes[:, 10:16] # ['gender','hairless', "short hair","longhair(straight)","knot","unvisible(hair)","burnette"]
atr_new[:, 16:18] = attributes[:, 18:20] #blue, green

#add pink, purple and red to red in cap
for i in range(np.shape(attributes)[0]):
    for j in (16,17,20):
        if attributes[i, j] == 1:
            atr_new[i, 18] = 1
            
atr_new[:, 19:23] = attributes[:, 21:25] #[brown, yellow, gray, black] - body

#add  no-color in cap part
for i in range(np.shape(attributes)[0]):
    for j in range(13,15):
        if attributes[i, j] == 1:
            atr_new[i, 23] = 1

# from Tshirt/shirt to l-white are the same
atr_new[:, 24:52] = attributes[:, 25:53] 
atr_new[:, 52:54] = attributes[:, 55:57] #blue, green - leg

#add pink, purple and red to red in cap
for i in range(np.shape(attributes)[0]):
    for j in (53, 54, 57):
        if attributes[i, j] == 1:
            atr_new[i, 54] = 1
            
atr_new[:, 55:59] = attributes[:, 58:62] #[brown, yellow, gray, black]
#add no color in leg
atr_new[:, 59] = attributes[:, 51]

atr_new[:, 60:65] = attributes[:, 62:67] #[shoes, f-white] 

#change purple, pink, blue, green, red and yellow to colorful
for i in range(np.shape(attributes)[0]):
    for j in (67,68,69,70,71,73):
        if attributes[i, j] == 1:
            atr_new[i, 65] = 1
            
            
atr_new[:, 66] = attributes[:, 72] #brown
atr_new[:, 67:69] = attributes[:, 74:76] #gray, black

#no color in hidden shoes
atr_new[:, 69] = attributes[:, 65] 

#accessories and viewpoint, race
atr_new[:, 70:79] = attributes[:, 76:85] 

sum_attr = np.sum(attributes, axis=0)
sum_attr_sum = np.sum(atr_new, axis=0)


attr_names_old =["male/female","hairless","short hair","longhair(straight)","knot","unvisible(hair)",
                 "burnette","blonde","gray/white","black",
                 'cap',"snowcap","hoodiecap","no cap", "unvisible(cap)",
                 "c_white","c_purple","c_pink","c_blue","c_green","c_red","c_brown","c_yellow","c_gray","c_black",  
                 "T-shirt/shirt","jacket/sweatshirt","overcoat","hoodie",
                 "b_white","b_purple","b_pink","b_blue","b_green","b_red","b_brown","b_yellow","b_gray","b_black",  
                 "backpack","bag/handbag",'no bags',
                 "umbrella(open)","umbrella(closed)","no umbrella",
                 "beard","shaved","hidden",
                 "pants","shorts","skirt","unvisible",
                 "l_white","l_purple","l_pink","l_blue","l_green","l_red","l_brown","l_yellow","l_gray","l_black",
                 'formal shoes',"sneakers","high boots",'hidden',
                 "f_white","f_purple","f_pink","f_blue","f_green","f_red","f_brown","f_yellow","f_gray","f_black",
                 "sunglasses","headphone","gloves","scarf","tie",
                 "front/back",
                 "white", "black", "unkown"
                 ]
attr_names_new = ['gender','hairless',"short hair","longhair(straight)","knot","unvisible(hair)",
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
print('new attribtes \n')
for i in range(len(attr_names_new)):
    print(i , ')', attr_names_new[i], '-->', int(sum_attr_sum[i]))
    


print('\n the old attributes \n')
for i in range(len(attr_names_old)):
    print(i , ')', attr_names_old[i], '-->', int(sum_attr[i]))
    
#%%

idd = np.reshape(np.array(id_), (len(id_),1))
attr_with_id = np.append(atr_new, idd, axis=1)
np.save('/home/hossein/SI_attributes/attributes/CA_Duke_test_with_id.npy',attr_with_id)

#%%
"""
attributes of PA100k : 
attrs = ['Female','AgeOver60','Age18-60','AgeLess18','Front','Side','Back',
        'Hat','Glasses','HandBag','ShoulderBag','Backpack','HoldObjectsInFront',
        'ShortSleeve','LongSleeve','UpperStride','UpperLogo','UpperPlaid',
        'UpperSplice','LowerStripe','LowerPattern','LongCoat','Trousers',
        'Shorts','Skirt&Dress','boots']
"""

import numpy as np
import os
from scipy import io
import pickle

main_path = 'E:/UT/NEW_WAY/DATASET\PA100k/release_data/release_data/'

# all_attr_path_tr = '/home/taarlab/SI_attributes/attributes/CA_Duke/trainlabel_final.pkl'
# attributes_dict = np.load(all_attr_path_tr,allow_pickle=True)
#
# attributes2 = np.array([attributes_dict[pth] for pth in attributes_dict])
#
# test_attr_path = '/home/taarlab/SI_attributes/attributes/CA_Duke/labeled test/final_attr_org.npy'
# attributes = np.load(test_attr_path)
attr_path = 'E:/UT/NEW_WAY/DATASET/PA100k/annotation.mat'
mat = io.loadmat(attr_path)
labtr = mat['train_label']
labval = mat['val_label']
labts = mat['test_label']
t = np.append(labtr, labval, axis= 0)
attributes = np.append(t, labts, axis= 0)

img_names = os.listdir(main_path)
img_names.sort()

id_ = []
for name in img_names:
    b = name.split('.')
    id_.append(int(b[0]))
print(id_)

attr_names = ['Female','AgeOver60','Age18-60','AgeLess18','Front','Side','Back',
              'Hat','Glasses','HandBag','ShoulderBag','Backpack','HoldObjectsInFront',
              'ShortSleeve','LongSleeve','UpperStride','UpperLogo','UpperPlaid',
              'UpperSplice','LowerStripe','LowerPattern','LongCoat','Trousers',
              'Shorts','Skirt&Dress','boots'
              ]
sum_attr = np.sum(attributes, axis=0)
print('new attribtes \n')
for i in range(len(attr_names)):
    print(i , ')', attr_names[i], '-->', int(sum_attr[i]))

idd = np.reshape(np.array(id_), (len(id_),1))
attr_with_id = np.append(attributes, idd, axis=1)
np.save('E:/UT/NEW_WAY/SI_attributes/attributes/PA100k_all_with_id.npy',attr_with_id)

#%%
import torch
import os
import numpy as np
import shutil

train_idx = torch.load('/home/taarlab/SI_attributes/attributes/train_idx_full.pth')
test_idx = torch.load('/home/taarlab/SI_attributes/attributes/test_idx_full.pth') 


def load_image_names(main_path):
    img_names = os.listdir(main_path)
    img_names.sort()    
    return np.array(img_names)


main_path = '/home/taarlab/SI_attributes/datasets/Market1501/Market-1501-v15.09.15/gt_bbox'
img_names = load_image_names(main_path)

attr_path = '/home/taarlab/SI_attributes/attributes/CA_Market_with_id.npy'
attr = np.load(attr_path)

train_dir = '/home/taarlab/SI_attributes/datasets/Market1501/Market-1501-v15.09.15/train_common'
test_dir = '/home/taarlab/SI_attributes/datasets/Market1501/Market-1501-v15.09.15/test_common'
for f in os.listdir(main_path):
    b = f.split('_')
    if int(b[0]) < 751:
        shutil.copy(f, train_dir)
# for name in img_names:
#     b = name.split('_')
#     if int(b[0]) < 751:
#         shutil.copy(name, train_dir)
        
#%% common attributes between CA-Market and CA-Duke

"""
common attributes between CA-Market and CA-Duke
common-attr = [gender, cap, hairless, short-hair, long-hair,, knot, h-colorful, h-black,
               b-white, b-red, b-yelllow, b-green, b-blue, b-gray, b-purple, b-black, 
               backpack, handbag, no-bag, pants, short, skirt, l-white, l-red, l-brown,
               l-yellow, l-green, l-blue, l-gray, l-black, shoes, hidden, no-color,
               f-white, f-colorful, f-black]

"""

camarket_attr = '/home/taarlab/SI_attributes/attributes/CA_Market_with_id.npy'
caduke_attr = '/home/taarlab/SI_attributes/attributes/CA_Duke_train_with_id.npy'
caduke_attr_ts = '/home/taarlab/SI_attributes/attributes/CA_Duke_test_with_id.npy'


def common_attr(path, key ='CA_Duke'):
    attributes = np.load(path)
    atr_new = np.zeros((np.shape(attributes)[0], 38), dtype=int)
    
    if key == "CA_Duke":
        atr_new[:,0] = attributes[:,0] #gender
        for i in range(np.shape(attributes)[0]):
            for j in (10, 11, 12):
                if attributes[i, j] == 1:
                    atr_new[i, 1] = 1 #cap
        atr_new[:,2:6] = attributes[:,1:5] #hair
        for i in range(np.shape(attributes)[0]):
            for j in (6,7,15,16,17,18,19,20,21):
                if attributes[i, j] == 1:
                    atr_new[i, 6] = 1    #hair-colorful
        for i in range(np.shape(attributes)[0]):
            for j in (8,22):
                if attributes[i, j] == 1:
                    atr_new[i, 7] = 1  
        atr_new[:,8] = attributes[:,28] #b-white
        atr_new[:,9] = attributes[:,33] #b-red
        atr_new[:,10] = attributes[:,35] #b-yellow
        atr_new[:,11] = attributes[:,32] #b-green
        atr_new[:,12] = attributes[:,31] #b-blue
        atr_new[:,13] = attributes[:,36] #b-gray
        atr_new[:,14] = attributes[:,29] #b-purple
        atr_new[:,15:19] = attributes[:,37:41] #b-black
        atr_new[:,19:22] = attributes[:,47:50] #gender
        atr_new[:,22] = attributes[:,51] #no-bag & leg and le-color
        atr_new[:,23:26] = attributes[:,54:57] #gender
        atr_new[:,26] = attributes[:,53] #gender
        atr_new[:,27] = attributes[:,52] #gender
        atr_new[:,28] = attributes[:,57] #gender
        atr_new[:,29] = attributes[:,54] #gender
        atr_new[:,30] = attributes[:,58] #gender
        for i in range(np.shape(attributes)[0]):
            for j in (60,61,62):
                if attributes[i, j] == 1:
                    atr_new[i, 31] = 1
        atr_new[:,32] = attributes[:,63] #gender
        atr_new[:,33] = attributes[:,69] #gender
        atr_new[:,34] = attributes[:,64] #gender
        for i in range(np.shape(attributes)[0]):
            for j in (65,66,67):
                if attributes[i, j] == 1:
                    atr_new[i, 35] = 1        
        atr_new[:,36] = attributes[:,68] #gender
        atr_new[:,37] = attributes[:,79] #gender
        
        
    if key == 'CA_Market':
        atr_new[:,0:8] = attributes[:,0:8] 
        atr_new[:,8:17] = attributes[:,13:22] 
        for i in range(np.shape(attributes)[0]):
            for j in (22,23):
                if attributes[i, j] == 1:
                    atr_new[i, 17] = 1  
                    
        atr_new[:,18:32] = attributes[:,24:38] 
        atr_new[:,32:37] = attributes[:,39:44]  
        atr_new[:,37] = attributes[:,48] 
        
    return atr_new
atr_duke_common = common_attr(caduke_attr, key ='CA_Duke')
np.save('/home/taarlab/SI_attributes/attributes/CA_Duke_train_common_with_id.npy',atr_duke_common)

atr_duke_common_ts = common_attr(caduke_attr_ts, key ='CA_Duke')
np.save('/home/taarlab/SI_attributes/attributes/CA_Duke_test_common_with_id.npy',atr_duke_common_ts)

atr_market_common = common_attr(camarket_attr, key ='CA_Market')
np.save('/home/taarlab/SI_attributes/attributes/CA_Market_common_with_id.npy',atr_market_common)


