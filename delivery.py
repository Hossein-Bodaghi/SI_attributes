#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 17:06:30 2021

@author: hossein
"""

import os
import numpy as np
from utils import load_attributes, load_image_names

            
def data_delivery(main_path,
                  path_attr=None,
                  need_id=False,
                  need_parts=False,
                  need_attr=True,
                  dataset = 'CA_Market'):
    '''
    
mode = ['CA_Market', 'Market_attribute', 'CA_Duke', 'Duke_attribute', 'PA100k]
    Parameters
    ----------
    main_path : TYPE string
        DESCRIPTION. the path of images folder
    path_attr : TYPE numpy array
        DESCRIPTION.
    need_parts : TYPE true/false
        DESCRIPTION. The default is False.
        if it is false returns a tuple containes a list of 
        image_names and their attributes in numpy and a list of ids  
    need_attr : when we want to see the whole attributes as a target vector 
    Returns
    -------
    '''

    output = {}
    attr_vec = load_attributes(path_attr) # numpy array
    if need_attr: output.update({'attributes':attr_vec[:,:-1]})
    
    if need_id: 
        output.update({'id':attr_vec[:,-1]})        
    img_names = load_image_names(main_path)
    if dataset != 'CA_Duke_Market': 
        output.update({'img_names':img_names,
                        'id':np.array([int(i.split('_')[0]) for i in img_names]),
                        'cam_id':np.array([int(i.split('_')[1][1]) for i in img_names])})
    else:
        output.update({'img_names':img_names,
                        'id':np.array([int(i.split('_')[1]) for i in img_names]),
                        'cam_id':np.array([int(i.split('_')[2][1]) for i in img_names])})
            
    if need_parts:  
        if dataset == 'CA_Market':                        
            output.update({'gender':attr_vec[:,0].reshape(len(attr_vec), 1),
                        'head':attr_vec[:,1:6],
                        'head_colour':attr_vec[:,6].reshape(len(attr_vec), 1),
                        'body':attr_vec[:,7:10],
                        'body_type':attr_vec[:,10].reshape(len(attr_vec), 1),
                        'body_colour':attr_vec[:,11:19],
                        'bags':attr_vec[:,19:22],
                        'leg':attr_vec[:,22:25],
                        'leg_colour':attr_vec[:,25:34],
                        'foot':attr_vec[:,34:37],                       
                        'foot_colour':attr_vec[:,37:41],
                        'age':attr_vec[:,41:45]})
            
        elif dataset == 'Market_attribute':
            output.update({'age':attr_vec[:,0:4],
                        'bags':attr_vec[:,4:7],
                        'leg_color':attr_vec[:,7:16],
                        'body_color':attr_vec[:,16:24],
                        'leg_type':attr_vec[:,24].reshape(len(attr_vec), 1),
                        'leg':attr_vec[:,25].reshape(len(attr_vec), 1),
                        'sleeve':attr_vec[:,26].reshape(len(attr_vec), 1),
                        'hair':attr_vec[:,27].reshape(len(attr_vec), 1),
                        'hat':attr_vec[:,28].reshape(len(attr_vec), 1),
                        'gender':attr_vec[:,29].reshape(len(attr_vec), 1)})        
            
        elif dataset == 'Duke_attribute':
            output.update({'bags':attr_vec[:,0:3],
                        'boot':attr_vec[:,3].reshape(len(attr_vec), 1),
                        'gender':attr_vec[:,4].reshape(len(attr_vec), 1),
                        'hat':attr_vec[:,5].reshape(len(attr_vec), 1),
                        'foot_color':attr_vec[:,6].reshape(len(attr_vec), 1),
                        'body':attr_vec[:,7].reshape(len(attr_vec), 1),
                        'leg_color':attr_vec[:,8:15],
                        'body_color':attr_vec[:,15:22]}) 
        
        elif dataset == "CA_Duke":
            output.update({'gender':attr_vec[:,0].reshape(len(attr_vec), 1),
                        'head':attr_vec[:,1:6],
                        'head_color':attr_vec[:,6:10],
                        'hat':attr_vec[:,10:15],
                        'cap_color':attr_vec[:,15:23],
                        'body':attr_vec[:,23:27],
                        'body_color':attr_vec[:,27:35],
                        'bags':attr_vec[:,35:38],
                        'face':attr_vec[:,38:41],
                        'leg':attr_vec[:,41:45],
                        'leg_color':attr_vec[:,45:53],
                        'foot':attr_vec[:,53:57],                       
                        'foot_color':attr_vec[:,57:63],
                        'accessories':attr_vec[:,63:70],
                        'position':attr_vec[:,70].reshape(len(attr_vec), 1),
                        'race':attr_vec[:,71:74]})
            
    if dataset == 'CA_Market':
        output.update({'names' : ['gender','cap','hairless','short hair','long hair',
                                  'knot', 'h_colorful/h_black','Tshirt/shirt', 'coat',
                                  'top','simple/patterned','b_w','b_r',
                                  'b_y','b_green','b_b',
                                  'b_gray','b_p','b_black','backpack', 'bag','no bag','pants',
                                  'short','skirt','l_w','l_r','l_br','l_y','l_green','l_b',
                                  'l_gray','l_p','l_black','shoes','sandal',
                                  'hidden','no color','f_w', 'f_colorful','f_black','young', 
                                  'teenager', 'adult', 'old']})

    elif dataset == 'Market_attribute':
        output.update({'names' : ['young', 'teenager', 'adult' ,'old', 'backpack', 'shoulder-bag', 
                                  'hand-bag', 'down-black','down-blue','down-brown',
                                  'down-gray','down-green','down-pink','down-purple','down-white',
                                  'down-yellow','up-black','up-blue','up-green','up-gray', 
                                  'up-purple','up-red','up-white','up-yellow','lower-body type', 
                                  'lower-body length','sleeve-length','hair-length','hat ','gender']})

    elif dataset == 'Duke_attribute':
        output.update({'names' : ['backpack','shoulder_bag','hand_bag','boots','gender','hat',
                                  'shoes_color','upper-body_length','downblack','downwhite',
                                  'downred','downgray','downblue','downgreen','downbrown',
                                  'upblack','upwhite','upred','uppurple','upgray',
                                  'upblue','upgreen','upbrown']})
                                  
    elif dataset == "CA_Duke":
        output.update({'names' : ['gender',
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
                                ]})
    elif dataset == "PA100k":
        output.update({'names': ['Female','AgeOver60','Age18-60','AgeLess18','Front','Side','Back',
                                 'Hat','Glasses','HandBag','ShoulderBag','Backpack','HoldObjectsInFront',
                                 'ShortSleeve','LongSleeve','UpperStride','UpperLogo','UpperPlaid',
                                 'UpperSplice','LowerStripe','LowerPattern','LongCoat','Trousers',
                                 'Shorts','Skirt&Dress','boots']})

            
    elif dataset == 'CA_Duke_Market':
        output.update({'names': ['gender','cap','hairless','short_hair','long_hair','knot','h_colorful','h_black',
                                 'b_white','b_red','b_yellow','b_green','b_blue','b_gray','b_purple','b_black',
                                 'backpack','hand_bag','no_bag','pants','short','skirt','l_w','l_r','l_br',
                                 'l_y','l_green','l_b','l_gray','l_p','l_black','shoes','hidden','no_color',
                                 'f_w','f_colorful','f_black']})        
    return output


    
def reid_delivery(main_path):
    names = os.listdir(main_path)
    names.sort()
    id_ = []
    cam_id = []
    img_names = []
    for name in names:
        b = name.split('_')
        if int(b[0]) != -1:
            id_.append(int(b[0]))
            cam_id.append(int(b[1][1]))
            img_names.append(name)
    return {'img_names':np.array(img_names),'id':np.array(id_), 'cam_id':np.array(cam_id)}