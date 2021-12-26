#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 17:06:30 2021

@author: hossein
"""
import numpy as np
import torch
from utils import load_attributes, load_image_names, one_hot_id

            
def data_delivery(main_path,
                  path_attr=None,
                  need_id=False,
                  need_collection=False,
                  need_attr=True,
                  dataset = 'CA_Market'):
    '''
    
mode = ['CA_Market', 'Market_attribute', 'CA_Duke', 'Duke_attribute']
    Parameters
    ----------
    main_path : TYPE string
        DESCRIPTION. the path of images folder
    path_attr : TYPE numpy array
        DESCRIPTION.
    path_start : TYPE 
        DESCRIPTION.
    double : TYPE true/false
        DESCRIPTION. will double everything and return 
    need_collection : TYPE true/false
        DESCRIPTION. The default is False.
        if it is false returns a tuple containes a list of 
        image_names and their attributes in numpy and a list of ids  
    need_attr : when we want to see the whole attributes as a target vector 
    Returns
    only_id : when you need only id and id_weights. 
    -------
    None.

    '''
    output = {}
    attr_vec_np = load_attributes(path_attr) # numpy array
    attr_vec = torch.from_numpy(attr_vec_np) # torch tensor 
    if need_attr: output.update({'attributes':attr_vec})
    
    img_names = load_image_names(main_path)
    output.update({'img_names':img_names})
    
    if need_id:
        id1 = one_hot_id(attr_vec[:,-1])    
        output.update({'id':id1})

        
    
    if need_collection:
        
        if dataset == 'CA_Market':
            
            head = []
            head_color = []
            body = []
            body_type = []
            leg = []
            foot = []
            gender = []
            bags = []
            body_colour = []
            leg_colour = []
            foot_colour = []
            age = []        
            for vec in attr_vec_np:
                gender.append(vec[0])
                head.append(vec[1:6])
                head_color.append(vec[6:8])
                body.append(vec[8:12])
                body_type.append(vec[12])
                body_colour.append(vec[13:21])
                bags.append(vec[21:25])
                leg.append(vec[25:28])
                leg_colour.append(vec[28:37])
                foot.append(vec[37:40])
                foot_colour.append(vec[40:44])
                age.append(vec[44:48])
            
        output.update({'head':torch.from_numpy(np.array(head)),
                    'head_colour':torch.from_numpy(np.array(head_color)),
                    'body':torch.from_numpy(np.array(body)),
                    'body_type':torch.tensor(body_type),
                    'leg':torch.from_numpy(np.array(leg)),
                    'foot':torch.from_numpy(np.array(foot)),
                    'gender':torch.tensor(gender),
                    'bags':torch.from_numpy(np.array(bags)),
                    'body_colour':torch.from_numpy(np.array(body_colour)),
                    'leg_colour':torch.from_numpy(np.array(leg_colour)),
                    'foot_colour':torch.from_numpy(np.array(foot_colour)),
                    'age':torch.from_numpy(np.array(age))})
    return output


    
