#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 17:06:30 2021

@author: hossein
"""

from utils import load_attributes, load_image_names

            
def data_delivery(main_path,
                  path_attr=None,
                  need_id=False,
                  need_parts=False,
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
    if need_attr: output.update({'attributes':attr_vec})
    
    img_names = load_image_names(main_path)
    output.update({'img_names':img_names,'id':attr_vec[:,-1]})
    
    if need_parts:  
        if dataset == 'CA_Market':                        
            output.update({'gender':attr_vec[:,0].reshape(len(attr_vec), 1),
                        'head':attr_vec[:,1:6],
                        'head_colour':attr_vec[:,6:8],
                        'body':attr_vec[:,8:12],
                        'body_type':attr_vec[:,12].reshape(len(attr_vec), 1),
                        'body_colour':attr_vec[:,13:21],
                        'bags':attr_vec[:,21:25],
                        'leg':attr_vec[:,25:28],
                        'leg_colour':attr_vec[:,28:37],
                        'foot':attr_vec[:,37:40],                       
                        'foot_colour':attr_vec[:,40:44],
                        'age':attr_vec[:,44:48]})
            
        elif dataset == 'Market_attribute':
            output.update({'age':attr_vec[:,0:4],
                        'bags':attr_vec[:,4:7],
                        'leg_colour':attr_vec[:,7:16],
                        'body_colour':attr_vec[:,16:24],
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
                        'foot_colour':attr_vec[:,6].reshape(len(attr_vec), 1),
                        'body':attr_vec[:,7].reshape(len(attr_vec), 1),
                        'leg_colour':attr_vec[:,8:15],
                        'body_colour':attr_vec[:,15:22]}) 
    return output


    
