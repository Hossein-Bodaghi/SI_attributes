#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 17:06:30 2021

@author: hossein
"""
import numpy as np
import torch
import os 

def data_delivery(main_path,
                  path_attr=None,
                  path_start=None,
                  path_market_attr=None,
                  only_id=False,
                  double = True,
                  need_collection=False,
                  need_attr=True,
                  CA_Market = True,
                  mode = 'CA_Market'):
    '''
    

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
    
    if path_start:
            # loading attributes
        start_point = np.load(path_start)
        attr_vec_np = np.load(path_attr)# loading attributes
    
            # attributes
        attr_vec_np = attr_vec_np.astype(np.int32)
        attr_vec_np = attr_vec_np[:start_point]
        if double:
            attr_vec_np = np.append(attr_vec_np,attr_vec_np,axis=0)
    else:
        attr_vec_np = np.load(path_attr)# loading attributes
            # attributes
        attr_vec_np = attr_vec_np.astype(np.int32)
        if double:
            attr_vec_np = np.append(attr_vec_np,attr_vec_np,axis=0)        
        
    # images names    
    img_names = os.listdir(main_path)
    img_names.sort()
    if only_id:
        pass
    else:
        if path_start:
            img_names = img_names[:start_point]
        else:
            pass
    img_names = np.array(img_names)
    if double:
        img_names = list(np.append(img_names,img_names,axis=0))
        
        # ids & ids_weights
    id_ = []
    cam_id = []
    for name in img_names:
        b = name.split('_')
        id_.append(int(b[0])-1)
        cam_id.append(int(b[1][1]))
    cam_id = np.array(cam_id)
    id_ = torch.from_numpy(np.array(id_))# becuase list doesnt take a list of indexes it should be slice or inegers.
    
    # one hot id vectors
    last_id = id_[-1]
    id1 = torch.zeros((len(id_),last_id+1))
    for i in range(len(id1)):
        a = id_[i]
        id1[i,a] = 1
        
    # numbers = torch.unique(id_) # return individual numbers in a tensor
    sum_ids_train = torch.sum(id1, axis=0)
    n_samples, n_classes = id1.size()
    ids_weights = []
    for n_sample in sum_ids_train:
        w = n_samples/(n_classes*n_sample).type(torch.float16)
        w = w.item()
        ids_weights.append(w)
    
    if only_id:
        return {'img_names':np.array(img_names),'id':id_,'id_weights':torch.from_numpy(np.array(ids_weights)), 'cam_id':cam_id}
    
    if mode == 'CA_Market':
        attr_vec =  np.delete(attr_vec_np, slice(6,15), axis=1)
    elif mode == 'Market_attribute':
        attr_vec = attr_vec_np
    attr_vec = torch.from_numpy(attr_vec)
    sum_attr_train = torch.sum(attr_vec, axis=0)
    n_samples, n_classes = attr_vec.size()
    attr_weights = []
    for n_sample in sum_attr_train:
        w = (n_samples/(n_classes*n_sample)).type(torch.float16)
        w = w.item()
        attr_weights.append(w)      
        
            
    if need_attr and not need_collection:
        return {'id':id1,
                'id_weights':torch.from_numpy(np.array(ids_weights)),
                'img_names':np.array(img_names),
                'attr_weights':torch.from_numpy(np.array(attr_weights)),
                'attributes':attr_vec,
                'cam_id':cam_id} 
    
    if need_collection:
        head = []
        body = []
        body_type = []
        leg = []
        foot = []
        gender = []
        bags = []
        body_colour = []
        leg_colour = []
        foot_colour = []
        
        for vec in attr_vec_np:
            
            gender.append(vec[0])
            head.append(vec[1:6])
            body.append(vec[15:18])
            body_type.append(vec[18])
            leg.append(vec[31:34])
            foot.append(vec[43:46])
            bags.append(vec[28:31])
            body_colour.append(vec[19:28])
            leg_colour.append(vec[34:43])
            foot_colour.append(vec[46:])
            
        # one hot id vectors
        last_id = id_[-1]
        id1 = torch.zeros((len(id_),last_id))
        for i in range(len(id1)):
            a = id_[i]
            id1[i,a-1] = 1
            
        if need_collection and not need_attr:    
            return {'id':id1,
                    'id_weights':torch.from_numpy(np.array(ids_weights)),
                    'attr_weights':torch.from_numpy(np.array(attr_weights)),
                    'img_names':np.array(img_names),
                    'head':torch.from_numpy(np.array(head)),
                    'body':torch.from_numpy(np.array(body)),
                    'body_type':torch.tensor(body_type),
                    'leg':torch.from_numpy(np.array(leg)),
                    'foot':torch.from_numpy(np.array(foot)),
                    'gender':torch.tensor(gender),
                    'bags':torch.from_numpy(np.array(bags)),
                    'body_colour':torch.from_numpy(np.array(body_colour)),
                    'leg_colour':torch.from_numpy(np.array(leg_colour)),
                    'foot_colour':torch.from_numpy(np.array(foot_colour)),
                    'cam_id':cam_id}
    
        if need_collection and need_attr:
            return {'id':id1,
                    'id_weights':torch.from_numpy(np.array(ids_weights)),
                    'attributes':attr_vec,
                    'attr_weights':torch.from_numpy(np.array(attr_weights)),
                    'img_names':np.array(img_names),
                    'head':torch.from_numpy(np.array(head)),
                    'body':torch.from_numpy(np.array(body)),
                    'body_type':torch.tensor(body_type),
                    'leg':torch.from_numpy(np.array(leg)),
                    'foot':torch.from_numpy(np.array(foot)),
                    'gender':torch.tensor(gender),
                    'bags':torch.from_numpy(np.array(bags)),
                    'body_colour':torch.from_numpy(np.array(body_colour)),
                    'leg_colour':torch.from_numpy(np.array(leg_colour)),
                    'foot_colour':torch.from_numpy(np.array(foot_colour)),
                    'cam_id':cam_id}            
            


    
