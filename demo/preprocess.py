#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 10:48:16 2021

@author: hossein
"""
import torch
from PIL import Image
from torch.utils.data import Dataset 
from torchvision import transforms
import os
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''
*
a function to take an image from path and change its size to a new height and width 
it is different from library functions because consider the proportion of h/w of base image
and if the proportion of new h/w is different it will add a white background
'''

def tensor_max(tensor):

    idx = torch.argmax(tensor, dim=1, keepdim=True)
    y = torch.zeros(tensor.size(),device=device).scatter_(1, idx, 1.)
    return y

def tensor_thresh(tensor, thr=0.5):
    out = (tensor>thr).float()
    return out

def get_image(addr,height,width):

        test_image = Image.open(addr)
        ratio_w = width / test_image.width
        ratio_h = height / test_image.height
        if ratio_w < ratio_h:
          # It must be fixed by width
          resize_width = width
          resize_height = round(ratio_w * test_image.height)
        else:
          # Fixed by height
          resize_width = round(ratio_h * test_image.width)
          resize_height = height
        image_resize = test_image.resize((resize_width, resize_height), Image.ANTIALIAS)
        background = Image.new('RGB', (width, height), (255, 255, 255))
        offset = (round((width - resize_width) / 2), round((height - resize_height) / 2))
        background.paste(image_resize, offset)  
        return background     
    
class Demo_Market_Loader(Dataset):
    '''
    attr is a dictionary contains:
        1) img_names: names of images in source path
        2) id is the identification number of each picture
    img_path: the folder of our source images. '/home/hossein/reid-data/market1501/Market-1501-v15.09.15/gt_bbox/'
    resolution: the final dimentions of images (height,width) (256,128)
    transform: images transformations
    
    
    '''
    def __init__(self,img_path,img_names,resolution, _id=None):

         
        self.img_path = img_path
        self.img_names = img_names    
        self.resolution = resolution
        
        self._id = _id

        self.transform_simple = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self,idx):
        
        img_path = os.path.join(self.img_path, self.img_names[idx])
        img = get_image(img_path, self.resolution[0], self.resolution[1])
        
        img = self.transform_simple(img)
        out = {'img' : img}
        if self._id:
            out.update({'id':self._id[idx]})        
        return out
    
def attr_evaluation(attr_net, img , device, need_attr=True):
    attr_net.to(device)
    softmax = torch.nn.Softmax(dim=1)
    # evaluation:     
    attr_net.eval()
    with torch.no_grad():
        # forward step
        out_data = attr_net.get_feature(img, get_attr=need_attr, get_feature=False)                      
        if not need_attr:
            # compute losses and evaluation metrics:
            # head 
            y_head = tensor_max(softmax(out_data['head']))                
            # body
            y_body = tensor_max(softmax(out_data['body']))
            # body_type 
            y_body_type = tensor_thresh(torch.sigmoid(out_data['body_type']), 0.5)
            # leg
            y_leg = tensor_max(softmax(out_data['leg']))                
            # foot 
            y_foot = tensor_max(softmax(out_data['foot']))                
            # gender
            y_gender = tensor_thresh(torch.sigmoid(out_data['gender']), 0.5)
            # bags
            y_bags = tensor_max(softmax(out_data['bags']))                
            # body_colour 
            y_body_colour = tensor_thresh(torch.sigmoid(out_data['body_colour']), 0.5)                
            # leg_colour
            y_leg_colour = tensor_max(softmax(out_data['leg_colour']))                
            # foot_colour
            y_foot_colour = tensor_max(softmax(out_data['foot_colour']))
            y_attr = torch.cat((y_gender, y_head, y_body, 
                                y_body_type, y_body_colour,
                                y_bags, y_leg, y_leg_colour,
                                y_foot, y_foot_colour), dim=1)
        else:
            y_attr = tensor_thresh(torch.sigmoid(out_data['attr']))

    return y_attr.to(device)


class CA_market_model2(nn.Module):
    
    '''
    this is our network in this version it just take output from features of
    original omni-scale network.
    
    if attr_inc=True then for each attribute has a seperate linear 
    layer for classification
    
    if id_inc=True the output of attribute detection and models features will be concatanated
    and then a clasiification will predict the id of input picture
    
    in this version forward function and predict function defined seperatetly 
    in forward we dont have 
    '''
    
    def __init__(self,
                 model,
                 num_id,
                 feature_dim = 1000,
                 attr_dim = 46,
                 need_id = True,
                 need_attr = True,
                 need_collection = True):
        
        super().__init__()
        self.feature_dim = feature_dim
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.batchnormalization = nn.BatchNorm1d(num_features=feature_dim, momentum=0.1, affine=True, track_running_stats=True)
        self.batchnormalization2 = nn.BatchNorm1d(num_features=feature_dim + attr_dim, momentum=0.1, affine=True, track_running_stats=True)

        self.model = model
        self.linear = nn.Linear(in_features=feature_dim , out_features=feature_dim)
        self.id_lin = nn.Linear(in_features=feature_dim , out_features=num_id)
        self.head_lin = nn.Linear(in_features=feature_dim , out_features=5)
        self.body_lin = nn.Linear(in_features=feature_dim , out_features=3)
        self.body_type_lin = nn.Linear(in_features=feature_dim , out_features=1)
        self.leg_lin = nn.Linear(in_features=feature_dim , out_features=3)
        self.foot_lin = nn.Linear(in_features=feature_dim , out_features=3)
        self.gender_lin = nn.Linear(in_features=feature_dim , out_features=1)
        self.bags_lin = nn.Linear(in_features=feature_dim , out_features=3)
        self.body_colour_lin = nn.Linear(in_features=feature_dim , out_features=9)
        self.leg_colour_lin = nn.Linear(in_features=feature_dim , out_features=9)
        self.foot_colour_lin = nn.Linear(in_features=feature_dim , out_features=9)       
        self.attr_lin = nn.Linear(in_features=feature_dim , out_features=attr_dim)  
        
        self.need_id = need_id
        self.need_attr = need_attr
        self.need_collection = need_collection
        
    def get_feature(self, x, get_attr=True, get_feature=True, get_collection=False):
        
        features = self.model(x)
        features = features.view(-1,self.feature_dim) 
        features = self.batchnormalization(features)
        out_features = {}
        if get_feature:
            out_features.update({'features':features})
        if get_attr:
            out_attr = self.attr_lin(features)
            out_features.update({'attr':out_attr})
        if get_collection:
            out_head = self.head_lin(features)          
            out_body = self.body_lin(features)           
            out_body_type = self.body_type_lin(features)          
            out_leg = self.leg_lin(features)          
            out_foot = self.foot_lin(features)          
            out_gender = self.body_type_lin(features)           
            out_bags = self.bags_lin(features)          
            out_body_colour = self.body_colour_lin(features)           
            out_leg_colour = self.leg_colour_lin(features)            
            out_foot_colour = self.foot_colour_lin(features)  
            out_features.update({'head':out_head,
                                     'body':out_body,
                                     'body_type':out_body_type,
                                     'leg':out_leg,
                                     'foot':out_foot,
                                     'gender':out_gender,
                                     'bags':out_bags,
                                     'body_colour':out_body_colour,
                                     'leg_colour':out_leg_colour,
                                     'foot_colour':out_foot_colour})
        return out_features
        
        
    def forward(self, x):
        
        features = self.model(x) 
        features = features.view(-1,self.feature_dim)              
        features = self.batchnormalization(features)
        features = self.dropout(features) 
                    
        
        if self.need_attr and not self.need_collection:
            
            out_attr = self.attr_lin(features)

            if self.need_id:
                out_id = self.id_lin(features)
                return {'id':out_id, 'attr':out_attr}
            else: 
                return {'id':out_attr}
        
        elif self.need_collection and not self.need_attr:
            out_head = self.head_lin(features)          
            out_body = self.body_lin(features)           
            out_body_type = self.body_type_lin(features)          
            out_leg = self.leg_lin(features)          
            out_foot = self.foot_lin(features)          
            out_gender = self.body_type_lin(features)           
            out_bags = self.bags_lin(features)          
            out_body_colour = self.body_colour_lin(features)           
            out_leg_colour = self.leg_colour_lin(features)            
            out_foot_colour = self.foot_colour_lin(features) 
            
            if self.need_id:                              
                out_id = self.id_lin(features)
                
                return {'head':out_head,
                         'body':out_body,
                         'body_type':out_body_type,
                         'leg':out_leg,
                         'foot':out_foot,
                         'gender':out_gender,
                         'bags':out_bags,
                         'body_colour':out_body_colour,
                         'leg_colour':out_leg_colour,
                         'foot_colour':out_foot_colour,
                         'id':out_id}
            else:

                return {'head':out_head,
                         'body':out_body,
                         'body_type':out_body_type,
                         'leg':out_leg,
                         'foot':out_foot,
                         'gender':out_gender,
                         'bags':out_bags,
                         'body_colour':out_body_colour,
                         'leg_colour':out_leg_colour,
                         'foot_colour':out_foot_colour}
    def save_baseline(self, save_path, name):
        import os
        os.chdir('/home/hossein/anaconda3/envs/torchreid/deep-person-reid/my_osnet/LUPerson/fast_reid/') 
        from fastreid.utils.checkpoint import Checkpointer
        Checkpointer(self.model,save_path).save(name=name)