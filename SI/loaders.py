#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:07:23 2021

@author: hossein

here we can find different types of loaders 
that are define for person-attribute detection. 
this is Hossein Bodaghies thesis
"""


import torch
import numpy as np
from PIL import Image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
'''
*
a function to take an image from path and change its size to a new height and width 
it is different from library functions because consider the proportion of h/w of base image
and if the proportion of new h/w is different it will add a white background
'''

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
   
#%%
'''
first loader 10/99

if you dont want to use dataloader from pytorch
market costume data loader
'''    
from torch.utils.data import Dataset 
from torchvision import transforms

class MarketLoader(Dataset):
    '''
    attr is a dictionary contains
    1) attributes (N,M) N is the number data and M is the number of attributes
    2) id is the identification number of each picture
    3) frequencies
    4) image_names
    img_path: the folder of our source images. '/home/hossein/reid-data/market1501/Market-1501-v15.09.15/gt_bbox/'
    resolution: the final dimentions of images (height,width) (256,128)
    transform: images transformations    
    '''
    def __init__(self,img_path,attr,resolution,transform,indexes):
        self.img_path = img_path
        
        self.id = attr['id'][indexes]
        self.attr = attr['attributes'][indexes]
        self.img_names = attr['img_names'][indexes]
        self.resolution = resolution
        self.transform_complex = transform            
        self.transform_simple = transforms.Compose([transforms.RandomRotation(degrees=15),
                                                    transforms.RandomVerticalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.attr)
    
    def __getitem__(self,idx):
        
        img = get_image(self.img_path+self.img_names[idx], self.resolution[0], self.resolution[1])
        # img = get_image(self.img_path+self.img_names[idx], self.resolution[0], self.resolution[1])
        # we have doublicated the images path so in one epoch half of images will be 
        # augmented and the other half wont change 
        t = torch.empty(1).random_(2)
        if t == 0:
            self.transform = self.transform_complex
        else:
            self.transform = self.transform_simple
        
        img = self.transform(img)
    
        return (img.to(device),
                self.id[idx].to(device),
                self.attr[idx].to(device))  
    
    
class MarketLoader2(Dataset):
    '''
    attr is a dictionary contains:
        
        1) head (cap,bald,sh,lhs,lhn)
        2) body (shirt,coat,top) 
        3) body_type (simple,patterned)
        4) leg (pants,shorts,skirt)
        5) foot (shoes,sandal,hidden)
        6) gender (male,female)
        7) bags (backpack,hand bag,nothing)
        8) body_colour (9 colours)
        9) leg_colour (9 colours)
        10) foot_colour (9 colours)
        11) img_names: names of images in source path
        12) id is the identification number of each picture
    img_path: the folder of our source images. '/home/hossein/reid-data/market1501/Market-1501-v15.09.15/gt_bbox/'
    resolution: the final dimentions of images (height,width) (256,128)
    transform: images transformations
    
    
    '''
    def __init__(self,img_path,attr,resolution,transform,indexes):

         
        self.img_path = img_path
        
        self.id = attr['id'][indexes]
        self.img_names = attr['img_names'][indexes]
        self.head = attr['head'][indexes]
        self.body = attr['body'][indexes]
        self.body_type = attr['body_type'][indexes]
        self.leg = attr['leg'][indexes]
        self.foot = attr['foot'][indexes]
        self.gender = attr['gender'][indexes]
        self.bags = attr['bags'][indexes]
        self.body_colour = attr['body_colour'][indexes]
        self.leg_colour = attr['leg_colour'][indexes]
        self.foot_colour = attr['foot_colour'][indexes]
        
        self.resolution = resolution
        
        self.transform_complex = transform            

        self.transform_simple = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            # imagenet normalization
            
    def __len__(self):
        return len(self.head)
    
    def __getitem__(self,idx):
        
        img = get_image(self.img_path+self.img_names[idx], self.resolution[0], self.resolution[1])
       
        # we have doublicated the images path so in one epoch half of images will be 
        # augmented and the other half wont change        
        
        t = torch.empty(1).random_(2)
        if t == 0:
            self.transform = self.transform_complex
        else:
            self.transform = self.transform_simple
        img = self.transform(img)
        return (img,
                self.id[idx].to(device),
                self.head[idx].to(device),
                self.body[idx].to(device),
                self.body_type[idx].to(device),
                self.leg[idx].to(device),
                self.foot[idx].to(device),
                self.gender[idx].to(device),
                self.bags[idx].to(device),
                self.body_colour[idx].to(device),
                self.leg_colour[idx].to(device),
                self.foot_colour[idx].to(device))
    
class MarketLoader3(Dataset):
    '''
    attr is a dictionary contains:
        
        1) head (cap,bald,sh,lhs,lhn)
        2) body (shirt,coat,top) 
        3) body_type (simple,patterned)
        4) leg (pants,shorts,skirt)
        5) foot (shoes,sandal,hidden)
        6) gender (male,female)
        7) bags (backpack,hand bag,nothing)
        8) body_colour (9 colours)
        9) leg_colour (9 colours)
        10) foot_colour (9 colours)
        11) img_names: names of images in source path
        12) id is the identification number of each picture
    img_path: the folder of our source images. '/home/hossein/reid-data/market1501/Market-1501-v15.09.15/gt_bbox/'
    resolution: the final dimentions of images (height,width) (256,128)
    transform: images transformations
    
    
    '''
    def __init__(self,img_path,attr,resolution,transform,indexes):

         
        self.img_path = img_path
        
        self.id = attr['id'][indexes]
        self.img_names = attr['img_names'][indexes]
        self.head = attr['head'][indexes]
        self.body = attr['body'][indexes]
        self.body_type = attr['body_type'][indexes]
        self.leg = attr['leg'][indexes]
        self.foot = attr['foot'][indexes]
        self.gender = attr['gender'][indexes]
        self.bags = attr['bags'][indexes]
        self.body_colour = attr['body_colour'][indexes]
        self.leg_colour = attr['leg_colour'][indexes]
        self.foot_colour = attr['foot_colour'][indexes]
        
        self.resolution = resolution
        
        self.transform_complex = transform            

        self.transform_simple = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            # imagenet normalization
            
    def __len__(self):
        return len(self.head)
    
    def __getitem__(self,idx):
        
        img = get_image(self.img_path+self.img_names[idx], self.resolution[0], self.resolution[1])
       
        # we have doublicated the images path so in one epoch half of images will be 
        # augmented and the other half wont change        
        
        t = torch.empty(1).random_(2)
        if t == 0:
            self.transform = self.transform_complex
        else:
            self.transform = self.transform_simple
        img = self.transform(img)
        return (img,
                self.id[idx],
                self.head[idx],
                self.body[idx],
                self.body_type[idx],
                self.leg[idx],
                self.foot[idx],
                self.gender[idx],
                self.bags[idx],
                self.body_colour[idx],
                self.leg_colour[idx],
                self.foot_colour[idx])

class MarketLoader4(Dataset):
    '''
    attr is a dictionary contains:
        
        1) head (cap,bald,sh,lhs,lhn)
        2) body (shirt,coat,top) 
        3) body_type (simple,patterned)
        4) leg (pants,shorts,skirt)
        5) foot (shoes,sandal,hidden)
        6) gender (male,female)
        7) bags (backpack,hand bag,nothing)
        8) body_colour (9 colours)
        9) leg_colour (9 colours)
        10) foot_colour (9 colours)
        11) img_names: names of images in source path
        12) id is the identification number of each picture
    img_path: the folder of our source images. '/home/hossein/reid-data/market1501/Market-1501-v15.09.15/gt_bbox/'
    resolution: the final dimentions of images (height,width) (256,128)
    transform: images transformations
    
    
    '''
    def __init__(self,img_path,
                 attr,
                 resolution,
                 indexes,
                 transform=None,
                 need_attr = True,
                 need_collection=True,
                 need_id = True,
                 two_transforms = True,
                 train_ids = None):
        
        # conditional variables:
        self.need_attr = need_attr
        self.need_collection = need_collection
        self.need_id = need_id
        self.two_transforms = two_transforms
        
        # images variables:
        self.img_path = img_path
        self.img_names = attr['img_names'][indexes]
        self.resolution = resolution
        
        # id variables:
        if self.need_id:
            self.id = train_ids
        
        # attributes variables:
        if self.need_collection:
            self.head = attr['head'][indexes]
            self.body = attr['body'][indexes]
            self.body_type = attr['body_type'][indexes]
            self.leg = attr['leg'][indexes]
            self.foot = attr['foot'][indexes]
            self.gender = attr['gender'][indexes]
            self.bags = attr['bags'][indexes]
            self.body_colour = attr['body_colour'][indexes]
            self.leg_colour = attr['leg_colour'][indexes]
            self.foot_colour = attr['foot_colour'][indexes]
        if self.need_attr:
            self.attr = attr['attributes'][indexes]           
        
        # transform variables:
        if transform:
            self.transform = transform
        else:
            self.transform = None
            
        self.normalizer = transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self,idx):
        
        img = get_image(self.img_path+self.img_names[idx], self.resolution[0], self.resolution[1])
        
        if self.transform:
            if self.two_transforms:
                t = torch.empty(1).random_(2)
                if t == 0:
                    img = self.transform(img) 
            else:
                img = self.transform(img) 
        
        img = self.normalizer(img)
        
        out = {'img' : img}
        
        if self.need_attr:
            out.update({'attr':self.attr[idx]})
        if self.need_collection:
            out.update({
                'head':self.head[idx],
                'body':self.body[idx],
                'body_type':self.body_type[idx],
                'leg':self.leg[idx],
                'foot':self.foot[idx],
                'gender':self.gender[idx],
                'bags':self.bags[idx],
                'body_colour':self.body_colour[idx],
                'leg_colour':self.leg_colour[idx],
                'foot_colour':self.foot_colour[idx]                
                })   
        if self.need_id:
            out.update({'id':self.id[idx]})
            
        return out
            
    
    
class Market_folder_Loader(Dataset):
    '''
    attr is a dictionary contains:
        1) img_names: names of images in source path
        2) id is the identification number of each picture
    img_path: the folder of our source images. '/home/hossein/reid-data/market1501/Market-1501-v15.09.15/gt_bbox/'
    resolution: the final dimentions of images (height,width) (256,128)
    transform: images transformations
    
    
    '''
    def __init__(self,img_path,attr,resolution, need_id=True):

         
        self.img_path = img_path
        
        self.id = attr['id']
        self.img_names = attr['img_names']     
        self.resolution = resolution

        self.need_id = need_id

        self.transform_simple = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

            # imagenet normalization
            
    def __len__(self):
        return len(self.id)
    
    def __getitem__(self,idx):
        
        img = get_image(self.img_path+self.img_names[idx], self.resolution[0], self.resolution[1])
        
        img = self.transform_simple(img)
        out = {'img' : img}
        if self.need_id:
            out.update({'id':self.id[idx]})        
        return out