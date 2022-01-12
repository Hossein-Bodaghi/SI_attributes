#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 14:35:31 2021

@author: hossein
"""
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import math
from PIL import Image
import random
import torchvision.transforms as transforms
import torch.nn as nn



def resampler(clss,img_names,Most_repetition=5):
    max_num = max(sum(clss))
    raw_len = len(img_names)
    for i, num in enumerate(sum(clss)):
        if num < max_num:
            if (max_num-num)//num <Most_repetition-1:
                w=0
                while (w < (max_num-num)//num):
                    for k in range (raw_len):
                        if clss[k, i]==1:
                            a=clss[k, i]
                            img_names = np.append(img_names,img_names[k])
                            clss = torch.cat((clss , clss[torch.tensor([k])]),0)
                    w+=1
                j = 0
                random_idx_list = []
                while j < (max_num % num):
                    random_idx = torch.randint(raw_len,(1,))
                    if clss[random_idx,i] == 1 and random_idx not in random_idx_list:
                        img_names = np.append(img_names,img_names[random_idx])
                        clss = torch.cat((clss,clss[random_idx]),0)
                        random_idx_list.append(random_idx)
                        j+=1 
            elif (max_num-num)//num >= Most_repetition-1:
                w=0
                while (w < Most_repetition-1):
                    for k in range (raw_len):
                        if clss[k, i]==1:
                            a=clss[k, i]
                            img_names = np.append(img_names,img_names[k])
                            clss = torch.cat((clss , clss[torch.tensor([k])]),0)
                    w+=1
    return (clss,img_names)
 
def validation_idx(test_idx, ratio=5):
    idxs = []
    i = 0
    for idx in test_idx:
        i += 1
        if i % ratio == 0: idxs.append(idx)
        else: pass
    return idxs

def attr_number(attr):
    attr_numbers = {}
    for key in attr:
        if key == 'img_names' or key == 'id':
            pass
        else:
            number = torch.sum(attr[key], dim=0)
            attr_numbers.update({key:number})    
    return attr_numbers

def attr_weight(attr, device, effective = 'effective', beta = 0.99):
    
    if effective == 'effective':
        '''
        source: 
            https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf
        '''
        attr_numbers = attr_number(attr)
        attr_weights = {}
        for key in attr_numbers:
            weight = torch.ones_like(attr_numbers[key], dtype=torch.float32, device = device)
            for i, n in enumerate(attr_numbers[key]):
                w = (1 - beta) / (1 - beta**n.item())
                weight[i] = w
            attr_weights.update({key : weight})
        return attr_weights
    else:
        attr_numbers = attr_number(attr)
        attr_weights = {}
        for key in attr_numbers:
            weight = torch.ones_like(attr_numbers[key], dtype=torch.float32, device = device)
            attr_weights.update({key : weight})
        return attr_weights        
        

def part_data_delivery(weights, device, dataset='CA_Market'):
    '''
    Parameters
    ----------
    dataset : ['CA_Market', 'Market_attribute', 'CA_Duke', 'Duke_attribute']
        
    weiights : should be a dict of required parts and their weights        

    Returns
    -------
    dict
        for each key it contains the loss function of that part.

    '''
    loss_dict = {}
    
    if dataset == 'CA_Market':
        for key in weights:
            if key == 'body_type' or key == 'gender' or key == 'body_colour':
                loss_dict.update({key : nn.BCEWithLogitsLoss(pos_weight= weights[key]).to(device)})
            else:
                loss_dict.update({key:nn.CrossEntropyLoss(weight= weights[key]).to(device)})
    
    elif dataset == 'Market_attribute':
        
        for key in weights:
            if key == 'age' or key == 'bags' or key == 'leg_colour' or key == 'body_colour':
                loss_dict.update({key:nn.CrossEntropyLoss(weight= weights[key]).to(device)})
                
            else:
                loss_dict.update({key : nn.BCEWithLogitsLoss(pos_weight= weights[key]).to(device)})
    elif dataset == 'Duke_attribute':
        
        for key in weights:
            if key == 'bags' or key == 'leg_colour' or key == 'body_colour':
                loss_dict.update({key:nn.CrossEntropyLoss(weight= weights[key]).to(device)})
                
            else:
                loss_dict.update({key : nn.BCEWithLogitsLoss(pos_weight= weights[key]).to(device)})
        
    return loss_dict
def load_attributes(path_attr):
    attr_vec_np = np.load(path_attr)# loading attributes
        # attributes
    attr_vec_np = attr_vec_np.astype(np.int32)
    return torch.from_numpy(attr_vec_np)


def load_image_names(main_path):
    img_names = os.listdir(main_path)
    img_names.sort()    
    return np.array(img_names)


def unique(list1):
    # initialize a null list
    unique_list = []    
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return len(unique_list)


def one_hot_id(id_):
    num_ids = unique(id_)
    id_ = torch.from_numpy(np.array(id_))# becuase list doesnt take a list of indexes it should be slice or inegers.
    id1 = torch.zeros((len(id_),num_ids))
    
    sample = id_[0]
    i = 0
    for j in range(len(id1)):
        if sample == id_[j]:
           id1[j, i] = 1
        else:
            i += 1
            sample = id_[j]
            id1[j, i] = 1     
    return id1


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def plot(imgs, orig_img, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [orig_img[row_idx]] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()


def augmentor(orig_img, transform, num_aug=7):
    augmented = [transform(orig_img) for i in range(num_aug)]
    return augmented



class LGT(object):
    '''
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    this code is copied from:
        https://github.com/finger-monkey/Data-Augmentation.git
    '''
    def __init__(self, probability=0.2, sl=0.02, sh=0.4, r1=0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        new = img.convert("L")   # Convert from here to the corresponding grayscale image
        np_img = np.array(new, dtype=np.uint8)
        img_gray = np.dstack([np_img, np_img, np_img])

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size[1] and h < img.size[0]:
                x1 = random.randint(0, img.size[0] - h)
                y1 = random.randint(0, img.size[1] - w)
                img = np.asarray(img).astype('float')

                img[y1:y1 + h, x1:x1 + w, 0] = img_gray[y1:y1 + h, x1:x1 + w, 0]
                img[y1:y1 + h, x1:x1 + w, 1] = img_gray[y1:y1 + h, x1:x1 + w, 1]
                img[y1:y1 + h, x1:x1 + w, 2] = img_gray[y1:y1 + h, x1:x1 + w, 2]

                img = Image.fromarray(img.astype('uint8'))

                return img

        return img
    
class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    this code is copied and modified from:
        https://github.com/zhunzhong07/Random-Erasing.git
    '''
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.to_pil_image = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
    def __call__(self, img):
        img = self.to_tensor(img)

        if random.uniform(0, 1) > self.probability:
            return self.to_pil_image(img)

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return self.to_pil_image(img)

        return self.to_pil_image(img)