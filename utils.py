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
import torch.nn as nn
from loaders import get_image
from models import Loss_weighting
from torchvision import transforms
import pandas

def common_attr(predicts, targets):
    '''ca_market & pa100k 
      CA_Market	PA100k
     0      0	0
  1      10, 11,12	7
     2      70	8
     3      39	9,10
     4      26	21
     5      47	22
     6      48	23
     7      49	24
     8      62	25
     9      38  11
    '''
    if predicts.shape[1] == 79 and targets.shape[1] == 26:
        attr_names = ['gender', 'Hat','Glasses','bag','LongCoat','pants','shorts', 'skirt', 'boots', 'Backpack']
        
        new_predicts = torch.zeros((predicts.shape[0], 10))
        new_targets = torch.zeros((targets.shape[0], 10))
        
        new_predicts[:, 0] = predicts[:,0]
        new_targets[:, 0] = targets[:,0]            

        new_predicts[:, 1] = (predicts[:,10] + predicts[:,11] + predicts[:,12] )/3

        new_targets[:, 1] = targets[:,7]  

        new_predicts[:, 2] = predicts[:, 70]
        new_targets[:, 2] = targets[:, 8]   

        new_predicts[:, 3] = predicts[:, 39]
        new_targets[:, 3] = (targets[:,9]+targets[:,10])/2

        new_predicts[:, 4] = predicts[:, 26]
        new_targets[:, 4] = targets[:, 21]  

        new_predicts[:,5:8] = predicts[:, 47:50]
        new_targets[:,5:8] = targets[:, 22:25]             

        new_predicts[:,8] = predicts[:, 62]
        new_targets[:,8] = targets[:, 25]    

        new_predicts[:,9] = predicts[:, 38]
        new_targets[:,9] = targets[:, 11] 
        
    elif predicts.shape[1] == 37 and targets.shape[1] == 26:
        attr_names = ['gender','Hat','ShoulderBag','Backpack','pants','shorts','skirt']
        
        new_predicts = torch.zeros((predicts.shape[0], len(attr_names)))
        new_targets = torch.zeros((targets.shape[0], len(attr_names)))
        
        new_predicts[:, 0] = predicts[:,0]
        new_targets[:, 0] = targets[:,0]            

        new_predicts[:, 1] = predicts[:,1]
        new_targets[:, 1] = targets[:,7]  

        new_predicts[:, 2] = predicts[:,17]
        new_targets[:, 2] = targets[:, 10]   

        new_predicts[:, 3] = predicts[:, 16]
        new_targets[:, 3] = targets[:, 11]

        new_predicts[:,4:7] = predicts[:,19:22]
        new_targets[:,4:7] = targets[:,22:25]          
    return new_predicts, new_targets, attr_names

def metrics_print(attr_metrics, attr_colomns, metricss='precision'):
    n = 0
    if metricss == 'precision': n = 0
    elif metricss=='recall': n = 1
    elif metricss=='accuracy': n = 2
    elif metricss=='f1': n = 3 
    elif metricss=='mean_accuracy': n = 4  
    
    print('\n'+'the result of',metricss+'')
    non_zeros = []
    for idx, m in enumerate(attr_colomns):
        if attr_metrics[n][idx].item() == 0:
            pass
        else:
            non_zeros.append(attr_metrics[n][idx].item())
        print(idx, ')', m, '-->', attr_metrics[n][idx].item()) 

    mean = sum(non_zeros)/len(non_zeros)
    print(idx+1, ')', 'mean_nonzero', '-->', mean) 
    print(idx+1, ')', 'mean_withzero', '-->', torch.mean(attr_metrics[n]).item())


def total_metrics(attr_metrics): 
    metrices = ['precision_total',
            'recall_total',
            'accuracy_total',
            'f1_total', 
            'mean_accuracy_total']
    print('\n')
    for i in range(5):
        print(i, ')', metrices[i], '-->', attr_metrics[i+5]) 
        
def persian_csv_format(path_table, path_save, read='excel', sep_col=1):
    
    if read == 'excel':
        table1 = pandas.read_excel(path_table)
    else:
        table1 = pandas.read_csv(path_table)
    columns = table1.columns
    columns = columns[sep_col:]
    for column in columns:
        col_vals = table1.get(column)
        for idx, val in enumerate(col_vals):
            col_vals[idx] = '${:.2f}$'.format(100*val)
        table1[column] = col_vals
    table1.to_csv(path_save)
    return table1


def resampler(attr, clss, Most_repetition=5):
    max_num = max(sum(attr[clss]))
    raw_len = len(attr['img_names'])
    
    for i, num in enumerate(sum(attr[clss])):
        if num < max_num:
            if (max_num-num)//num < Most_repetition-1:
                w=0
                while (w < (max_num-num)//num):
                    for k in range (raw_len):
                        if attr[clss][k, i]==1:
                            for key in attr:
                                if key not in ['need_attr','need_collection','need_id','two_transforms','dataset','img_path','resolution','transform','normalizer']:
                                    if key == 'img_names':
                                        attr[key] = np.append(attr[key], attr[key][k])
                                    else:
                                        attr[key] = torch.cat((attr[key] , attr[key][torch.tensor([k])]),0)
                    w+=1
                j = 0
                random_idx_list = []
                while j < (max_num % num):
                    random_idx = torch.randint(raw_len,(1,))
                    if attr[clss][random_idx,i] == 1 and random_idx not in random_idx_list:
                        for key in attr:
                            if key not in ['need_attr','need_collection','need_id','two_transforms','dataset','img_path','resolution','transform','normalizer']:
                                if key == 'img_names':
                                    attr[key] = np.append(attr[key],attr[key][random_idx])
                                else:
                                    attr[key] = torch.cat((attr[key],attr[key][random_idx]),0)
                        random_idx_list.append(random_idx)
                        j+=1 
            elif (max_num-num)//num >= Most_repetition-1:
                w=0
                while (w < Most_repetition-1):
                    for k in range (raw_len):
                        if attr[clss][k, i]==1:
                            for key in attr:
                                if key not in ['need_attr','need_collection','need_id','two_transforms','dataset','img_path','resolution','transform','normalizer']:
                                    if key == 'img_names':
                                        attr[key] = np.append(attr[key],attr[key][k])
                                    else:                                    
                                        attr[key] = torch.cat((attr[key] , attr[key][torch.tensor([k])]),0)
                    w+=1
    return attr   


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
        if key == 'img_names' or key == 'id' or key == 'cam_id' or key == 'names':
            pass
        else:
            number = torch.sum(attr[key], dim=0)
            attr_numbers.update({key:number})    
    return attr_numbers

def Normalizer(x):
    if x.size()[0] > 1:
        maxx = x.max()
        minn = x.min()
        return (x - minn)/(maxx - minn)
    else:
        return torch.ones_like(x, dtype=torch.float32)

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
    elif effective == 'dynamic':
        attr_numbers = attr_number(attr)
        attr_weights = {}
        for key in attr_numbers:
            weight = Normalizer(attr_numbers[key].to(device))
            attr_weights.update({key : weight})
        return attr_weights        
    else:
        attr_numbers = attr_number(attr)
        attr_weights = {}
        for key in attr_numbers:
            weight = torch.ones_like(attr_numbers[key], dtype=torch.float32, device = device)
            attr_weights.update({key : weight})
        return attr_weights  
      
class BCE_Loss(torch.nn.Module):
    
    def __init__(self, weights=None):
        super(BCE_Loss,  self).__init__()
        self.weights = weights
        self.eps=1e-3
    def forward(self, y_pred, y_true):
      
        L = ( y_true * torch.log( self.eps + y_pred ) ) \
                        + ( self.eps + 1-y_true ) * torch.log( self.eps + 1 - y_pred )  #Estimate loss for each node
        #Put weights on classe losses (And Sum of one each ROW!!!)
        if self.weights is None:
            Sum = torch.sum( L , -1 )
        else: 
            Sum = torch.sum( self.weights * L , -1 ) 
        Loss =  torch.mean(torch.mean(Sum))            
        return -Loss          

def part_data_delivery(weights, device, dataset='CA_Market', dynamic=False):
    '''
    Parameters
    ----------
    dataset : ['CA_Market', 'Market_attribute', 'CA_Duke', 'Duke_attribute']
        
    weights : should be a dict of required parts and their weights        

    Returns
    -------
    dict
        for each key it contains the loss function of that part.

    '''
    loss_dict = {}
    if dynamic:
        for key in weights:
            loss_dict.update({key : BCE_Loss(weights= weights[key]).to(device)})
        
    else:
        if dataset == 'CA_Market':
            bces = ['body_type', 'gender', 'head_colour', 'body_colour', 'attributes']    
            for key in weights:
                if key in bces:
                    loss_dict.update({key : nn.BCEWithLogitsLoss(pos_weight= weights[key]).to(device)})
                else:
                    loss_dict.update({key:nn.CrossEntropyLoss(weight= weights[key]).to(device)})

        if dataset == 'CA_Duke':

            for key in weights:
                if key == 'body_type' or key == 'gender' or key == 'position' or key == 'accessories':
                    loss_dict.update({key : nn.BCEWithLogitsLoss(pos_weight= weights[key]).to(device)})
                else:
                    loss_dict.update({key : nn.CrossEntropyLoss(weight= weights[key]).to(device)})
                    
        elif dataset == 'Market_attribute':
            
            for key in weights:
                if key == 'age' or key == 'bags' or key == 'leg_color' or key == 'body_color':
                    loss_dict.update({key:nn.CrossEntropyLoss(weight= weights[key]).to(device)})
                    
                else:
                    loss_dict.update({key : nn.BCEWithLogitsLoss(pos_weight= weights[key]).to(device)})
        elif dataset == 'Duke_attribute':
            
            for key in weights:
                if key == 'bags' or key == 'leg_color' or key == 'body_color':
                    loss_dict.update({key:nn.CrossEntropyLoss(weight= weights[key]).to(device)})
                    
                else:
                    loss_dict.update({key:nn.BCEWithLogitsLoss(pos_weight= weights[key]).to(device)})
            
        elif dataset == 'PA100k':
            
            for key in weights:
                if key == 'bags' or key == 'leg_colour' or key == 'body_colour':
                    loss_dict.update({key:nn.CrossEntropyLoss(weight= weights[key]).to(device)})
                    
                else:
                    loss_dict.update({key:nn.BCEWithLogitsLoss(pos_weight= weights[key]).to(device)})
                    
        elif dataset == 'CA_Duke_Market':
            
            for key in weights:
                if key == 'bags' or key == 'leg_colour' or key == 'body_colour':
                    loss_dict.update({key:nn.CrossEntropyLoss(weight= weights[key]).to(device)})
                    
                else:
                    loss_dict.update({key:nn.BCEWithLogitsLoss(pos_weight= weights[key]).to(device)})
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




def plot(imgs, orig_img=None, with_orig=True, row_title=None, iou_result=None, **imshow_kwargs):
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
            if iou_result is not None:
                ax.set(title='{:.4f}'.format(iou_result[col_idx].item()))
                ax.title.set_size(8)

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()


def iou_worst_plot(iou_result, valid_idx, main_path, attr, num_worst = 5):

    min_sort_iou_idx = iou_result.sort()[1][:num_worst]
    min_sort_iou_result = iou_result.sort()[0][:num_worst]
    
    img_idx = [valid_idx[idx] for idx in min_sort_iou_idx]
    # make paths 
    img_paths = [os.path.join(main_path, attr['img_names'][i]) for i in img_idx]
    # load path as images
    orig_imgs = [get_image(addr,256, 128) for addr in img_paths]
    # plot augmented images
    plot(orig_imgs, with_orig=False, iou_result=min_sort_iou_result)
    
    
def map_evaluation(names, probability, targets):
    
    average_precision = []
    for i in range(len(names)):
        sorted, indices = torch.sort(probability[:,i], descending=True)
        correct_prediction = 0 # the total positive until that array
        running_sum = 0
        for j in range(len(indices)):
            idx = int(indices[j])
            if targets[idx, i] == 1:
                correct_prediction += 1
                running_sum += correct_prediction/(j+1)
            else:
                pass
        if correct_prediction != 0:
            average_precision.append(running_sum/correct_prediction)
        else:
            average_precision.append(0)
    mean_average_precision = sum(average_precision)/len(average_precision)
    return [average_precision, mean_average_precision]    


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
    
def show_loss(name,train_path,test_path,device):

    im = torch.load(test_path,map_location= torch.device(device))[:]
    im2 = torch.load(train_path,map_location= torch.device(device))[:]
    plt.figure('train')
    
    plt.xlabel('Epoch')
    plt.ylabel(name)
    plt.plot(im2, label='train')
    plt.plot(im, label = 'test')
    # plt.legend([plt_train, plt_test], ['train_'+name , 'val_'+name])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    for e, v in enumerate(im):
        if e%1 ==0:
            plt.text(e, v, '{:.2}'.format(v), color='g', fontsize= 'large') 
            plt.text(e, im2[e], '{:.2}'.format(im2[e]), color='r', fontsize= 'large')
    plt.show()
    
def show_loss_list(title, name, pathes, labels, device):
    
    plt.figure(title) 
    plt.xlabel('Epoch')
    plt.ylabel(name)
    for idx,path in enumerate(pathes):
        im = torch.load(path, map_location= torch.device(device))[:]
        plt.plot(im, label = labels[idx])
    # plt.legend([plt_train, plt_test], ['train_'+name , 'val_'+name])
    plt.legend(loc='lower right', borderaxespad=0.)
    plt.show()