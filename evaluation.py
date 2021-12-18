#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 18:34:12 2021

@author: hossein
"""

import os
os.chdir('/home/hossein/anaconda3/envs/torchreid/deep-person-reid/my_osnet') 
#%%
import torchreid
"""
version v1 is:
    1) we consider whole vector of output as our target oposite 
    of v1 that we seperatly had a loss function for every collection
"""
from delivery import data_delivery 
from models import my_load_pretrain,MyOsNet,feature_model,MyOsNet2, CA_market_model5
from loaders import MarketLoader4, Market_folder_Loader
from metrics import tensor_metrics, boolian_metrics, tensor_metrics_detailes
import time
import torch
import torch.nn as nn 
from torchvision import transforms
from torch.utils.data import DataLoader

# import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print('calculation is on:',device)
torch.cuda.empty_cache()

#%%
def tensor_max(tensor):

    idx = torch.argmax(tensor, dim=1, keepdim=True)
    y = torch.zeros(tensor.size(),device=device).scatter_(1, idx, 1.)
    return y

def tensor_thresh(tensor, thr=0.5):
    out = (tensor>thr).float()
    return out    

def attr_evaluation(attr_net, test_loader, device, need_attr=True):
    attr_net.to(device)
    softmax = torch.nn.Softmax(dim=1)
    # evaluation:     
    attr_net.eval()
    with torch.no_grad():
        targets = []
        predicts = []
        for idx, data in enumerate(test_loader):
            for key, _ in data.items():
                data[key] = data[key].to(device)
            # forward step
            out_data = attr_net.get_feature(data['img'], get_attr=need_attr, get_feature=False)                      
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
                    
                y_target = torch.cat((data['gender'].unsqueeze(dim=1), data['head'],
                                      data['body'], data['body_type'].unsqueeze(dim=1),
                                      data['body_colour'], data['bags'],
                                      data['leg'], data['leg_colour'],
                                      data['foot'], data['foot_colour']), dim=1)  
                predicts.append(y_attr.to('cpu'))
                targets.append(y_target.to('cpu'))
            else:
                y_attr = tensor_thresh(torch.sigmoid(out_data['attr']))
                # y_attr = tensor_thresh(out_data['attr'], 0)
                y_target = data['attr'].to('cpu')  
                predicts.append(y_attr.to('cpu'))
                targets.append(y_target.to('cpu'))     
        predicts = torch.cat(predicts)
        targets = torch.cat(targets)
        test_attr_metrics = tensor_metrics(y_target.float(), y_attr)    
    return test_attr_metrics 



#%%
from torchreid import models

model = models.build_model(
    name='osnet_x1_0',
    num_classes=751,
    loss='softmax',
    pretrained=False
)

model = model.cuda()
#%%

attr_net_camarket = CA_market_model5(model=model,
                  feature_dim = 512,
                  num_id = 751,
                  attr_dim = 30,
                  need_id = False,
                  need_attr = True,
                  need_collection = False)

model_path = './result/V8_03/best_attr_net.pth'
trained_net = torch.load(model_path)
attr_net_camarket.load_state_dict(trained_net.state_dict())

#%%
for idx, m in enumerate(attr_net_camarket.children()):
    print(idx, '->', m) 
#%%
main_path = '/home/hossein/deep-person-reid/my_osnet/Market-1501-v15.09.15/gt_bbox/'
path_train = '/home/hossein/deep-person-reid/my_osnet/Market-1501-v15.09.15/bounding_box_train/'
path_attr = '/home/hossein/SI_attribute/attributes/gt_bbox_market_attribute.npy'
train_idx = torch.load( './attributes/train_idx_full.pth')
test_idx = torch.load('./attributes/test_idx_full.pth')

attr = data_delivery(main_path=main_path,
                     path_attr=path_attr,
                     need_collection=False,
                     double=False,
                     need_attr=True,
                     mode = 'Market_attribute')

test_data = MarketLoader4(img_path=main_path,
                          attr=attr,
                          resolution=(256, 128),
                          indexes=test_idx,
                          need_attr = True,
                          need_collection=False,
                          need_id = False,
                          two_transforms = False,                          
                          ) 
batch_size = 200
test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=False)

#%%
attr_metrics = attr_evaluation(attr_net_camarket, test_loader, device, need_attr=True)

#%%

attr_colomns = ['young','teenager','adult','old','backpack','Shoulder-bag','Hand-bag','Down-black','Down-blue',
'Down-brown','Down-gray','Down-green','Down-pink','Down-purple','Down-white','Down-yellow','Up-black','Up-blue',
'Up-green','Up-gray','Up-purple','Up-red','Up-white','Up-yellow','type of lower-body','length of lower-body',
'sleeve length','Hair-length','hat','Gender'
]

def metrics_print(attr_metrics, attr_colomns, metricss='f1'):
    n = 0
    if metricss == 'precision': n = 0
    elif metricss=='recall': n = 1
    elif metricss=='accuracy': n = 2
    elif metricss=='f1': n = 3 
    elif metricss=='mean_accuracy': n = 4  
    
    print('the result of',metricss+''+'\n')
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

    
metrics_print(attr_metrics, attr_colomns, metricss='f1')
total_metrics(attr_metrics)

def change2list(torch_tensor):
    list_tensor = []
    for met in torch_tensor:
        list_tensor.append(met.item())
    return list_tensor

precision_list = change2list(attr_metrics[0])
recall_list = change2list(attr_metrics[1])
accuracy_list = change2list(attr_metrics[2])
f1_list = change2list(attr_metrics[3])
mean_accuracy_list = change2list(attr_metrics[4])

