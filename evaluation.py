#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 18:34:12 2021

@author: hossein
"""

#%%
import torchreid
from metrics import tensor_metrics, boolian_metrics, tensor_metrics_detailes
import time
import os
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

# networks_main = '/home/hossein/SI_attributes/results/'
# training_types = os.listdir(networks_main)


# for training_type in training_types:
#     typee = training_type.split('_')[2]
#     wei_status = training_type.split('_')[3]
#     dataset = training_type.split('_')[4]
    
    
    
    
    
# if typee == '12branches':
#     pass
# elif typee == 'objects&colors':
#     pass
# elif typee == 'all':
#     pass
    
    
# def 12branches_evaluation    
    
    

    
def strategy_handling():
    pass

def tensor_max(tensor):

    idx = torch.argmax(tensor, dim=1, keepdim=True)
    y = torch.zeros(tensor.size(),device=device).scatter_(1, idx, 1.)
    return y

def tensor_thresh(tensor, thr=0.5):
    out = (tensor>thr).float()
    return out

def get_features(model,test_loader,device, need_attr=True, get_attr=True, get_feature=True):
        
    # taking output from loader 
    torch.cuda.empty_cache()
    model = model.to(device)
    features = []
    attributes = []
    model.eval()
    with torch.no_grad():
        
        start = time.time()
        for idx, data in enumerate(test_loader):
            # data = data.to(device) 'list' object has no attribute 'to'
            out_features = model.get_feature(data['img'].to(device), get_attr=get_attr, get_feature=get_feature)
            if need_attr:
                attributes.append(out_features['attr'])                
            else:
                attrs = torch.cat((out_features['gender'].unsqueeze(dim=1), out_features['head'],
                                      out_features['body'], out_features['body_type'].unsqueeze(dim=1),
                                      out_features['body_colour'], out_features['bags'],
                                      out_features['leg'], out_features['leg_colour'],
                                      out_features['foot'], out_features['foot_colour']), dim=1)  
                attributes.append(attrs)
            features.append(out_features['features'])
            
    finish = time.time()
    print('the time of getting feature is:', finish - start)
    features = torch.cat(features)
    attributes = torch.cat(attributes)
    return {'features':features, 'attributes':attributes}
    

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
            out_data = attr_net(data['img'])                      
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
                y_attr = tensor_thresh(torch.sigmoid(out_data))
                # y_attr = tensor_thresh(out_data['attr'], 0)
                y_target = data['attr'].to('cpu')  
                predicts.append(y_attr.to('cpu'))
                targets.append(y_target.to('cpu'))     
        predicts = torch.cat(predicts)
        targets = torch.cat(targets)
        test_attr_metrics = tensor_metrics(y_target.float(), y_attr)    
    return test_attr_metrics 


def map_evaluation(query, gallery, dist_matrix, n = 10):
    
    sorted, indices = torch.sort(dist_matrix)
    # calculating map:
    average_precision = []
    for i in range(len(dist_matrix)):
        m = 0 # the total positive until that array
        sum_precision = 0
        for j in range(n):
            if query['id'][i] == gallery['id'][indices[i,j]]:
                m += 1
                sum_precision += m/(j+1)
        if m != 0:
            average_precision.append(sum_precision/m)
        else:
            average_precision.append(0)
    mean_average_precision = sum(average_precision)/len(average_precision)
    return mean_average_precision


def get_feature_fromloader(attr_net,query_loader, gallery_loader,feature_mode=['concat','cnn','attr'],
                           need_attr=True, get_attr=True, get_feature=True):
    
    
    query_features = get_features(attr_net, query_loader, device=device, 
                                  need_attr=need_attr, get_attr=get_attr, get_feature=get_feature)
    gallery_features = get_features(attr_net, gallery_loader, device=device,
                                    need_attr=need_attr, get_attr=get_attr, get_feature=get_feature)
    
    dist_matrix = {}
    if 'concat' in feature_mode:
        # concat attributes output and features output and create new feature vectors.
        query_cat = torch.cat((query_features['features'], query_features['attributes']),dim=1)
        gallery_cat = torch.cat((gallery_features['features'], gallery_features['attributes']),dim=1)        
        dist_matrix0 = torch.cdist(query_cat, gallery_cat)
        
        dist_matrix.update({'concat':dist_matrix0})
    if 'cnn' in feature_mode:
        dist_matrix0 = torch.cdist(query_features['features'], gallery_features['features'], compute_mode='use_mm_for_euclid_dist_if_necessary')
        dist_matrix.update({'cnn':dist_matrix0})
        
    if 'attr' in feature_mode:        
        dist_matrix0 = torch.cdist(query_features['attributes'], gallery_features['attributes'])
        dist_matrix.update({'attr':dist_matrix0})
        
    return dist_matrix 

def cmc_map_fromdist(query, gallery, dist_matrix, feature_mode='concat', max_rank=20):

    mean_average_precision = map_evaluation(query, gallery, dist_matrix,  n=max_rank)    
    print('ca_map on version 6.2\n features:{} is:'.format(feature_mode) , mean_average_precision)
    
    query_np = query['id'].to('cpu').numpy()
    gallery_np = gallery['id'].to('cpu').numpy()
    dist_matrix = dist_matrix.to('cpu').numpy()
    
    rank = torchreid.metrics.rank.evaluate_rank(dist_matrix, query_np, gallery_np, query['cam_id'],
                                                gallery['cam_id'], max_rank=max_rank, use_metric_cuhk03=False, use_cython=False)
    print('os_map on version 6.2\n features:{}  is:'.format(feature_mode) , rank[1])
    return {'ca_map':mean_average_precision, 'os_rank':rank}


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


def change2list(torch_tensor):
    list_tensor = []
    for met in torch_tensor:
        list_tensor.append(met.item())
    return list_tensor

# # precision
# thr_half_sigmoid = tensor([1.0000, 0.0000, 0.0000, 0.9016, 0.8333, 0.5000, 0.9419, 0.0000, 0.8000,
#         0.5000, 0.8621, 0.8750, 0.0000, 1.0000, 0.6667, 0.5000, 0.8571, 0.7500,
#         0.8235, 0.7222, 0.6774, 0.7222, 0.8929, 0.9444, 0.9091, 0.8000, 1.0000,
#         0.0000, 1.0000, 0.0000, 0.7500, 0.7000, 0.0000, 0.8421, 0.8293, 0.7500,
#         1.0000, 0.7857, 1.0000, 0.0000, 0.0000, 0.0000, 0.5000, 0.0000, 0.0000,
#         0.6786])

# thr_zero = tensor([0.9474, 1.0000, 0.0000, 0.8364, 0.8000, 0.4667, 0.9444, 0.0000, 1.0000,
#         0.0000, 0.8333, 0.9000, 0.0000, 1.0000, 0.6000, 0.8182, 1.0000, 0.6667,
#         0.8125, 0.6154, 0.5152, 0.8400, 1.0000, 0.8600, 0.8333, 0.4000, 1.0000,
#         0.0000, 0.5000, 1.0000, 0.7143, 0.7273, 0.0000, 0.9250, 0.7763, 0.7500,
#         0.0000, 0.5714, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
#         0.7083])

# # recall
# tensor([0.8108, 0.0000, 0.0000, 0.9821, 0.5882, 0.3889, 0.9878, 0.0000, 0.5000,
#         0.1667, 0.8929, 0.8750, 0.0000, 0.5000, 0.5714, 0.3636, 0.5455, 0.4286,
#         0.9333, 0.5652, 0.5833, 0.3939, 0.9259, 0.9444, 0.9091, 0.4444, 1.0000,
#         0.0000, 0.4000, 0.0000, 0.5455, 0.6364, 0.0000, 0.8889, 0.9714, 0.3529,
#         0.2000, 0.4231, 0.2000, 0.0000, 0.0000, 0.0000, 0.3333, 0.0000, 0.0000,
#         0.7917])

# tensor([0.8571, 0.1667, 0.0000, 0.9200, 0.5333, 0.3333, 0.9884, 0.0000, 0.5000,
#         0.0000, 0.9259, 0.6923, 0.0000, 0.8889, 1.0000, 0.6923, 0.4286, 0.4000,
#         0.7647, 0.4000, 0.4595, 0.6000, 0.9167, 0.9149, 0.5556, 0.6667, 1.0000,
#         0.0000, 0.2500, 0.2500, 0.7500, 0.4211, 0.0000, 0.9250, 0.9077, 0.3913,
#         0.0000, 0.2500, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
#         0.6296])

#%%

# for i in range(9):
    
# #%%


# part_metrics_np = part_metrics.numpy()    
# attr_metrics_np = attr_metrics.numpy()  
# attr_colomns = ['cap','hairless','short hair','long hair',
#            'knot', 'Tshirt/shirt','coat',
#            'top','simple/patterned','pants',
#            'short','skirt','shoes','sandal',
#            'hidden','gender','backpack',
#            'hand bag','no bag','b_w','b_r',
#            'b_o','b_y','b_green','b_b',
#            'b_gray','b_p','b_black','l_w',
#            'l_r','l_o','l_y','l_green','l_b',
#            'l_gray','l_p','l_black','f_w',
#            'f_r','f_o','f_y','f_green','f_b',
#            'f_gray','f_p','f_black']
# part_colomns = ['head','body','clothes type','leg','foot','gender','bags', 'body color','leg color','foot color']
# indexes = ['precision', 'recall', 'accuracy', 'F1']
# attr_metrics_pd = pd.DataFrame(attr_metrics_np, columns=attr_colomns, index=indexes)
# part_metrics_pd = pd.DataFrame(part_metrics_np, columns=part_colomns, index=indexes)

# body_color_column = ['b_w','b_r',
#            'b_o','b_y','b_green','b_b',
#            'b_gray','b_p','b_black']
# body_color_indexes = ['real_positiv', 't_p', 'f_p','real_negative',  't_n', 'f_n']
# body_color_detailes_pd = pd.DataFrame(body_color_detailes, columns=body_color_column, index=body_color_indexes)
# #%%

# path_out_attr = '/home/hossein/deep-person-reid/dr_tale/result/V1_5/attr_metrixs_V1_5.xlsx'
# path_out_part = '/home/hossein/deep-person-reid/dr_tale/result/V1_5/part_metrixs_V1_5.xlsx'
# path_out_body_color_detail = '/home/hossein/deep-person-reid/dr_tale/result/V1_5/body_color_metrixs_V1_5.xlsx'

# attr_metrics_pd.to_excel(path_out_attr)
# part_metrics_pd.to_excel(path_out_part)
# body_color_detailes_pd.to_excel(path_out_body_color_detail)
