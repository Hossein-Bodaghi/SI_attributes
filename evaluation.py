#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 18:34:12 2021

@author: hossein
"""

#%%
"""
version v1 is:
    1) we consider whole vector of output as our target oposite 
    of v1 that we seperatly had a loss function for every collection
"""
from delivery import data_delivery 
from trainings import CA_target_attributes_12
from models import mb_build_model
from loaders import CA_Loader
from metrics import tensor_metrics
import time
import torch
from torch.utils.data import DataLoader

# import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print('calculation is on:',device)
torch.cuda.empty_cache()

#%%

def attr_evaluation(attr_net, test_loader, device):
    
    attr_net.to(device)
    # evaluation:     
    attr_net.eval()
    with torch.no_grad():
        targets = []
        predicts = []
        for idx, data in enumerate(test_loader):
            for key, _ in data.items():
                data[key] = data[key].to(device)
            # forward step
            out_data = attr_net(data['img'], need_feature=False)                      
            y_attr, y_target = CA_target_attributes_12(out_data, data, softmax=True, tensor_max=True)
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

attr_net_camarket = mb_build_model(model = model,
                 main_cov_size = 512,
                 attr_dim = 64,
                 dropout_p = 0.3,
                 sep_conv_size = 128,
                 sep_fc = False,
                 sep_clf = True)

model_path = './results/sif_convt_128_flf_64_clft_bce_CA/best_attr_net.pth'
trained_net = torch.load(model_path)
attr_net_camarket.load_state_dict(trained_net)

#%%
main_path = './datasets/Market1501/Market-1501-v15.09.15/gt_bbox/'
path_attr = './attributes/new_total_attr.npy'
test_idx_path = './attributes/test_idx_full.pth'
test_idx = torch.load(test_idx_path)

attr = data_delivery(main_path=main_path,
                     path_attr=path_attr,
                     need_collection=True,
                     double=False,
                     need_attr=False,
                     mode = 'Market_attribute')

test_data = CA_Loader(img_path=main_path,
                          attr=attr,
                          resolution=(256, 128),
                          indexes=test_idx,
                          need_attr = False,
                          need_collection=True,
                          need_id = False,
                          two_transforms = False,                          
                          ) 
batch_size = 200
test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=False)

#%%
start = time.time()
attr_metrics = attr_evaluation(attr_net_camarket, test_loader, device)
finish = time.time()

print('inferencing from {} images takes {:.4f} s'.format(len(test_data), finish-start))
#%%

attr_colomns = ['gender','cap','hairless','short hair','long hair',
           'knot', 'h_colorful', 'h_black','Tshirt_shs', 'shirt_ls','coat',
           'top','simple/patterned','b_w','b_r',
           'b_y','b_green','b_b',
           'b_gray','b_p','b_black','backpack', 'shoulder bag',
           'hand bag','no bag','pants',
           'short','skirt','l_w','l_r','l_br','l_y','l_green','l_b',
           'l_gray','l_p','l_black','shoes','sandal',
           'hidden','no color','f_w', 'f_colorful','f_black', 'young', 
           'teenager', 'adult', 'old']
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

