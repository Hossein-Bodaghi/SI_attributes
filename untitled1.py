#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 16:28:47 2022

@author: hossein
"""

for idx, data in enumerate(valid_loader):
    break
a = data['img'].to('cuda')
with torch.no_grad():
    out = attr_net(a[0:5])

import torch.nn as nn
import copy
import torch
class attributes_model(nn.Module):
    
    '''
    a model for training whole attributes 
    '''
    def __init__(self,
                 model,
                 feature_dim = 512,
                 attr_dim = 37,
                 branch_place = None):
        
        super().__init__()
        self.feature_dim = feature_dim
        self.model = model     
        self.attr_lin = nn.Linear(in_features=feature_dim , out_features=attr_dim)  
        
        if branch_place:
            self.branch_place = branch_place
            self.layer_list = ['conv1', 'maxpool', 'conv2', 'conv3',
                               'conv4', 'conv5', 'global_avgpool', 'fc']
            self.idx = self.layer_list.index(branch_place)    
            for i in range(self.idx+1, len(self.layer_list)):
                setattr(self, self.layer_list[i], copy.deepcopy(getattr(model, self.layer_list[i])))
        else: self.branch_place = 'fc'
                
    def out_layers_extractor(self, x, layer):
        out_os_layers = self.model.layer_extractor(x, layer) 
        return out_os_layers   
        
    def forward(self, x, get_features = False):

        if get_features:
            features = self.out_layers_extractor(x, 'fc') 
            return features        
        else:
            x = self.out_layers_extractor(x, self.branch_place) 
            if self.branch_place != 'fc':
                for i in range(self.idx+1, len(self.layer_list)): 
                    if self.layer_list[i] != 'fc':
                        x = getattr(self, self.layer_list[i])(x)
                    else:
                        x = x.view(x.size(0), -1)
                        x = self.fc(x)
            return {'attributes':self.attr_lin(x)}
    def save_baseline(self, saving_path):
        torch.save(self.model.state_dict(), saving_path)
        print('baseline model save to {}'.format(saving_path)) 
        
       
path_model = '/home/hossein/SI_attributes/results/CA_Duke_Market_vec_osnet_x1_0_msmt17/best_attr_net.pth'
trained_net = torch.load(path_model)
attr_net = attributes_model(model = model)
attr_net.load_state_dict(trained_net) 
