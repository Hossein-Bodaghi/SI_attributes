#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:03:10 2021

@author: hossein

here we can find different types of models 
that are define for person-attribute detection. 
this is Hossein Bodaghies thesis 
"""

'''
*

when load a pretrained model from torchreid it just brings imagenet trained models
so if we want to bring pretrained on other datasets we should use this function

'''

from collections import OrderedDict
import torch.nn as nn
from torch import load
import torch


def feature_model(model, last=None):
    if last:
        new_model1 = nn.Sequential(*list(model.children())[:-last])
    else:
        new_model1 = nn.Sequential(*list(model.children())[:-2])
    return new_model1

def my_load_pretrain(model1 , pretrain_path):
    
    state_dict = load(pretrain_path)
    model_dict = model1.state_dict()
    new_state_dict = OrderedDict()
    
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items(): # state dict is our loaded weights
            if k.startswith('module.'):
                k = k[7:] # discard module.
            if k in model_dict and model_dict[k].size() == v.size():
                new_state_dict[k] = v
                matched_layers.append(k)
            else:
                discarded_layers.append(k)
                
    model_dict.update(new_state_dict)
    model1.load_state_dict(model_dict)   
    
    if len(matched_layers) == 0:
        print(
            'The pretrained weights from "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'
        ) 
    return model1


#%%

class MyOsNet(nn.Module):
    
    '''
    this is our network in this version it just take output from features of
    original omni-scale network.
    
    if attr_inc=True then for each attribute has a seperate linear 
    layer for classification
    
    if id_inc=True the output of attribute detection and models features will be concatanated
    and then a clasiification will predict the id of input picture
    '''
    
    def __init__(self,
                 model,
                 num_id,
                 feature_dim=512,
                 attr_dim=55,
                 id_inc=True,
                 attr_inc=True):
        
        super().__init__()
        self.feature_dim = feature_dim
        self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.batchnormalization = nn.BatchNorm1d(num_features=feature_dim)
        self.dropout = nn.Dropout(0.3)


        self.model = model
        self.linear = nn.Linear(in_features=feature_dim , out_features=feature_dim)
        self.id_lin = nn.Linear(in_features=feature_dim+attr_dim , out_features=num_id)
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
        
        self.id_inc = id_inc
        self.attr_inc = attr_inc
        
    def forward(self, x):
        
        features = self.model(x)
        features = features.view(-1,self.feature_dim)
        features = self.dropout(features)
        features = self.linear(features)
        features = self.batchnormalization(features)
        features = self.leakyrelu(features)
        features = self.dropout(features)
        
        if self.attr_inc:
            
            out_attr = self.softmax(self.attr_lin(features))
            out_attr = self.dropout(out_attr)
            if self.id_inc:
                concated = torch.cat((features,out_attr),dim=1) # dim 1 because all the tensors dimentions start with (batch,..)            
                out_id = self.softmax(self.id_lin(concated))
                return (out_id,out_attr)
            else: 
                return out_attr
        
        else:
            out_head = self.softmax(self.head_lin(features))
            out_body = self.softmax(self.body_lin(features))
            out_body_type = self.sigmoid(self.body_type_lin(features))
            out_leg = self.softmax(self.leg_lin(features))
            out_foot = self.softmax(self.foot_lin(features))
            out_gender = self.sigmoid(self.gender_lin(features))
            out_bags = self.softmax(self.bags_lin(features))
            out_body_colour = self.softmax(self.body_colour_lin(features))
            out_leg_colour = self.softmax(self.leg_colour_lin(features))
            out_foot_colour = self.softmax(self.foot_colour_lin(features))
            if self.id_inc:
                concated = torch.cat((features,
                                      out_head,
                                      out_body,
                                      out_body_type,
                                      out_leg,
                                      out_foot,
                                      out_gender,
                                      out_bags,
                                      out_bags,
                                      out_body_colour,
                                      out_leg_colour,
                                      out_foot_colour),dim=1) # the first parameter of torch.cat() should be checked that takes list or tuple or what
                out_id = self.softmax(self.id_lin(concated))
                return (out_head,
                         out_body,
                         out_body_type,
                         out_leg,
                         out_foot,
                         out_gender,
                         out_bags,
                         out_body_colour,
                         out_leg_colour,
                         out_foot_colour,
                         out_id)
            else:
                # id will be added 
                return (out_head,
                        out_body,
                        out_body_type,
                        out_leg,
                        out_foot,
                        out_gender,
                        out_bags,
                        out_body_colour,
                        out_leg_colour,
                        out_foot_colour)
        
    def predict(self, x):
        features = self.model(x)
        features = features.view(-1,self.feature_dim)
        features = self.dropout(features)
        features = self.linear(features)
        features = self.batchnormalization(features)
        features = self.leakyrelu(features)
        features = self.dropout(features)
        
        if self.attr_inc:
            
            out_attr = self.softmax(self.attr_lin(features))
#            out_attr = self.dropout(out_attr)
            if self.id_inc:
                concated = torch.cat((features,out_attr),dim=1) # dim 1 because all the tensors dimentions start with (batch,..)            
                out_id = self.softmax(self.id_lin(concated))
                return (out_id,out_attr)
            else: 
                return out_attr
        
        else:
            out_head = self.softmax(self.head_lin(features))
            out_body = self.softmax(self.body_lin(features))
            out_body_type = self.sigmoid(self.body_type_lin(features))
            out_leg = self.softmax(self.leg_lin(features))
            out_foot = self.softmax(self.foot_lin(features))
            out_gender = self.sigmoid(self.gender_lin(features))
            out_bags = self.softmax(self.bags_lin(features))
            out_body_colour = self.softmax(self.body_colour_lin(features))
            out_leg_colour = self.softmax(self.leg_colour_lin(features))
            out_foot_colour = self.softmax(self.foot_colour_lin(features))
            if self.id_inc:
                concated = torch.cat((features,
                                      out_head,
                                      out_body,
                                      out_body_type,
                                      out_leg,
                                      out_foot,
                                      out_gender,
                                      out_bags,
                                      out_bags,
                                      out_body_colour,
                                      out_leg_colour,
                                      out_foot_colour),dim=1) # the first parameter of torch.cat() should be checked that takes list or tuple or what
                out_id = self.softmax(self.id_lin(concated))
                return (out_head,
                         out_body,
                         out_body_type,
                         out_leg,
                         out_foot,
                         out_gender,
                         out_bags,
                         out_body_colour,
                         out_leg_colour,
                         out_foot_colour,
                         out_id)
            else:
                # id will be added 
                return (out_head,
                        out_body,
                        out_body_type,
                        out_leg,
                        out_foot,
                        out_gender,
                        out_bags,
                        out_body_colour,
                        out_leg_colour,
                        out_foot_colour)

#%%

class MyOsNet2(nn.Module):
    
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
                 feature_dim=512,
                 attr_dim=46,
                 id_inc=True,
                 attr_inc=True):
        
        super().__init__()
        self.feature_dim = feature_dim
        # self.sigmoid = torch.sigmoid()
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.batchnormalization = nn.BatchNorm1d(num_features=feature_dim)
        self.dropout = nn.Dropout(0.3)


        self.model = model
        self.linear = nn.Linear(in_features=feature_dim , out_features=feature_dim,)
        self.id_lin = nn.Linear(in_features=feature_dim+attr_dim , out_features=num_id)
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
        
        self.id_inc = id_inc
        self.attr_inc = attr_inc
        
    def get_feature(self, x):
        features = self.model(x)
        features = features.view(-1,self.feature_dim)
        features = self.linear(features)
        return features
        
        
    def forward(self, x):
        
        features = self.model(x)
        features = features.view(-1,self.feature_dim)
        features = self.dropout(features)
        features = self.linear(features)
        features = self.batchnormalization(features)
        features = self.leakyrelu(features)
        features = self.dropout(features)
        
        if self.attr_inc:
            
            out_attr = self.attr_lin(features)

            if self.id_inc:
                concated = torch.cat((features,out_attr),dim=1) # dim 1 because all the tensors dimentions start with (batch,..)            
                out_id = self.id_lin(concated)
                return (out_id,out_attr)
            else: 
                return out_attr
        
        else:
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
            
            if self.id_inc:
                out_head1 = self.softmax(out_head)
                out_body1 = self.softmax(out_body)
                out_body_type1 = torch.sigmoid(out_body_type)
                out_leg1 = self.softmax(out_leg)
                out_foot1 = self.softmax(out_foot)
                out_gender1 = torch.sigmoid(out_gender)
                out_bags1 = self.softmax(out_bags)
                out_body_colour1 = self.softmax(out_body_colour)
                out_leg_colour1 = self.softmax(out_leg_colour)
                out_foot_colour1 = self.softmax(out_foot_colour)  
                
                concated = torch.cat((features,
                                      out_head1,
                                      out_body1,
                                      out_body_type1,
                                      out_leg1,
                                      out_foot1,
                                      out_gender1,
                                      out_bags1,
                                      out_bags1,
                                      out_body_colour1,
                                      out_leg_colour1,
                                      out_foot_colour1),dim=1) # the first parameter of torch.cat() should be checked that takes list or tuple or what
                # print('the size of out_body_colour layer',out_body_colour.size())
                # print('the size of features layer',features.size())
                # print('the size of concatanated layer',concated.size())
                out_id = self.id_lin(concated)
                return (out_head,
                         out_body,
                         out_body_type,
                         out_leg,
                         out_foot,
                         out_gender,
                         out_bags,
                         out_body_colour,
                         out_leg_colour,
                         out_foot_colour,
                         out_id)
            else:

                return (out_head,
                        out_body,
                        out_body_type,
                        out_leg,
                        out_foot,
                        out_gender,
                        out_bags,
                        out_body_colour,
                        out_leg_colour,
                        out_foot_colour)
        
    def predict(self, x):
        features = self.model(x)
        features = features.view(-1,self.feature_dim)
        features = self.dropout(features)
        features = self.linear(features)
        features = self.batchnormalization(features)
        features = self.leakyrelu(features)
        features = self.dropout(features)
        
        if self.attr_inc:
            
            # we didnt put any activation becuase regression doesnt need any activation (mse loss can be ok)
            out_attr = self.attr_lin(features)
#            out_attr = self.dropout(out_attr)
            if self.id_inc:
                concated = torch.cat((features,out_attr),dim=1) # dim 1 because all the tensors dimentions start with (batch,..)            
                out_id = self.softmax(self.id_lin(concated))
                return (out_id,out_attr)
            else: 
                return out_attr
        
        else:
            out_head = self.softmax(self.head_lin(features))
            out_body = self.softmax(self.body_lin(features))
            out_body_type = torch.sigmoid(self.body_type_lin(features))
            out_leg = self.softmax(self.leg_lin(features))
            out_foot = self.softmax(self.foot_lin(features))
            out_gender = torch.sigmoid(self.gender_lin(features))
            out_bags = self.softmax(self.bags_lin(features))
            out_body_colour = self.softmax(self.body_colour_lin(features))
            out_leg_colour = self.softmax(self.leg_colour_lin(features))
            out_foot_colour = self.softmax(self.foot_colour_lin(features))
            
            if self.id_inc:
                concated = torch.cat((features,
                                      out_head,
                                      out_body,
                                      out_body_type,
                                      out_leg,
                                      out_foot,
                                      out_gender,
                                      out_bags,
                                      out_bags,
                                      out_body_colour,
                                      out_leg_colour,
                                      out_foot_colour),dim=1) # the first parameter of torch.cat() should be checked that takes list or tuple or what
                
                concated = self.dropout(concated)
                out_id = self.softmax(self.id_lin(concated))
                return (out_head,
                         out_body,
                         out_body_type,
                         out_leg,
                         out_foot,
                         out_gender,
                         out_bags,
                         out_body_colour,
                         out_leg_colour,
                         out_foot_colour,
                         out_id)
            else:
                # id will be added 
                return (out_head,
                        out_body,
                        out_body_type,
                        out_leg,
                        out_foot,
                        out_gender,
                        out_bags,
                        out_body_colour,
                        out_leg_colour,
                        out_foot_colour)
#%%
class CA_market_model(nn.Module):
    
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
        self.id_lin = nn.Linear(in_features=feature_dim+attr_dim , out_features=num_id)
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
        
    def get_feature(self, x):
        features = self.model(x)
        features = self.batchnormalization(features)
        features = self.dropout(features)        
        # features = self.relu(features) 
        
        return features
        
        
    def forward(self, x):
        
        features = self.model(x) 
        features = features.view(-1,self.feature_dim)              
        # features = self.relu(features)
        features = self.batchnormalization(features)
        features = self.dropout(features) 
                    
        
        if self.need_attr and not self.need_collection:
            
            out_attr = self.attr_lin(features)

            if self.need_id:
                concated = torch.cat((features,self.sigmoid(out_attr)),dim=1) # dim 1 because all the tensors dimentions start with (batch,..)            
                concated = self.batchnormalization2(concated)
                concated = self.dropout(concated)
                out_id = self.id_lin(concated)
                return {'id':out_id, 'attr':out_attr}
            else: 
                return {'id':out_id}
        
        elif self.need_collection and not self.need_attr:
            # out_head = self.head_lin(features)
            # out_head = self.batchnormalization(out_head)            
            # out_body = self.body_lin(features)
            # out_body = self.batchnormalization(out_body)            
            # out_body_type = self.body_type_lin(features)
            # out_body_type = self.batchnormalization(out_body_type)            
            # out_leg = self.leg_lin(features)
            # out_leg = self.batchnormalization(out_leg)            
            # out_foot = self.foot_lin(features)
            # out_foot = self.batchnormalization(out_foot)            
            # out_gender = self.body_type_lin(features)
            # out_gender = self.batchnormalization(out_gender)            
            # out_bags = self.bags_lin(features)
            # out_bags = self.batchnormalization(out_bags)            
            # out_body_colour = self.body_colour_lin(features)
            # out_body_colour = self.batchnormalization(out_body_colour)            
            # out_leg_colour = self.leg_colour_lin(features)
            # out_leg_colour = self.batchnormalization(out_leg_colour)            
            # out_foot_colour = self.foot_colour_lin(features)
            # out_foot_colour = self.batchnormalization(out_foot_colour)
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
                out_head1 = self.softmax(out_head)
                out_body1 = self.softmax(out_body)
                out_body_type1 = self.sigmoid(out_body_type)
                out_leg1 = self.softmax(out_leg)
                out_foot1 = self.softmax(out_foot)
                out_gender1 = self.sigmoid(out_gender)
                out_bags1 = self.softmax(out_bags)
                out_body_colour1 = self.sigmoid(out_body_colour)
                out_leg_colour1 = self.softmax(out_leg_colour)
                out_foot_colour1 = self.softmax(out_foot_colour)  
                
                concated = torch.cat((features,
                                      out_head1,
                                      out_body1,
                                      out_body_type1,
                                      out_leg1,
                                      out_foot1,
                                      out_gender1,
                                      out_bags1,
                                      out_body_colour1,
                                      out_leg_colour1,
                                      out_foot_colour1),dim=1) # the first parameter of torch.cat() should be checked that takes list or tuple or what
               
                # concated = torch.cat((features,
                #                       out_head,
                #                       out_body,
                #                       out_body_type,
                #                       out_leg,
                #                       out_foot,
                #                       out_gender,
                #                       out_bags,
                #                       out_body_colour,
                #                       out_leg_colour,
                #                       out_foot_colour),dim=1)
                concated = self.dropout(concated)
                concated = self.batchnormalization2(concated)                                
                # print('the size of out_body_colour layer',out_body_colour.size())
                # print('the size of features layer',features.size())
                # print('the size of concatanated layer',concated.size())
                out_id = self.id_lin(concated)
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
#%%
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
        
    def out_layers_extractor(self, x, layer):
        out_os_layers = self.model.layer_extractor(x, layer) 
        return out_os_layers   
        
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
#%%
class CA_market_model3(nn.Module):
    
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
        # self.linear = nn.Linear(in_features=feature_dim , out_features=feature_dim)
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
        # features = self.batchnormalization(features)
        features = self.dropout(features)

        if self.need_attr and not self.need_collection:            
            out_attr = self.attr_lin(features)

            if self.need_id:
                out_id = self.id_lin(features)
                return {'id':out_id, 'attr':out_attr}
            else: 
                return {'attr':out_attr}
        
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
            
    def save_baseline(self, save_path):
        torch.save(self.model.state_dict(), save_path)
#%%
class CA_market_model4(nn.Module):
    
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
        
        # for param in self.model.parameters():
        #     param.requires_grad = False        
            
    def get_feature(self, x, get_attr=True, get_feature=True, get_collection=False):
        
        features = self.model(x)
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
    
    def vector_features(self, x):
        features = self.model(x)
        out_attr = self.attr_lin(features) 
        out_features = torch.cat(features, out_attr, dim=1)
        return out_features
        
    def out_layers_extractor(self, x, layer):
        out_os_layers = self.model.layer_extractor(x, layer) 
        return out_os_layers   
       
    def forward(self, x):
        
        features = self.model(x) 
        features = self.dropout(features) 
                    
        if self.need_attr and not self.need_collection:            
            out_attr = self.attr_lin(features)

            if self.need_id:
                out_id = self.id_lin(features)
                return {'id':out_id, 'attr':out_attr}
            else: 
                return {'attr':out_attr}
        
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
            
    def save_baseline(self, saving_path):
        torch.save(self.model.state_dict(), saving_path)
        print('baseline model save to {}'.format(saving_path))