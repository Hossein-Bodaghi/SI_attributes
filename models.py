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

import torch.nn as nn
import torch

#%%
from torchreid.models.osnet import Conv1x1, OSBlock

blocks = [OSBlock, OSBlock, OSBlock]
layers = [2, 2, 2]
channels = [16, 64, 384, 512] # channels are the only difference between os_net_x_1 and others 


def _make_layer(
    block,
    layer,
    in_channels,
    out_channels,
    reduce_spatial_size,
    IN=False
):
    layers = []

    layers.append(block(in_channels, out_channels, IN=IN))
    for i in range(1, layer):
        layers.append(block(out_channels, out_channels, IN=IN))

    if reduce_spatial_size:
        layers.append(
            nn.Sequential(
                Conv1x1(out_channels, out_channels),
                nn.AvgPool2d(2, stride=2)
            )
        )

    return nn.Sequential(*layers)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
class mb_build_model(nn.Module):
    
    def __init__(self,
                 model,
                 num_id,
                 main_cov_size = 512,
                 attr_feat_dim = 128,
                 attr_dim = 128,
                 dropout_p = 0.3,
                 sep_conv_size = None,
                 sep_fc = False,
                 sep_clf = False):
        
        super().__init__()
        
        self.feature_dim = main_cov_size
        self.attr_feat_dim = attr_feat_dim
        self.dropout_p = dropout_p 
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.model = model
        self.sep_conv_size = sep_conv_size
        self.attr_dim = attr_dim
        self.sep_fc = sep_fc
        self.sep_clf = sep_clf
        # convs
        if self.sep_conv_size:
            # head
            self.conv_head = _make_layer(blocks[2],
                                        layers[2],
                                        self.feature_dim,
                                        self.sep_conv_size,
                                        reduce_spatial_size=False
                                        )
            # body
            self.conv_body = _make_layer(
                                            blocks[2],
                                            layers[2],
                                            self.feature_dim,
                                            self.sep_conv_size,
                                            reduce_spatial_size=False
                                        )            
            # leg
            self.conv_leg = _make_layer(
                                            blocks[2],
                                            layers[2],
                                            self.feature_dim,
                                            self.sep_conv_size,
                                            reduce_spatial_size=False
                                        )                
            # foot
            self.conv_foot = _make_layer(
                                            blocks[2],
                                            layers[2],
                                            self.feature_dim,
                                            self.sep_conv_size,
                                            reduce_spatial_size=False
                                        )            
            
            # gender & age
            self.conv_gender_age = _make_layer(
                                            blocks[2],
                                            layers[2],
                                            self.feature_dim,
                                            self.sep_conv_size,
                                            reduce_spatial_size=False
                                        )
            # bags
            self.conv_bags = _make_layer(
                                            blocks[2],
                                            layers[2],
                                            self.feature_dim,
                                            self.sep_conv_size,
                                            reduce_spatial_size=False
                                        )        
        # fully connecteds
        if self.sep_fc:
            # head
            self.head_fc = self._construct_fc_layer(self.attr_dim, self.attr_feat_dim, dropout_p=dropout_p)
            self.head_color_fc = self._construct_fc_layer(self.attr_dim, self.attr_feat_dim, dropout_p=dropout_p)           
            # upper body
            self.body_fc = self._construct_fc_layer(self.attr_dim, self.attr_feat_dim, dropout_p=dropout_p)
            self.body_type_fc = self._construct_fc_layer(self.attr_dim, self.attr_feat_dim, dropout_p=dropout_p)
            self.body_color_fc = self._construct_fc_layer(self.attr_dim, self.attr_feat_dim, dropout_p=dropout_p)            
            #lower body
            self.leg_fc = self._construct_fc_layer(self.attr_dim, self.attr_feat_dim, dropout_p=dropout_p)
            self.leg_color_fc = self._construct_fc_layer(self.attr_dim, self.attr_feat_dim, dropout_p=dropout_p)            
            #foot
            self.foot_fc = self._construct_fc_layer(self.attr_dim, self.attr_feat_dim, dropout_p=dropout_p)
            self.foot_color_fc = self._construct_fc_layer(self.attr_dim, self.attr_feat_dim, dropout_p=dropout_p)            
            #bags
            self.bag_fc = self._construct_fc_layer(self.attr_dim, self.attr_feat_dim, dropout_p=dropout_p)            
            # general
            self.age_fc = self._construct_fc_layer(self.attr_dim, self.attr_feat_dim, dropout_p=dropout_p)
            self.gender_fc = self._construct_fc_layer(self.attr_dim, self.attr_feat_dim, dropout_p=dropout_p)
        else:
            # head
            self.head_fc = self._construct_fc_layer(self.attr_dim, self.attr_feat_dim, dropout_p=dropout_p)                        
            # upper body
            self.body_fc = self._construct_fc_layer(self.attr_dim, self.attr_feat_dim, dropout_p=dropout_p)
            #lower body
            self.leg_fc = self._construct_fc_layer(self.attr_dim, self.attr_feat_dim, dropout_p=dropout_p)
            #foot
            self.foot_fc = self._construct_fc_layer(self.attr_dim, self.attr_feat_dim, dropout_p=dropout_p)
            #bags
            self.bag_fc = self._construct_fc_layer(self.attr_dim, self.attr_feat_dim, dropout_p=dropout_p) 
            # general
            self.age_fc = self._construct_fc_layer(self.attr_dim, self.attr_feat_dim, dropout_p=dropout_p)
            
        # classifiers
        # head
        self.head_clf = nn.Linear(self.attr_dim, 5)
        # head color
        self.head_color_clf = nn.Linear(self.attr_dim, 2)
        # body
        self.body_clf = nn.Linear(self.attr_dim, 4)
        # body type
        self.body_type_clf = nn.Linear(self.attr_dim, 1)
        # body_color
        self.body_color_clf = nn.Linear(self.attr_dim, 8)
        # leg
        self.leg_clf = nn.Linear(self.attr_dim, 3)
        # leg_color
        self.leg_color_clf = nn.Linear(self.attr_dim, 9)
        # foot
        self.foot_clf = nn.Linear(self.attr_dim, 3)
        # foot color
        self.foot_color_clf = nn.Linear(self.attr_dim, 4)
        # bag
        self.bag_clf = nn.Linear(self.attr_dim, 4)
        # gender
        self.gender_clf = nn.Linear(self.attr_dim, 1)        
        # age
        self.age_clf = nn.Linear(self.attr_feat_dim, 4)
        
     
    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None
    
        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]
    
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
    
        self.feature_dim = fc_dims[-1]
    
        return nn.Sequential(*layers)

    def get_feature(self, x, get_attr=True, get_feature=True, get_collection=False):
        
        out_conv4 = self.out_layers_extractor(x, 'out_conv4')
        # The path for multi-branches for attributes 
        out_head = self.attr_branch(out_conv4, self.conv_head, self.head_fc, self.head_clf, need_feature=True)          
        out_body = self.attr_branch(out_conv4, self.conv_body, self.body_fc, self.body_clf, need_feature=True)     
        out_body_type = self.attr_branch(out_conv4, self.conv_body_type, self.body_type_fc, self.body_type_clf, need_feature=True)          
        out_leg = self.attr_branch(out_conv4, self.conv_leg ,self.leg_fc, self.leg_clf, need_feature=True)           
        out_foot = self.attr_branch(out_conv4, self.conv_foot, self.foot_fc, self.foot_clf, need_feature=True)            
        out_gender = self.attr_branch(out_conv4, self.conv_gender, self.gender_fc, self.gender_clf, need_feature=True)             
        out_bags = self.attr_branch(out_conv4, self.conv_bags, self.bags_fc, self.bags_clf, need_feature=True)            
        out_body_colour = self.attr_branch(out_conv4, self.conv_body_color, self.body_color_fc, self.body_color_clf, need_feature=True)             
        out_leg_colour = self.attr_branch(out_conv4, self.conv_leg_color, self.leg_color_fc, self.leg_color_clf, need_feature=True)              
        out_foot_colour = self.attr_branch(out_conv4, self.conv_foot_color, self.foot_color_fc, self.foot_color_clf, need_feature=True)  
        
        # The path for person re-id:
        del out_conv4
        x = self.out_layers_extractor(x, 'out_fc')
        x = [out_head, out_body, out_body_type, out_leg,
                   out_foot, out_gender, out_bags, out_body_colour,
                   out_leg_colour, out_foot_colour, x]
        outputs = torch.cat(x, dim=1)
        return outputs
        
    
    def vector_features(self, x):
        features = self.model(x)
        out_attr = self.attr_lin(features) 
        out_features = torch.cat(features, out_attr, dim=1)
        return out_features
        
    def out_layers_extractor(self, x, layer):
        out_os_layers = self.model.layer_extractor(x, layer) 
        return out_os_layers   
    
    def fc2clf(self, x, fc_layer, clf_layer, sep_clf=False, sep_fc=False, need_feature=False):

        fc_out = []
        clf_put = []        
        if sep_fc:
            for i,fc in enumerate(fc_layer):
                feature = fc(x)
                fc_out.append(feature)
                clf_put.append(clf_layer[i](feature))
            if need_feature:
                return fc_out
            else:
                return clf_put                
        else:
            fc_out.append(fc_layer[0](x))
            if need_feature:
                return fc_out
            else:
                if sep_clf:
                    for clf in clf_layer:
                       clf_put.append(clf(fc_out[0]))
                    return clf_put
                else:
                    clf_put.append(clf_layer[0](fc_out[0]))
                    return clf_put
                
    def attr_branch(self, x, fc_layer, clf_layer,
                    conv_layer=None, sep_fc=False,
                    sep_clf=False, need_feature=False):
        ''' fc_layer should be a list of fully connecteds
            clf_layer hould be a list of classifiers
        '''
        # handling conv layer
        if conv_layer:
            x = conv_layer(x)
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.fc2clf(x=x, fc_layer=fc_layer,
                          clf_layer=clf_layer,
                          sep_clf=sep_clf, sep_fc=sep_fc,
                          need_feature=need_feature)
        return out 
    
    def forward(self, x):
        out_conv4 = self.out_layers_extractor(x, 'out_conv4')       
        out_attributes = {}

        if self.sep_fc:
            # head
            head_fcc = []
            head_fcc.append(self.head_fc)
            head_fcc.append(self.head_color_fc)
            # upper body
            upper_body_fcc = []
            upper_body_fcc.append(self.body_fc)
            upper_body_fcc.append(self.body_type_fc )
            upper_body_fcc.append(self.body_color_fc)   
            #lower_body
            lower_body_fcc = []
            lower_body_fcc.append(self.leg_fc)
            lower_body_fcc.append(self.leg_color_fc)             
            #foot
            foot_fcc = []
            foot_fcc.append(self.foot_fc)
            foot_fcc.append(self.foot_color_fc)   
            #bags
            bag_fcc = []
            bag_fcc.append(self.bag_fc)       
            # general
            general_fcc = []
            general_fcc.append(self.age_fc)
            general_fcc.append(self.gender_fc)
            
        out_head = self.attr_branch(out_conv4, self.conv_head, self.head_fc, self.head_clf)          
        out_body = self.attr_branch(out_conv4, self.conv_body, self.body_fc, self.body_clf)     
        out_body_type = self.attr_branch(out_conv4, self.conv_body_type, self.body_type_fc, self.body_type_clf)          
        out_leg = self.attr_branch(out_conv4, self.conv_leg ,self.leg_fc, self.leg_clf)           
        out_foot = self.attr_branch(out_conv4, self.conv_foot, self.foot_fc, self.foot_clf)            
        out_gender = self.attr_branch(out_conv4, self.conv_gender, self.gender_fc, self.gender_clf)             
        out_bags = self.attr_branch(out_conv4, self.conv_bags, self.bags_fc, self.bags_clf)            
        out_body_colour = self.attr_branch(out_conv4, self.conv_body_color, self.body_color_fc, self.body_color_clf)             
        out_leg_colour = self.attr_branch(out_conv4, self.conv_leg_color, self.leg_color_fc, self.leg_color_clf)              
        out_foot_colour = self.attr_branch(out_conv4, self.conv_foot_color, self.foot_color_fc, self.foot_color_clf)    
        out_attributes.update({'head':out_head,
                                 'body':out_body,
                                 'body_type':out_body_type,
                                 'leg':out_leg,
                                 'foot':out_foot,
                                 'gender':out_gender,
                                 'bags':out_bags,
                                 'body_colour':out_body_colour,
                                 'leg_colour':out_leg_colour,
                                 'foot_colour':out_foot_colour})
        return out_attributes
    
    def save_baseline(self, saving_path):
        torch.save(self.model.state_dict(), saving_path)
        print('baseline model save to {}'.format(saving_path))

#%%
                                        
class CD_builder(nn.Module):
    
    def __init__(self,
                 model,
                 num_id,
                 feature_dim = channels[3],
                 attr_feat_dim = channels[1],
                 attr_dim = 46,
                 dropout_p = 0.3):
        
        super().__init__()
        
        self.feature_dim = feature_dim
        self.attr_feat_dim = attr_feat_dim
        self.dropout_p = dropout_p 
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.model = model       

        self.fc = self._construct_fc_layer(self.attr_feat_dim, channels[-1], dropout_p=dropout_p)
        
        self.attr_clf = nn.Linear(self.attr_feat_dim, attr_dim)         

        
    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None
    
        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]
    
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
    
        self.feature_dim = fc_dims[-1]
    
        return nn.Sequential(*layers)

    def get_feature(self, x, get_attr=True, get_feature=True, get_collection=False):
        
        out_conv4 = self.out_layers_extractor(x, 'out_conv4')
        # The path for multi-branches for attributes 
        out_head = self.attr_branch(out_conv4, self.conv_head, self.head_fc, self.head_clf, need_feature=True)          
        out_body = self.attr_branch(out_conv4, self.conv_body, self.body_fc, self.body_clf, need_feature=True)     
        out_body_type = self.attr_branch(out_conv4, self.conv_body_type, self.body_type_fc, self.body_type_clf, need_feature=True)          
        out_leg = self.attr_branch(out_conv4, self.conv_leg ,self.leg_fc, self.leg_clf, need_feature=True)           
        out_foot = self.attr_branch(out_conv4, self.conv_foot, self.foot_fc, self.foot_clf, need_feature=True)            
        out_gender = self.attr_branch(out_conv4, self.conv_gender, self.gender_fc, self.gender_clf, need_feature=True)             
        out_bags = self.attr_branch(out_conv4, self.conv_bags, self.bags_fc, self.bags_clf, need_feature=True)            
        out_body_colour = self.attr_branch(out_conv4, self.conv_body_color, self.body_color_fc, self.body_color_clf, need_feature=True)             
        out_leg_colour = self.attr_branch(out_conv4, self.conv_leg_color, self.leg_color_fc, self.leg_color_clf, need_feature=True)              
        out_foot_colour = self.attr_branch(out_conv4, self.conv_foot_color, self.foot_color_fc, self.foot_color_clf, need_feature=True)  
        
        # The path for person re-id:
        del out_conv4
        x = self.out_layers_extractor(x, 'out_fc')
        x = [out_head, out_body, out_body_type, out_leg,
                   out_foot, out_gender, out_bags, out_body_colour,
                   out_leg_colour, out_foot_colour, x]
        outputs = torch.cat(x, dim=1)
        return outputs
        
    
    def vector_features(self, x):
        features = self.model(x)
        out_attr = self.attr_lin(features) 
        out_features = torch.cat(features, out_attr, dim=1)
        return out_features
        
    def out_layers_extractor(self, x, layer):
        out_os_layers = self.model.layer_extractor(x, layer) 
        return out_os_layers   
    
    def attr_branch(self, x, conv_layer, fc_layer, clf_layer, need_feature=False):
        x = conv_layer(x)
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        x = fc_layer(x)
        if need_feature:
            return x
        else:
            x = clf_layer(x)
        return x
       
    def forward(self, x):
        features = self.out_layers_extractor(x, 'out_globalavg')
        features = features.view(features.size(0), -1) 
        features = self.fc(features)
        out_attr = self.attr_clf(features)       

        return {'attr':out_attr}
    
    def save_baseline(self, saving_path):
        torch.save(self.model.state_dict(), saving_path)
        print('baseline model save to {}'.format(saving_path))   
            
        
    
    