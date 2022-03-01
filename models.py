#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:03:10 2021

@author: hossein

here we can find different types of models 
that are define for person-attribute detection. 
this is Hossein Bodaghies thesis 
"""

import torch.nn as nn
import torch
from transformers import MBConvBlock, ScaledDotProductAttention
import copy
#%%
from torchreid.models.osnet import Conv1x1, OSBlock
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

blocks = [OSBlock, OSBlock, OSBlock]
layers = [2, 2, 2]
channels = [16, 64, 96, 128] # channels are the only difference between os_net_x_1 and others 

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

#%%

class mb_transformer_build_model(nn.Module):
    
    def __init__(self,
                 model,
                 device,
                 main_cov_size = 128,
                 attr_dim = 128,
                 dropout_p = 0.3):
        
        super().__init__()        
        self.feature_dim = main_cov_size
        self.dropout_p = dropout_p 
        self.model = model
        self.attr_dim = attr_dim
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool1d = nn.MaxPool1d(kernel_size=2, stride=2)
        self.device = device
        
        # convs + transforms:
        # head
        self.conv_head = nn.Sequential(MBConvBlock(ksize=3,input_filters=512, output_filters=128, image_size=(16,8)),
                                            nn.Conv2d(128, 128, kernel_size=1),
                                            nn.ReLU(),
                                            nn.Conv2d(128, 128, kernel_size=1)
                                        )
        self.trans_head1 = ScaledDotProductAttention(128, 16, 16, 8)
        self.trans_head2 = ScaledDotProductAttention(128, 16, 16, 8)
        self.mlp_head1 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        self.mlp_head2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32))
        # head_color
        self.conv_head_color = nn.Sequential(MBConvBlock(ksize=3,input_filters=512, output_filters=128, image_size=(16,8)),
                                            nn.Conv2d(128, 128, kernel_size=1),
                                            nn.ReLU(),
                                            nn.Conv2d(128, 128, kernel_size=1)
                                        )
        self.trans_head_color1 = ScaledDotProductAttention(128, 16, 16, 8)
        self.trans_head_color2 = ScaledDotProductAttention(128, 16, 16, 8)
        self.mlp_head_color1 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        self.mlp_head_color2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32))
        # body
        self.conv_body = nn.Sequential(MBConvBlock(ksize=3,input_filters=512, output_filters=128, image_size=(16,8)),
                                            nn.Conv2d(128, 128, kernel_size=1),
                                            nn.ReLU(),
                                            nn.Conv2d(128, 128, kernel_size=1)
                                        )           
        self.trans_body1 = ScaledDotProductAttention(128, 16, 16, 8)
        self.trans_body2 = ScaledDotProductAttention(128, 16, 16, 8)
        self.mlp_body1 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        self.mlp_body2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32))
        # body_color
        self.conv_body_color = nn.Sequential(MBConvBlock(ksize=3,input_filters=512, output_filters=128, image_size=(16,8)),
                                            nn.Conv2d(128, 128, kernel_size=1),
                                            nn.ReLU(),
                                            nn.Conv2d(128, 128, kernel_size=1)
                                        ) 
        self.trans_body_color1 = ScaledDotProductAttention(128, 16, 16, 8)
        self.trans_body_color2 = ScaledDotProductAttention(128, 16, 16, 8)
        self.mlp_body_color1 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        self.mlp_body_color2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32))
        # body_type
        self.conv_body_type = nn.Sequential(MBConvBlock(ksize=3,input_filters=512, output_filters=128, image_size=(16,8)),
                                            nn.Conv2d(128, 128, kernel_size=1),
                                            nn.ReLU(),
                                            nn.Conv2d(128, 128, kernel_size=1)
                                        ) 
        self.trans_body_type1 = ScaledDotProductAttention(128, 16, 16, 8)
        self.trans_body_type2 = ScaledDotProductAttention(128, 16, 16, 8)
        self.mlp_body_type1 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        self.mlp_body_type2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32))
        # leg
        self.conv_leg = nn.Sequential(MBConvBlock(ksize=3,input_filters=512, output_filters=128, image_size=(16,8)),
                                            nn.Conv2d(128, 128, kernel_size=1),
                                            nn.ReLU(),
                                            nn.Conv2d(128, 128, kernel_size=1)
                                        )               
        self.trans_leg1 = ScaledDotProductAttention(128, 16, 16, 8)
        self.trans_leg2 = ScaledDotProductAttention(128, 16, 16, 8)
        self.mlp_leg1 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        self.mlp_leg2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32))
        # leg_color
        self.conv_leg_color = nn.Sequential(MBConvBlock(ksize=3,input_filters=512, output_filters=128, image_size=(16,8)),
                                            nn.Conv2d(128, 128, kernel_size=1),
                                            nn.ReLU(),
                                            nn.Conv2d(128, 128, kernel_size=1)
                                        ) 
        self.trans_leg_color1 = ScaledDotProductAttention(128, 16, 16, 8)
        self.trans_leg_color2 = ScaledDotProductAttention(128, 16, 16, 8)
        self.mlp_leg_color1 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        self.mlp_leg_color2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32))
        # foot
        self.conv_foot = nn.Sequential(MBConvBlock(ksize=3,input_filters=512, output_filters=128, image_size=(16,8)),
                                            nn.Conv2d(128, 128, kernel_size=1),
                                            nn.ReLU(),
                                            nn.Conv2d(128, 128, kernel_size=1)
                                        )            
        self.trans_foot1 = ScaledDotProductAttention(128, 16, 16, 8)
        self.trans_foot2 = ScaledDotProductAttention(128, 16, 16, 8)
        self.mlp_foot1 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        self.mlp_foot2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32))
        # foot_color
        self.conv_foot_color = nn.Sequential(MBConvBlock(ksize=3,input_filters=512, output_filters=128, image_size=(16,8)),
                                            nn.Conv2d(128, 128, kernel_size=1),
                                            nn.ReLU(),
                                            nn.Conv2d(128, 128, kernel_size=1)
                                        )          
        self.trans_foot_color1 = ScaledDotProductAttention(128, 16, 16, 8)
        self.trans_foot_color2 = ScaledDotProductAttention(128, 16, 16, 8)
        self.mlp_foot_color1 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        self.mlp_foot_color2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32))
        # gender
        self.conv_gender = nn.Sequential(MBConvBlock(ksize=3,input_filters=512, output_filters=128, image_size=(16,8)),
                                            nn.Conv2d(128, 128, kernel_size=1),
                                            nn.ReLU(),
                                            nn.Conv2d(128, 128, kernel_size=1)
                                        )
        self.trans_gender1 = ScaledDotProductAttention(128, 16, 16, 8)
        self.trans_gender2 = ScaledDotProductAttention(128, 16, 16, 8)
        self.mlp_gender1 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        self.mlp_gender2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32))
        # age
        self.conv_age = nn.Sequential(MBConvBlock(ksize=3,input_filters=512, output_filters=128, image_size=(16,8)),
                                            nn.Conv2d(128, 128, kernel_size=1),
                                            nn.ReLU(),
                                            nn.Conv2d(128, 128, kernel_size=1)
                                        )
        self.trans_age1 = ScaledDotProductAttention(128, 16, 16, 8)
        self.trans_age2 = ScaledDotProductAttention(128, 16, 16, 8)
        self.mlp_age1 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        self.mlp_age2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32))
        # bags
        self.conv_bags = nn.Sequential(MBConvBlock(ksize=3,input_filters=512, output_filters=128, image_size=(16,8)),
                                            nn.Conv2d(128, 128, kernel_size=1),
                                            nn.ReLU(),
                                            nn.Conv2d(128, 128, kernel_size=1)
                                        )             
            
        self.trans_bags1 = ScaledDotProductAttention(128, 16, 16, 8)
        self.trans_bags2 = ScaledDotProductAttention(128, 16, 16, 8)
        self.mlp_bags1 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        self.mlp_bags2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32))
        # fully connecteds
        # head
        self.head_fc = [self._construct_fc_layer(self.attr_dim, 1024, dropout_p=dropout_p).to(self.device)]
        self.head_color_fc = [self._construct_fc_layer(self.attr_dim, 1024, dropout_p=dropout_p).to(self.device)]           
        # upper body
        self.body_fc = [self._construct_fc_layer(self.attr_dim, 1024, dropout_p=dropout_p).to(self.device)]
        self.body_type_fc = [self._construct_fc_layer(self.attr_dim, 1024, dropout_p=dropout_p).to(self.device)]
        self.body_color_fc = [self._construct_fc_layer(self.attr_dim, 1024, dropout_p=dropout_p).to(self.device)]            
        #lower body
        self.leg_fc = [self._construct_fc_layer(self.attr_dim, 1024, dropout_p=dropout_p).to(self.device)]
        self.leg_color_fc = [self._construct_fc_layer(self.attr_dim, 1024, dropout_p=dropout_p).to(self.device)]           
        #foot
        self.foot_fc = [self._construct_fc_layer(self.attr_dim, 1024, dropout_p=dropout_p).to(self.device)]
        self.foot_color_fc = [self._construct_fc_layer(self.attr_dim, 1024, dropout_p=dropout_p).to(self.device)]            
        #bags
        self.bags_fc = [self._construct_fc_layer(self.attr_dim, 1024, dropout_p=dropout_p).to(self.device)]             
        # general
        self.age_fc = [self._construct_fc_layer(self.attr_dim, 1024, dropout_p=dropout_p).to(self.device)]
        self.gender_fc = [self._construct_fc_layer(self.attr_dim, 1024, dropout_p=dropout_p).to(self.device)]
            
        # classifiers
        # head
        self.head_clf = [nn.Linear(self.attr_dim, 5).to(self.device)]
        self.head_color_clf = [nn.Linear(self.attr_dim, 2).to(self.device)]
        # body
        self.body_clf = [nn.Linear(self.attr_dim, 4).to(self.device)]
        self.body_type_clf = [nn.Linear(self.attr_dim, 1).to(self.device)]
        self.body_color_clf = [nn.Linear(self.attr_dim, 8).to(self.device)]
        # leg
        self.leg_clf = [nn.Linear(self.attr_dim, 3).to(self.device)]
        self.leg_color_clf = [nn.Linear(self.attr_dim, 9).to(self.device)]
        # foot
        self.foot_clf = [nn.Linear(self.attr_dim, 3).to(self.device)]
        self.foot_color_clf = [nn.Linear(self.attr_dim, 4).to(self.device)]
        # bag
        self.bags_clf = [nn.Linear(self.attr_dim, 4).to(self.device)]
        # gender
        self.age_clf = [nn.Linear(self.attr_dim, 4).to(self.device)]
        self.gender_clf = [nn.Linear(self.attr_dim, 1).to(self.device)]                    
    
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
        clf_out = []        
        if sep_fc:
            for i,fc in enumerate(fc_layer):
                feature = fc_layer[i](x)
                fc_out.append(feature)
                clf_out.append(clf_layer[i](feature))
            if need_feature:
                return fc_out
            else:
                return clf_out                
        else:
            fc_out.append(fc_layer[0](x))
            if need_feature:
                return fc_out
            else:
                if sep_clf:
                    for clf in clf_layer:
                       clf_out.append(clf(fc_out[0]))
                    return clf_out
                else:
                    clf_out.append(clf_layer[0](fc_out[0]))
                    return clf_out
                
    def attr_branch(self, x, fc_layer, clf_layer,
                    conv_layer, trans1, trans2, mlp1, mlp2, sep_fc=False,
                    sep_clf=False, need_feature=False):
        ''' fc_layer should be a list of fully connecteds
            clf_layer hould be a list of classifiers
        '''
        # handling conv layer
        if conv_layer:
            x = conv_layer(x)
        x = x.reshape(x.size(0), 128, -1).permute(0,2,1)
        x = mlp1(trans1(x, x, x))
        x = self.maxpool1d(x.permute(0,2,1)).permute(0,2,1) 
        x = mlp2(trans2(x, x, x))
        x = self.maxpool1d(x.permute(0,2,1))
        x = x.reshape(x.size(0), -1)
        out = self.fc2clf(x=x, fc_layer=fc_layer,
                          clf_layer=clf_layer,
                          sep_clf=sep_clf, sep_fc=sep_fc,
                          need_feature=need_feature)
        return out 
    
    def forward(self, x, need_feature=False):
        out_conv4 = self.out_layers_extractor(x, 'out_conv4')       
        # head
        out_head = self.attr_branch(out_conv4, fc_layer = self.head_fc,
                                                     clf_layer = self.head_clf,
                                                     conv_layer = self.conv_head,
                                                     trans1 = self.trans_head1, 
                                                     trans2 = self.trans_head2,
                                                     mlp1 = self.mlp_head1, mlp2 = self.mlp_head2,
                                                     need_feature = need_feature)[0] 
        # head_colour
        out_head_colour = self.attr_branch(out_conv4, fc_layer = self.head_color_fc,
                                                     clf_layer = self.head_color_clf,
                                                     conv_layer = self.conv_head_color,
                                                     trans1 = self.trans_head_color1, 
                                                     trans2 = self.trans_head_color2,
                                                     mlp1 = self.mlp_head_color1, mlp2 = self.mlp_head_color2,
                                                     need_feature = need_feature)[0] 
        # body
        out_body = self.attr_branch(out_conv4, fc_layer = self.body_fc,
                                                     clf_layer = self.body_clf,
                                                     conv_layer = self.conv_body,
                                                     trans1 = self.trans_body1, 
                                                     trans2 = self.trans_body2,
                                                     mlp1 = self.mlp_body1, mlp2 = self.mlp_body2,
                                                     need_feature = need_feature)[0] 
        # out_body_colour
        out_body_colour = self.attr_branch(out_conv4, fc_layer = self.body_color_fc,
                                                     clf_layer = self.body_color_clf,
                                                     conv_layer = self.conv_body_color,
                                                     trans1 = self.trans_body_color1, 
                                                     trans2 = self.trans_body_color2,
                                                     mlp1 = self.mlp_body_color1, mlp2 = self.mlp_body_color2,
                                                     need_feature = need_feature)[0] 
        # body_type
        out_body_type = self.attr_branch(out_conv4, fc_layer = self.body_type_fc,
                                                     clf_layer = self.body_type_clf,
                                                     conv_layer = self.conv_body_type,
                                                     trans1 = self.trans_body_type1, 
                                                     trans2 = self.trans_body_type2,
                                                     mlp1 = self.mlp_body_type1, mlp2 = self.mlp_body_type2,
                                                     need_feature = need_feature)[0] 
        # leg
        out_leg = self.attr_branch(out_conv4, fc_layer = self.leg_fc,
                                                     clf_layer = self.leg_clf,
                                                     conv_layer = self.conv_leg,
                                                     trans1 = self.trans_leg1, 
                                                     trans2 = self.trans_leg2,
                                                     mlp1 = self.mlp_leg1, mlp2 = self.mlp_leg2,
                                                     need_feature = need_feature)[0] 
        # out_leg_colour
        out_leg_colour = self.attr_branch(out_conv4, fc_layer = self.leg_color_fc,
                                                     clf_layer = self.leg_color_clf,
                                                     conv_layer = self.conv_leg_color,
                                                     trans1 = self.trans_leg_color1, 
                                                     trans2 = self.trans_leg_color2,
                                                     mlp1 = self.mlp_leg_color1, mlp2 = self.mlp_leg_color2,
                                                     need_feature = need_feature)[0] 
        # foot
        out_foot = self.attr_branch(out_conv4, fc_layer = self.foot_fc,
                                                     clf_layer = self.foot_clf,
                                                     conv_layer = self.conv_foot,
                                                     trans1 = self.trans_foot1, 
                                                     trans2 = self.trans_foot2,
                                                     mlp1 = self.mlp_foot1, mlp2 = self.mlp_foot2,
                                                     need_feature = need_feature)[0] 
        # out_foot_colour
        out_foot_colour = self.attr_branch(out_conv4, fc_layer = self.foot_color_fc,
                                                     clf_layer = self.foot_color_clf,
                                                     conv_layer = self.conv_foot_color,
                                                     trans1 = self.trans_foot_color1, 
                                                     trans2 = self.trans_foot_color2,
                                                     mlp1 = self.mlp_foot_color1, mlp2 = self.mlp_foot_color2,
                                                     need_feature = need_feature)[0]
        # bag
        out_bags = self.attr_branch(out_conv4, fc_layer = self.bags_fc,
                                                     clf_layer = self.bags_clf,
                                                     conv_layer = self.conv_bags,
                                                     trans1 = self.trans_bags1, 
                                                     trans2 = self.trans_bags2,
                                                     mlp1 = self.mlp_bags1, mlp2 = self.mlp_bags2,
                                                     need_feature = need_feature)[0]
        # age
        out_age = self.attr_branch(out_conv4, fc_layer = self.age_fc,
                                                     clf_layer = self.age_clf,
                                                     conv_layer = self.conv_age,
                                                     trans1 = self.trans_age1, 
                                                     trans2 = self.trans_age2,
                                                     mlp1 = self.mlp_age1, mlp2 = self.mlp_age2,
                                                     need_feature = need_feature)[0]
        # out_gender
        out_gender = self.attr_branch(out_conv4, fc_layer = self.gender_fc,
                                                     clf_layer = self.gender_clf,
                                                     conv_layer = self.conv_gender,
                                                     trans1 = self.trans_gender1, 
                                                     trans2 = self.trans_gender2,
                                                     mlp1 = self.mlp_gender1, mlp2 = self.mlp_gender2,
                                                     need_feature = need_feature)[0]
        out_attributes = {'head':out_head,
                                 'head_colour':out_head_colour,
                                 'body':out_body,
                                 'body_type':out_body_type,
                                 'leg':out_leg,
                                 'foot':out_foot,
                                 'gender':out_gender,
                                 'bags':out_bags,
                                 'body_colour':out_body_colour,
                                 'leg_colour':out_leg_colour,
                                 'foot_colour':out_foot_colour,
                                 'age':out_age}        
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
                                
#%%

class attributes_model(nn.Module):
    
    '''
    a model for training whole attributes 
    '''
    def __init__(self,
                 model,
                 feature_dim = 512,
                 attr_dim = 79):
        
        super().__init__()
        self.feature_dim = feature_dim
        self.model = model     
        self.attr_lin = nn.Linear(in_features=feature_dim , out_features=attr_dim)  
                
    def out_layers_extractor(self, x, layer):
        out_os_layers = self.model.layer_extractor(x, layer) 
        return out_os_layers   
        
    def forward(self, x, get_features = False):
        
        features = self.out_layers_extractor(x, 'fc') 
        if get_features:
            return features
        else:
            return {'attributes':self.attr_lin(features)}

    def save_baseline(self, saving_path):
        torch.save(self.model.state_dict(), saving_path)
        print('baseline model save to {}'.format(saving_path))  
        
#%%
from torchvision import transforms
class Loss_weighting(nn.Module):
    
    '''
    a model for training weights of loss functions
    '''
    def __init__(self, weights_dim=48):
        
        super().__init__()
        self.weights_dim = weights_dim

        self.weights_lin1 = nn.Linear(in_features=weights_dim , out_features=weights_dim) 
        self.weights_lin2 = nn.Linear(in_features=weights_dim , out_features=weights_dim)
        self.relu = nn.ReLU()
    def forward(self, weights):
        weights = self.weights_lin1(weights) 
        weights = self.relu(weights)
        weights = self.weights_lin2(weights) 
        weights = torch.sigmoid(weights)
        return weights

    def save_baseline(self, saving_path):
        torch.save(self.weights_lin.state_dict(), saving_path)
        print('loss_weights saved to {}'.format(saving_path)) 


class mb_CA_auto_build_model(nn.Module):
    
    def __init__(self,
                 model,
                 main_cov_size = 512,
                 attr_dim = 128,
                 dropout_p = 0.3,
                 sep_conv_size = 64,
                 branch_names = None,
                 feature_selection = None):

        super().__init__()
        self.feat_indices = feature_selection
        self.feature_dim = main_cov_size
        if self.feature_dim != 384 and self.feature_dim != 512:
            raise Exception('main_cov_size should be 384 or 512')
        self.dropout_p = dropout_p 
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.model = model
        self.sep_conv_size = sep_conv_size
        self.attr_dim = attr_dim
        self.branch_names = branch_names

        if self.feat_indices is not None:
            self.feature_dim = 25
        
        self.attr_feat_dim = sep_conv_size
        self.branches = {}
        for k in self.branch_names.keys():
            # convs
            setattr(self, 'conv_'+k, _make_layer(blocks[2],
                                                    layers[2],
                                                    self.feature_dim,
                                                    self.sep_conv_size,
                                                    reduce_spatial_size=False
                                                    ))
            
            # fully connecteds
            setattr(self, 'fc_'+k, self._construct_fc_layer(self.attr_dim, self.attr_feat_dim, dropout_p=dropout_p))
            # classifiers
            setattr(self, 'clf_'+k, nn.Linear(self.attr_dim, branch_names[k]))

        
    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
    
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
    
        return nn.Sequential(*layers)

    def get_feature(self, x, get_attr=True, get_feature=True, method='both', get_collection=False):
        
        if self.feature_dim == 512:
            out_conv4 = self.out_layers_extractor(x, 'out_conv4')       
        elif self.feature_dim == 384:
            out_conv4 = self.out_layers_extractor(x, 'out_conv3')  
        else:
            raise Exception('main_cov_size should be 384 or 512')
        
        out_features = {}

        for k in self.branch_names.keys():
            out_features.setdefault(k, self.attr_branch(out_conv4 if self.feat_indices == None else torch.index_select(out_conv4, 1, self.feat_indices[0]),
                                                            fc_layer = getattr(self,'fc_'+k),
                                                            clf_layer = getattr(self,'clf_'+k),
                                                            conv_layer = getattr(self,'conv_'+k), need_feature = True)
            )

        del out_conv4
        out_fc_branches = [item[0] for item in list(out_features.values())]
        outputs_clfs = {}
        for k, v in out_features.items():
            outputs_clfs.update({k: v[1]})

        x = self.out_layers_extractor(x, 'out_fc')
        out_fc_branches = torch.cat(out_fc_branches, dim=1)
        if method == 'both':
            outputs_fcs = torch.cat((out_fc_branches,x), dim=1)
        elif method == 'baseline':
            outputs_fcs = x
        elif method == 'branches':
            outputs_fcs = out_fc_branches
        return outputs_fcs, outputs_clfs
        
    
    def vector_features(self, x):
        features = self.model(x)
        out_attr = self.attr_lin(features) 
        out_features = torch.cat(features, out_attr, dim=1)
        return out_features
        
    def out_layers_extractor(self, x, layer):
        out_os_layers = self.model.layer_extractor(x, layer) 
        return out_os_layers   
                
    def attr_branch(self, x, fc_layer, clf_layer,
                    conv_layer=None, need_feature=False):
        ''' fc_layer should be a list of fully connecteds
            clf_layer hould be a list of classifiers
        '''
        # handling conv layer
        if conv_layer:
            x = conv_layer(x)
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        x = fc_layer(x)
        if need_feature:
                out = clf_layer(x)
                return x, out
        out = clf_layer(x)
        return out 
    
    def forward(self, x, need_feature=False):
        if self.feature_dim == 512:
            out_conv4 = self.out_layers_extractor(x, 'out_conv4')       
        elif self.feature_dim == 384:
            out_conv4 = self.out_layers_extractor(x, 'out_conv3')  
        else:
            raise Exception('main_cov_size should be 384 or 512')
        
        out_attributes = {}

        for k in self.branch_names.keys():
            out_attributes.setdefault(k, self.attr_branch(out_conv4 if self.feat_indices == None else torch.index_select(out_conv4, 1, self.feat_indices[0]),
                                                            fc_layer = getattr(self,'fc_'+k),
                                                            clf_layer = getattr(self,'clf_'+k),
                                                            conv_layer = getattr(self,'conv_'+k), need_feature = need_feature)
            )

        return out_attributes
    
    def save_baseline(self, saving_path):
        torch.save(self.model.state_dict(), saving_path)
        print('baseline model save to {}'.format(saving_path))


branch_channels = [16, 64, 96, 128]
base_channels = [64,64,256,384,512]
class mb_CA_auto_same_depth_build_model(nn.Module):
    
    def __init__(self,
                 model,
                 branch_place,
                 dropout_p = 0.3,
                 branch_names = None,
                 feature_selection = None):
        super().__init__()
        self.branch_place = branch_place
        self.feat_indices = feature_selection
        self.dropout_p = dropout_p 
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.model = model
        self.branch_fcs = branch_names

        self.layer_list = ['conv1', 'maxpool', 'conv2', 'conv3', 'conv4']
        self.layer_init_dim = [64,64,256,384,512]

        if self.feat_indices is not None:
            self.feature_dim = 25
        
        branches = {k:[] for k,v in self.branch_fcs.items()}
        for k in self.branch_fcs.keys():
            # convs
            if branch_place not in ['conv5', 'conv4']:
                
                idx = self.layer_list.index(branch_place)-1

                for i, layer in enumerate(self.layer_list[self.layer_list.index(branch_place)+1:]):
                    
                    branches[k].append(_make_layer(
                                                    blocks[idx],
                                                    layers[idx],
                                                    self.layer_init_dim[idx+1] if i==0 else branch_channels[idx],
                                                    branch_channels[idx+1],
                                                    reduce_spatial_size=False if layer=='conv4' else True
                                                ))
                    idx += 1
                
            # classifiers
            # if branch_place == 'conv4' or branch_place == 'conv5':
            idx = self.layer_list.index('conv4')-1
            if branch_place != 'conv5':
                branches[k].append(Conv1x1(base_channels[idx+1] if branch_place=='conv4' else branch_channels[idx] ,
                                           branch_channels[idx]))
                branches[k].append(nn.AdaptiveAvgPool2d(1))
                branches[k].append(self._construct_fc_layer(branch_channels[idx],
                                                            branch_channels[idx], dropout_p=None))
            else:
                branches[k].append(nn.AdaptiveAvgPool2d(1))
                branches[k].append(self._construct_fc_layer(branch_channels[idx],
                                                            base_channels[idx+1], dropout_p=None))
            branches[k].append(nn.Linear(branch_channels[idx], branch_names[k]))
            setattr(self, 'branch_'+k, nn.Sequential(*branches[k]))
            '''# convs
            for layer in self.layer_list[self.layer_list.index(branch_place)+1:]:
                branches[k].append(copy.deepcopy(getattr(model, layer)))
                branches[k][-1].load_state_dict(getattr(model, layer).state_dict())

            # classifiers
            branches[k].append(nn.Linear(self.attr_dim, branch_names[k]))
            setattr(self, 'branch_'+k, nn.Sequential(*branches[k]))
        self.layer_list.append('clf')'''

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
    
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
    
        return nn.Sequential(*layers)

    def get_feature(self, x, get_attr=True, get_feature=True, method='both', get_collection=False):
        
        out_baseline = self.out_layers_extractor(x, self.branch_place)       
        out_attributes = {}

        for k in self.branch_fcs.keys():
            out_attributes.setdefault(k, self.attr_branch(out_baseline if self.feat_indices == None else torch.index_select(out_baseline, 1, self.feat_indices[0]),
                                                            getattr(self,'branch_'+k),
                                                            need_feature = False)
            )
        out_baseline = self.out_layers_extractor(x, 'fc')

        return out_attributes, out_baseline
        
    
    def vector_features(self, x):
        features = self.model(x)
        out_attr = self.attr_lin(features) 
        out_features = torch.cat(features, out_attr, dim=1)
        return out_features
        
    def out_layers_extractor(self, x, layer):
        out_os_layers = self.model.layer_extractor(x, layer) 
        return out_os_layers   
                
    def attr_branch(self, x, branch_layers, need_feature=False):
        ''' fc_layer should be a list of fully connecteds
            clf_layer hould be a list of classifiers
        '''
        # handling conv layer
        if self.branch_place != 'conv5':
            start_point = self.layer_list.index(self.branch_place)
        else:
            start_point = self.layer_list.index('conv4')
        
        for idx, layer in enumerate(branch_layers):

            if need_feature and self.layer_list[start_point+idx+1] == 'clf':
                features = x
                attr = layer(features)
                return features, attr
            if layer == branch_layers[-2]:
                x = x.view(x.size(0), -1)
            x = layer(x)

        return x 
    
    def forward(self, x, need_feature=False):

        out_baseline = self.out_layers_extractor(x, self.branch_place)       
        out_attributes = {}

        for k in self.branch_fcs.keys():
            out_attributes.setdefault(k, self.attr_branch(out_baseline if self.feat_indices == None else torch.index_select(out_baseline, 1, self.feat_indices[0]),
                                                            getattr(self,'branch_'+k),
                                                            need_feature = need_feature)
            )

        return out_attributes
    
    def save_baseline(self, saving_path):
        torch.save(self.model.state_dict(), saving_path)
        print('baseline model save to {}'.format(saving_path))