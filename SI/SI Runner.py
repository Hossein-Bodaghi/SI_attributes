#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 14:15:34 2021

@author: hossein
"""

from functions import load_model, latent_feat_extractor, si_calculator, load_saved_features, forward_selection_SI
from loaders import MarketLoader4
import torch
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

models = ['osnet', 'attr_net']
model_path = './result/V8_01/best_attr_net.pth'
weight_path = './osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth'

network = load_model(model=models[0], weight_path=weight_path)

from delivery import data_delivery 

test_idx = torch.load('./attributes/test_idx_full.pth')
main_path = './Market-1501-v15.09.15/gt_bbox/'
path_attr = './attributes/total_attr.npy'

attr = data_delivery(main_path=main_path,
                        path_attr=path_attr,
                        need_collection=True,
                        double=False,
                        need_attr=False)

test_data = MarketLoader4(img_path=main_path,
                            attr=attr,
                            resolution=(256, 128),
                            indexes=test_idx[0:100],
                            need_attr = False,
                            need_collection=True,
                            need_id = False,
                            two_transforms = False,                          
                            ) 


key = 'foot'
aa = test_data[0][key]

layer = 'out_conv4'
features_ready = False

if features_ready == True:
    out_layers = load_saved_features(layer)
else:
    test_loader = DataLoader(test_data, batch_size=200, shuffle=False)

    out_layers = latent_feat_extractor(net=network, test_loader=test_loader,
                                        layer=layer, save_path='./saving/', 
                                        device=device, use_adapt=False,
                                        final_size = (8, 8), mode='return')


run_SI_on_all = False

if run_SI_on_all == True:
    si_foot = si_calculator(out_layers, test_data(key))

trend, layer_nums = forward_selection_SI(out_layers.to('cuda'), test_data.foot.to('cuda'))

import numpy as np
trend = np.array(trend)
np.save('./results/trend'++'.npy', trend)