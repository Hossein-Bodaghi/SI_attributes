#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 19:34:14 2022

@author: hossein
"""

import os 
from models import mb12_CA_build_model
import torch
from torchreid.models import build_model
from torchreid import utils



model = build_model(
    name='osnet_x1_0',
    num_classes=751,
    loss='softmax',
    pretrained=False
)

utils.load_pretrained_weights(model, '/home/hossein/SI_attributes/checkpoints/osnet_x1_0_market.pth')



net_paths = '/home/hossein/SI_attributes/results/mb_conv3_12branches_nowei_CA'
net_names = os.listdir(net_paths)

main_net = mb12_CA_build_model(
                    model=model,
                  main_cov_size = 384,
                  attr_dim = 64,
                  dropout_p = 0.3,
                  sep_conv_size = 64,
                  feature_selection = None)
init_dict = main_net.state_dict()

part_names = []

for net_name in net_names:
    
    attr_net = mb12_CA_build_model(
                        model=model,
                      main_cov_size = 384,
                      attr_dim = 64,
                      dropout_p = 0.3,
                      sep_conv_size = 64,
                      feature_selection = None)
    
    trained_net_path = torch.load(os.path.join(net_paths, net_name+'/best_attr_net.pth'))
    attr_net.load_state_dict(trained_net_path)
    
    flag = 0
    b = net_name.split('_')
    if len(b) == 5:
        part_name = b[2]
    else:
        part_name = '_'.join([b[2],b[3]])
        flag += 1
        
    for state_key in attr_net.state_dict():
        # print(state_key)
        state_names = state_key.split('.')
        if state_names[0] != 'model':

            state_name = state_names[0].split('_')
            if flag == 1:
                alter_part_name = '_'.join([b[2],'color'])
                if state_name[0] == alter_part_name:
                    part_state = main_net.state_dict()
                    part_state[state_key] = attr_net.state_dict()[state_key]
                    main_net.load_state_dict(part_state)
                    
                elif '_'.join([state_name[0], state_name[1]]) == alter_part_name:
                    part_state = main_net.state_dict()
                    part_state[state_key] = attr_net.state_dict()[state_key]
                    main_net.load_state_dict(part_state)
                    
                elif '_'.join([state_name[-2], state_name[-1]]) == part_name:
                    part_state = main_net.state_dict()
                    part_state[state_key] = attr_net.state_dict()[state_key]
                    main_net.load_state_dict(part_state)            
            else:   
                if part_name == 'bags':
                    alter_part_name = 'bag'
                else:
                    alter_part_name = part_name
                if state_name[0] == alter_part_name:
                    part_state = main_net.state_dict()
                    part_state[state_key] = attr_net.state_dict()[state_key]
                    main_net.load_state_dict(part_state)
                    
                elif '_'.join([state_name[0], state_name[1]]) == alter_part_name:
                    part_state = main_net.state_dict()
                    part_state[state_key] = attr_net.state_dict()[state_key]
                    main_net.load_state_dict(part_state)
                
                elif '_'.join([state_name[-2], state_name[-1]]) == part_name:
                    part_state = main_net.state_dict()
                    part_state[state_key] = attr_net.state_dict()[state_key]
                    main_net.load_state_dict(part_state)

saving_path = '/home/hossein/SI_attributes/results/mb_conv3_12branches_nowei_CA_network'
torch.save(main_net.state_dict(), os.path.join(saving_path, 'best_attr_net.pth'))
        
        

        

