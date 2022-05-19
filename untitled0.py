#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 00:47:26 2022

@author: hossein
"""

import torch

optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.99), eps=1e-08)
op_state = optimizer.state_dict()
optimizer.load_state_dict(torch.load('./results/CA_Duke_vec_conv3_osnet_x1_0_duke_softmax/optimizer.pth', map_location = device))

'/home/hossein/SI_attributes/results/CA_Duke_vec_conv3_osnet_x1_0_duke_softmax/attr_net.pth'
'/home/hossein/SI_attributes/results/CA_Duke_vec_conv3_osnet_x1_0_duke_softmax/best_attr_net.pth'
'/home/hossein/SI_attributes/results/CA_Duke_vec_conv3_osnet_x1_0_duke_softmax/best_epoch.pth'
'/home/hossein/SI_attributes/results/CA_Duke_vec_conv3_osnet_x1_0_duke_softmax/scheduler.pth'
'/home/hossein/SI_attributes/results/CA_Duke_vec_conv3_osnet_x1_0_duke_softmax/optimizer.pth'
'/home/hossein/SI_attributes/results/CA_Duke_vec_conv3_osnet_x1_0_duke_softmax/test_attr_f1.pth'
'/home/hossein/SI_attributes/results/CA_Duke_vec_conv3_osnet_x1_0_duke_softmax/test_attr_acc.pth'
'/home/hossein/SI_attributes/results/CA_Duke_vec_conv3_osnet_x1_0_duke_softmax/test_part_loss.pth'
'/home/hossein/SI_attributes/results/CA_Duke_vec_conv3_osnet_x1_0_duke_softmax/test_attr_loss.pth'
'/home/hossein/SI_attributes/results/CA_Duke_vec_conv3_osnet_x1_0_duke_softmax/train_attr_f1.pth'
'/home/hossein/SI_attributes/results/CA_Duke_vec_conv3_osnet_x1_0_duke_softmax/train_attr_acc.pth'
'/home/hossein/SI_attributes/results/CA_Duke_vec_conv3_osnet_x1_0_duke_softmax/train_loss.pth'
'/home/hossein/SI_attributes/results/CA_Duke_vec_conv3_osnet_x1_0_duke_softmax/training_epoch.pth'
'/home/hossein/SI_attributes/results/CA_Duke_vec_conv3_osnet_x1_0_duke_softmax/train_part_loss.pth'