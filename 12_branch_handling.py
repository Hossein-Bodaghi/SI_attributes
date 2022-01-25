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





import os 
from trainings import take_out_multi_branch

net_paths = '/home/hossein/SI_attributes/results/mb_conv3_12branches_nowei_CA'
net_names = os.listdir(net_paths)
final_target = torch.zeros((len(test_data), len(attr['names'])))

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
    
    predicts, targets = take_out_multi_branch(attr_net = attr_net,
                                   test_loader = test_loader,
                                   save_path = save_path,
                                   device = device,
                                   part_loss = part_loss,
                                   categorical = part_based)   
    flag = 0
    b = net_name.split('_')
    if len(b) == 5:
        part_name = b[2]
        part_names.append(part_name)
    else:
        part_name = '_'.join([b[2],b[3]])
        part_names.append(part_name)
        flag += 1
        
    if part_name == 'gender': final_target[:,0] = predicts[:,0]
    elif part_name == 'head': final_target[:,1:6] = predicts[:,1:6]
    elif part_name == 'head_colour': final_target[:,6:8] = predicts[:,6:8]
    elif part_name == 'body': final_target[:,8:12] = predicts[:,8:12]
    elif part_name == 'body_type': final_target[:,12] = predicts[:,12]
    elif part_name == 'body_colour': final_target[:,13:21] = predicts[:,13:21]
    elif part_name == 'bags': final_target[:,21:25] = predicts[:,21:25]
    elif part_name == 'leg': final_target[:,25:28] = predicts[:,25:28]
    elif part_name == 'leg_colour': final_target[:,28:37] = predicts[:,28:37]
    elif part_name == 'foot': final_target[:,37:40] = predicts[:,37:40]                      
    elif part_name == 'foot_colour': final_target[:,40:44] = predicts[:,40:44]
    elif part_name == 'age': final_target[:,44:48] = predicts[:,44:48]
    
from metrics import tensor_metrics, IOU
attr_metrics = tensor_metrics(targets, final_target)    


for metric in ['precision', 'recall', 'accuracy', 'f1', 'mean_accuracy']:
    if args.dataset == 'CA_Market' or args.dataset == 'Market_attribute' or args.dataset == 'PA100k':
        metrics_print(attr_metrics, attr['names'], metricss=metric)
    else:
        metrics_print(attr_metrics, attr_test['names'], metricss='precision')

total_metrics(attr_metrics)
iou_result = IOU(final_target, targets)
print('\n','the mean of iou is: ',str(iou_result.mean().item()))
if args.dataset == 'CA_Market' or args.dataset == 'Market_attribute' or args.dataset == 'PA100k':            
    iou_worst_plot(iou_result=iou_result, valid_idx=test_idx, main_path=main_path, attr=attr, num_worst = args.num_worst)
else:    
    iou_worst_plot(iou_result=iou_result, valid_idx=test_idx, main_path=test_img_path, attr=attr, num_worst = args.num_worst)
 
metrics_result = attr_metrics[:5]
attr_metrics_pd = pd.DataFrame(data = np.array([result.numpy() for result in metrics_result]).T,
                               index=attr['names'], columns=['precision','recall','accuracy','f1','MA'])
attr_metrics_pd.to_excel('/home/hossein/SI_attributes/results/mb_conv3_12branches_nowei_CA_network/attr_metrics.xlsx')

mean_metrics = attr_metrics[5:]
mean_metrics.append(iou_result.mean().item())
mean_metrics_pd = pd.DataFrame(data = np.array(mean_metrics), index=['precision','recall','accuracy','f1','MA', 'IOU'])
peices = args.save_attr_metrcis.split('/')
peices[-1] = 'mean_metrics.xlsx'
path_mean_metrcis = '/'.join(peices)
mean_metrics_pd.to_excel('/home/hossein/SI_attributes/results/mb_conv3_12branches_nowei_CA_network/mean_metrics.xlsx')

#%%%%%%%%%%%%%%%%%%%%

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
        
        
        
#%%

import numpy as np
from delivery import data_delivery


CA_Duke_test_with_id = '/home/hossein/SI_attributes/attributes/CA_Duke_test_with_id.npy'
CA_Market_with_id = '/home/hossein/SI_attributes/attributes/CA_Market_with_id.npy'
Duke_attribute_test_with_id = '/home/hossein/SI_attributes/attributes/Duke_attribute_test_with_id.npy'
Market_attribute_with_id = '/home/hossein/SI_attributes/attributes/Market_attribute_with_id.npy'
PA100k_all_with_id = '/home/hossein/SI_attributes/attributes/PA100k_all_with_id.npy'


duke_path = '/home/hossein/SI_attributes/datasets/Dukemtmc/bounding_box_test'
market_path = '/home/hossein/SI_attributes/datasets/Market1501/Market-1501-v15.09.15/gt_bbox/'
pa100k_path = '/home/hossein/SI_attributes/datasets/PA-100K/release_data/release_data/'

# [CA_Market, Market_attribute, CA_Duke, Duke_attribute, PA100k]
part_based = False
attr_duke_attr = data_delivery(duke_path,
      path_attr=Duke_attribute_test_with_id,
      need_parts=part_based,
      need_attr=not part_based,
      dataset = 'Duke_attribute')
for idx, name in enumerate(attr_duke_attr['names']):
    print(idx, ') --> ', name)
print('\n')

attr_ca_duke = data_delivery(duke_path,
      path_attr=CA_Duke_test_with_id,
      need_parts=part_based,
      need_attr=not part_based,
      dataset = 'CA_Duke')
for idx, name in enumerate(attr_ca_duke['names']):
    print(idx, ') --> ', name)
print('\n')

attr_market_attr = data_delivery(market_path,
      path_attr=Market_attribute_with_id,
      need_parts=part_based,
      need_attr=not part_based,
      dataset = 'Market_attribute')
for idx, name in enumerate(attr_market_attr['names']):
    print(idx, ') --> ', name)
print('\n')

attr_ca_market = data_delivery(market_path,
      path_attr=CA_Market_with_id,
      need_parts=part_based,
      need_attr=not part_based,
      dataset = 'CA_Market')
for idx, name in enumerate(attr_ca_market['names']):
    print(idx, ') --> ', name)
print('\n')
    
attr_pa100k = data_delivery(pa100k_path,
      path_attr=PA100k_all_with_id,
      need_parts=part_based,
      need_attr=not part_based,
      dataset = 'PA100k')
for idx, name in enumerate(attr_pa100k['names']):
    print(idx, ') --> ', name)
