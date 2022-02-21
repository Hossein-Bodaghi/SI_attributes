

from utils import get_n_params, part_data_delivery, resampler, attr_weight, validation_idx, LGT, iou_worst_plot, common_attr
from trainings import dict_training_multi_branch, dict_evaluating_multi_branch, take_out_multi_branch, dict_training_dynamic_loss
from models import mb12_CA_build_model, attributes_model, Loss_weighting, mb_CA_auto_build_model
from evaluation import metrics_print, total_metrics
from delivery import data_delivery
from metrics import tensor_metrics, IOU
from loaders import CA_Loader
# torch requirements
from torch.utils.data import DataLoader
from torchvision import transforms
import torch, re, os
import argparse
# others
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('calculation is on:',device)
torch.cuda.empty_cache()

part_based = True  # if categotized, for each part one tesor will be genetrated
cross_domain =  False # if categotized, for each part one tesor will be genetrated
                                                                        # elif vectorize, only an attribute vector will be generated 

attr = data_delivery('./datasets/Market1501/Market-1501-v15.09.15/gt_bbox/',
                path_attr='./attributes/Market_attribute_with_id.npy',
                need_parts=part_based,
                need_attr=not part_based,
                dataset = 'Market_attribute')

old_attr = np.load('C:/Users/ASUS/Desktop/total_attr.npy')
old_attr = torch.from_numpy(old_attr)
old_bags = old_attr[:,28:31]
for idx, x in enumerate(attr['bags']):
    if x[1] == 1:
        old_bags[idx][1] = 1
        old_bags[idx][0] = 0
        old_bags[idx][2] = 0
    
        #print(attr['img_names'][idx])


attr_ca = data_delivery('./datasets/Market1501/Market-1501-v15.09.15/gt_bbox/',
                path_attr='./attributes/CA_Market_with_id.npy',
                need_parts=part_based,
                need_attr=not part_based,
                dataset = 'CA_Market')

attr_ca['bags'] = old_bags

for idx, x in enumerate(attr_ca['body']):
    if x[0] == 1:
        x[1] = 1
attr_ca['body'] = attr_ca['body'][:,1:]

attr_ca.pop('age')
all = torch.Tensor()
for k,v in attr_ca.items():
    if k not in ['id','cam_id','img_names','names']:
        all = torch.cat((all, v), 1)
        '''for vv in v:
            if sum(vv)>1:
                print('oh no')'''
print('done')