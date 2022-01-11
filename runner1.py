#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 20:07:29 2022

@author: hossein
"""
# repository imports
from utils import get_n_params, part_data_delivery, resampler, attr_weight
from trainings import dict_training_multi_branch
from models import mb_transformer_build_model, mb_os_build_model
from delivery import data_delivery
from loaders import CA_Loader
# torch requirements
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('calculation is on:',device)
torch.cuda.empty_cache()


#%%
def parse_args():
    parser = argparse.ArgumentParser(
        description ='identify the most similar clothes to the input image')
    parser.add_argument(
        '--dataset',
        type = str,
        help = 'one of dataset = [CA_Market, Market_attribute, CA_Duke, Duke_attribute]',
        default='Duke_attribute')
    
    parser.add_argument(
        '--main_path',
        type = str,
        help = 'if your dataset is CA_Market or Market_attribute our work use gt_bbox folder of dataset',
        default = './datasets/Market1501/Market-1501-v15.09.15/gt_bbox/')

    parser.add_argument(
        '--train_path',
        type = str,
        help = 'path of training images. only for Dukes',
        default = './datasets/Dukemtmc/bounding_box_train')
    
    parser.add_argument(
        '--test_path',
        type = str,
        help = 'path of training images. only for Dukes',
        default = './datasets/Dukemtmc/bounding_box_test')
    
    parser.add_argument(
        '--attr_path',
        type = str,
        help ='./attributes/CA_Market.npy'+
                './attributes/Market_attribute_with_id.npy',
        default = './attributes/Market_attribute_with_id.npy')

    parser.add_argument(
        '--attr_path_train',
        type = str,
        help ='path of attributes: for Dukes train_attr and test_attr are required and for Markets attributes vector is enough',
        default = './attributes/Duke_attribute_test_with_id.npy')

    parser.add_argument(
        '--attr_path_test',
        type = str,
        help ='path of attributes: for Dukes train_attr and test_attr are required and for Markets attributes vector is enough',
        default = './attributes/Duke_attribute_test_with_id.npy')
    
    parser.add_argument(
        '--training_strategy',
        type = str,
        help = 'categorized or vectorized',
        default='categorized')
    
    parser.add_argument(
        '--model',
        type = str,
        help = 'it should be one the [osnet, sbs_s50, mgn_r50_ibn]',
        default='osnet')
    
    parser.add_argument(
        '--topN', type = int, default=20, help = 'retrieve topN items')
    
    parser.add_argument(
        '--source_clothes',
        help = 'path of source images which are shops items',
        default = '/home/hossein/deep-person-reid/datasets/market1501/Market-1501-v15.09.15/bounding_box_test')
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='/home/hossein/Downloads/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth',
        help='the checkpoint network to resume from')
    
    parser.add_argument(
        '--use_ready_features', action='store_true')
    parser.set_defaults(use_ready_features=True)
    
    parser.add_argument(
        '--all_features',
        type=str,
        default='/home/hossein/anaconda3/envs/torchreid/deep-person-reid/my_osnet/demo/features/market_gallery_osnet.npy',
        help='features of whole data that we want to retrieve')
    
    parser.add_argument(
        '--save_features',
        type=str,
        default='/home/hossein/anaconda3/envs/torchreid/deep-person-reid/my_osnet/demo/features',
        help='features of whole data that we want to retrieve')

    args = parser.parse_args()
    return args


#%%

args = parse_args()

''' Delivering data as attr dictionaries '''
part_based = True if args.training_strategy == 'categorized' else False # if categotized, for each part one tesor will be genetrated
                                                                        # elif vectorize, only an attribute vector will be generated 

if args.dataset == 'CA_Market' or args.dataset == 'Market_attribute':
    main_path = args.main_path  
    path_attr = args.attr_path 
    attr = data_delivery(main_path,
                  path_attr=path_attr,
                  need_parts=part_based,
                  need_attr=not part_based,
                  dataset = args.dataset)
    
    for key , value in attr.items():
      try: print(key , 'size is: \t {}'.format((value.size())))
      except TypeError:
        print(key)
    
else:
    
    train_img_path = args.train_path
    test_img_path = args.test_path
    path_attr_train = args.attr_path_train
    path_attr_test = args.attr_path_test
    
    attr_train = data_delivery(train_img_path,
                      path_attr=path_attr_train,
                      need_parts=part_based,
                      need_attr=not part_based,
                      dataset = args.dataset)
    
    attr_test = data_delivery(test_img_path,
                      path_attr=path_attr_test,
                      need_parts=part_based,
                      need_attr=not part_based,
                      dataset = args.dataset)
    print('\n', 'train-set specifications') 
    for key , value in attr_train.items():       
        try: print(key , 'size is: \t {}'.format((value.size())))
        except TypeError:
          print(key)
   
    print('\n', 'test-set specifications') 
    for key , value in attr_test.items():
        try: print(key , 'size is: \t {}'.format((value.size())))
        except TypeError:
          print(key)

#%%    
# train_transform = transforms.Compose([
#                             transforms.RandomRotation(degrees=10),
#                             transforms.RandomHorizontalFlip(),
#                             transforms.ColorJitter(saturation=[1,3])
#                             ])

# train_data = CA_Loader(img_path=main_path,
#                           attr=attr,
#                           resolution=(256,128),
#                           transform=train_transform,
#                           indexes=train_idx,
#                           need_attr =False,
#                           need_collection=True,
#                           need_id = False,
#                           two_transforms = False)

# train_data.head , train_data.img_names = resampler(train_data.head ,
#                                                      train_data.img_names,
#                                                      Most_repetition = 5)

# test_data = CA_Loader(img_path=main_path,
#                           attr=attr,
#                           resolution=(256, 128),
#                           indexes=test_idx,
#                           need_attr = False,
#                           need_collection=True,
#                           need_id = False,
#                           two_transforms = False,                          
#                           ) 

# batch_size = 32
# train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
# test_loader = DataLoader(test_data,batch_size=100,shuffle=False)
# #%%
# torch.cuda.empty_cache()
# from torchreid import models
# from torchreid import utils

# model = models.build_model(
#     name='osnet_x1_0',
#     num_classes=751,
#     loss='softmax',
#     pretrained=False
# )

# weight_path = './checkpoints/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth'
# utils.load_pretrained_weights(model, weight_path)
# # sep_fc = True and sep_clf = False is not possible
# attr_net = mb_os_build_model(model = model,
#                  main_cov_size = 512,
#                  attr_dim = 64,
#                  dropout_p = 0.3,
#                  sep_conv_size = 128,
#                  sep_fc = False,
#                  sep_clf = True)

# attr_net = attr_net.to(device)

# i = 0
# for child in attr_net.children():
#     i += 1
#     # print(i)
#     # print(child)
#     if i == 10:
#     #     # print(child)
#         a = get_n_params(child)
        
# get_n_params(attr_net)
# #%%
# weights = {'gender':torch.rand(1, device=device),'leg' : torch.rand(3, device=device)}
# #%%
# part_loss = part_data_delivery(weights, dataset='CA_Market')

# params = attr_net.parameters()

# lr = 3.5e-4
# optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.99), eps=1e-08)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 17], gamma=0.1)

# #%%
# save_path = './results/'
# dict_training_multi_branch(num_epoch = 30,
#                       attr_net = attr_net,
#                       train_loader = train_loader,
#                       test_loader = test_loader,
#                       optimizer = optimizer,
#                       scheduler = scheduler,
#                       save_path = save_path,  
#                       part_loss = part_loss,
#                       device = device,
#                       version = 'sif_convt_128_flf_64_clft_CA',
#                       resume=False,
#                       loss_train = None,
#                       loss_test=None,
#                       train_attr_F1=None,
#                       test_attr_F1=None,
#                       train_attr_acc=None,
#                       test_attr_acc=None,  
#                       stoped_epoch=None)





