#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 20:07:29 2022

@author: hossein
"""
#%%
# repository imports
from utils import get_n_params, part_data_delivery, resampler, attr_weight, validation_idx, LGT, iou_worst_plot
from trainings import dict_training_multi_branch, dict_evaluating_multi_branch
from models import mb12_CA_build_model, attributes_model
from evaluation import metrics_print, total_metrics
from delivery import data_delivery
from loaders import CA_Loader
# torch requirements
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import argparse
# others
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('calculation is on:',device)
torch.cuda.empty_cache()


#%%
def parse_args():
    parser = argparse.ArgumentParser(description ='identify the most similar clothes to the input image')
    parser.add_argument('--dataset', type = str, help = 'one of dataset = [CA_Market, Market_attribute, CA_Duke, Duke_attribute, PA100k]', default='Duke_attribute')
    parser.add_argument('--mode', type = str, help = 'mode of runner = [train, eval]', default='train')
    parser.add_argument('--main_path',type = str,help = 'image_path = [Market1501/Market-1501-v15.09.15/gt_bbox/,PA-100K/release_data/release_data/]',default = './datasets/Market1501/Market-1501-v15.09.15/gt_bbox/')
    parser.add_argument('--train_path',type = str,help = 'path of training images. only for Dukes',default = './datasets/Dukemtmc/bounding_box_train')
    parser.add_argument('--test_path',type = str,help = 'path of training images. only for Dukes',default = './datasets/Dukemtmc/bounding_box_test')
    parser.add_argument('--attr_path',type = str,help = '[CA_Market_with_id, PA100k_all_with_id, Market_attribute_with_id]',default = './attributes/CA_Market_with_id.npy' )
    parser.add_argument('--attr_path_train',type = str,help =' [CA_Duke_train_with_id path , Duke_attribute_train_with_id]',default = './attributes/Duke_attribute_train_with_id.npy')
    parser.add_argument('--attr_path_test',type = str,help ='[Duke_attribute_test_with_id, CA_Duke_test_with_id]',default = './attributes/Duke_attribute_test_with_id.npy')
    parser.add_argument('--training_strategy',type = str,help = 'categorized or vectorized',default='categorized')    
    parser.add_argument('--training_part',type = str,help = 'all, CA_Market: [age, head_colour, head, body, body_type, leg, foot, gender, bags, body_colour, leg_colour, foot_colour]'
                                                          +'Market_attribute: [age, bags, leg_colour, body_colour, leg_type, leg ,sleeve hair, hat, gender]'
                                                           +  'Duke_attribute: [bags, boot, gender, hat, foot_colour, body, leg_colour,body_colour]',default='all')
    parser.add_argument('--sampler_max',type = int,help = 'maxmimum iteration of images, if 1 nothing would change',default = 1)
    parser.add_argument('--num_worst',type = int,help = 'to plot how many of the worst images in eval mode',default = 10)
    parser.add_argument('--lr',type = float,help = 'learning rate',default = 3.5e-5)
    parser.add_argument('--batch_size',type = int,help = 'training batch size',default = 32)
    parser.add_argument('--loss_weights',type = str,help = 'loss_weights if None without weighting None, effective',default='None')
    parser.add_argument('--baseline',type = str,help = 'it should be one the [osnet_x1_0, osnet_ain_x1_0, lu_person]',default='osnet_x1_0')
    parser.add_argument('--baseline_path',type = str,help = 'path of network weights [osnet_x1_0_market, osnet_ain_x1_0_msmt17, osnet_x1_0_msmt17,osnet_x1_0_duke_softmax]',default='./checkpoints/osnet_x1_0_market.pth')
    parser.add_argument('--trained_multi_branch',type = str,help = 'path of trained attr_net multi-branch network[./results/mb_conv3_all_bothwei_CA/mb_conv3_all_wei_CA/best_attr_net.pth]',default= None)
    parser.add_argument('--save_attr_metrcis',type = str,help = 'path to save attributes metrics',default='./results/mb_conv3_all_bothwei_CA/mb_conv3_all_wei_CA/attr_metrics.xlsx')
    args = parser.parse_args()
    return args


#%%

args = parse_args()

''' Delivering data as attr dictionaries '''
part_based = True if args.training_strategy == 'categorized' else False # if categotized, for each part one tesor will be genetrated
                                                                        # elif vectorize, only an attribute vector will be generated 

if args.dataset == 'CA_Market' or args.dataset == 'Market_attribute' or args.dataset == 'PA100k':
    main_path = args.main_path   
    path_attr = args.attr_path 
    attr = data_delivery(main_path,
                  path_attr=path_attr,
                  need_parts=part_based,
                  need_attr=not part_based,
                  dataset = args.dataset)
    
    for key , value in attr.items():
      try: print(key , 'size is: \t {}'.format((value.size())))
      except:
        print(key)
        
    if args.dataset == 'PA100k':
        train_idx = np.arange(int(0.8*len(attr['img_names'])))
        valid_idx = np.arange(int(0.8*len(attr['img_names'])),int(0.9*len(attr['img_names']))) 
        test_idx = np.arange(int(0.9*len(attr['img_names'])))
    else:
        train_idx = torch.load('./attributes/train_idx_full.pth')
        test_idx = torch.load('./attributes/test_idx_full.pth') 
        valid_idx = validation_idx(test_idx)

else:
    
    train_img_path = args.train_path
    test_img_path = args.test_path
    path_attr_train = args.attr_path_train
    path_attr_test = args.attr_path_test
    
    attr_train = data_delivery(train_img_path, path_attr=path_attr_train,
                               need_parts=part_based, need_attr=not part_based, dataset=args.dataset)
    attr_test = data_delivery(test_img_path, path_attr=path_attr_test,
                              need_parts=part_based, need_attr=not part_based, dataset=args.dataset)
    
    train_idx = np.arange(len(attr_train['img_names']))
    if args.dataset == 'CA_Duke':
        test_idx = np.arange(len(attr_test['attributes']))
    else:
        test_idx = np.arange(len(attr_test['img_names']))
    valid_idx = validation_idx(test_idx)
        
    print('\n', 'train-set specifications') 
    for key , value in attr_train.items():   
        try: print(key , 'size is: \t {}'.format((value.size())))
        except:
          print(key)
   
    print('\n', 'test-set specifications') 
    for key , value in attr_test.items():
        try: print(key , 'size is: \t {}'.format((value.size())))
        except:
          print(key)

#%%    
''' Delivering data as attr dictionaries '''

train_transform =  transforms.Compose([transforms.RandomRotation(degrees=10),
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(saturation=[0.8,1.25], brightness = (0.8, 1.2), contrast = (0.8, 1.2)),
                            transforms.RandomPerspective(distortion_scale=0.2, p = 0.8),
                            # LGT(probability=0.8, sl=0.02, sh=0.8, r1=0.8)
                            ])
#torchvision.transforms.RandomPerspective(distortion_scale, p)

if args.dataset == 'CA_Market' or args.dataset == 'Market_attribute' or args.dataset == 'PA100k':
        
    train_data = CA_Loader(img_path=main_path,
                              attr=attr,
                              resolution=(256,128),
                              transform=train_transform,
                              indexes=train_idx,
                              dataset = args.dataset,
                              need_attr = not part_based,
                              need_collection = part_based,
                              need_id = False,
                              two_transforms = True)
    
    test_data = CA_Loader(img_path=main_path,
                              attr=attr,
                              resolution=(256, 128),
                              indexes=test_idx,
                              dataset = args.dataset,
                              need_attr = not part_based,
                              need_collection = part_based,
                              need_id = False,
                              two_transforms = True) 

    valid_data = CA_Loader(img_path=main_path,
                              attr=attr,
                              resolution=(256, 128),
                              indexes=valid_idx,
                              dataset = args.dataset,
                              need_attr = not part_based,
                              need_collection = part_based,
                              need_id = False,
                              two_transforms = True)     


    if args.training_part == 'all':
        weights = attr_weight(attr=attr, effective=args.loss_weights, device=device, beta=0.99)
    else:
    
        weights = attr_weight(attr={key: attr[key] for key in args.training_part.split(',')},
        effective = args.loss_weights, device=device, beta=0.99)
        
        train_data.__dict__ = resampler(train_data.__dict__, args.training_part, args.sampler_max)
        
else:
    train_data = CA_Loader(img_path=train_img_path,
                              attr=attr_train,
                              resolution=(256,128),
                              transform=train_transform,
                              indexes=train_idx,
                              dataset = args.dataset,
                              need_attr = not part_based,
                              need_collection = part_based,
                              need_id = False,
                              two_transforms = False) 
    test_data = CA_Loader(img_path=test_img_path,
                              attr=attr_test,
                              resolution=(256, 128),
                              indexes=test_idx,
                              dataset = args.dataset,
                              need_attr = not part_based,
                              need_collection = part_based,
                              need_id = False,
                              two_transforms = True) 
    
    valid_data = CA_Loader(img_path=test_img_path,
                              attr=attr_test,
                              resolution=(256, 128),
                              indexes=valid_idx,
                              dataset = args.dataset,
                              need_attr = not part_based,
                              need_collection = part_based,
                              need_id = False,
                              two_transforms = True)  
    if args.training_part == 'all':      
        weights = attr_weight(attr=attr_train, effective=args.loss_weights, device=device, beta=0.99)
    else:
        weights = attr_weight(attr={key: attr[key] for key in args.training_part.split(',')},
        effective=args.loss_weights, device=device, beta=0.99)
        train_data.__dict__ = resampler(train_data.__dict__, args.training_part, args.sampler_max)
        
part_loss = part_data_delivery(weights, dataset = args.dataset, device = device)

batch_size = 32
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_data,batch_size=100,shuffle=False)
valid_loader = DataLoader(valid_data,batch_size=100,shuffle=False)

#%%
torch.cuda.empty_cache()

if args.baseline[:5] == 'osnet':
    
    from torchreid import models
    from torchreid import utils
    
    model = models.build_model(
        name=args.baseline,
        num_classes=751,
        loss='softmax',
        pretrained=False
    )
    
    utils.load_pretrained_weights(model, args.baseline_path)
else:
    raise Exception("The SI feature_extractor for lu_person is not ready")
    
### freezing the network
params = model.parameters()
for idx, param in enumerate(params):
    if idx <= 214: param.requires_grad = False
    
if part_based:
    attr_net = mb12_CA_build_model(
                      model,
                      main_cov_size = 384,
                      attr_dim = 64,
                      dropout_p = 0.3,
                      sep_conv_size = 64,
                      feature_selection = None)
else:
    if args.dataset == 'CA_Market' or args.dataset == 'Market_attribute' or args.dataset == 'PA100k':
        attr_dim = len(attr['names'])
    else:
        attr_dim = len(attr_train['names'])
    attr_net = attributes_model(model, feature_dim = 512, attr_dim = attr_dim)

if args.trained_multi_branch is not None:
    trained_net = torch.load(args.trained_multi_branch)
    attr_net.load_state_dict(trained_net)
else:
    print('\n', 'there is no trained branches', '\n')
    
attr_net = attr_net.to(device)
baseline_size = get_n_params(model)
mb_size = get_n_params(attr_net)
print('baseline has {} parameters'.format(baseline_size), 
      '\t', 'multi-branch net has {} parameters'.format(mb_size), '\n'
      'multi-branch is {:.2f} times biger than baseline'.format(mb_size/baseline_size))

#%%

params = attr_net.parameters()

lr = 3.5e-5
optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.99), eps=1e-08)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 10, 17], gamma=0.9)

#%%
save_path = './results/'
if args.mode == 'train':
    dict_training_multi_branch(num_epoch = 30,
                          attr_net = attr_net,
                          train_loader = train_loader,
                          test_loader = valid_loader,
                          optimizer = optimizer,
                          scheduler = scheduler,
                          save_path = save_path,  
                          part_loss = part_loss,
                          device = device,
                          version = 'check_net_with_CA_Market_alaki',
                          categorical = part_based,
                          resume=False,
                          loss_train = None,
                          loss_test=None,
                          train_attr_F1=None,
                          test_attr_F1=None,
                          train_attr_acc=None,
                          test_attr_acc=None,  
                          stoped_epoch=None)
    
#%%
if args.mode == 'eval':
    import pandas as pd
    
    attr_metrics = dict_evaluating_multi_branch(attr_net = attr_net,
                                   test_loader = test_loader,
                                   save_path = save_path,
                                   device = device,
                                   part_loss = part_loss,
                                   categorical = part_based,
                                   loss_train=None,
                                   loss_test=None,
                                   train_attr_F1=None,
                                   test_attr_F1=None,
                                   train_attr_acc=None,
                                   test_attr_acc=None,
                                   stoped_epoch=None)

    for metric in ['precision', 'recall', 'accuracy', 'f1', 'mean_accuracy']:
        if args.dataset == 'CA_Market' or args.dataset == 'Market_attribute' or args.dataset == 'PA100k':
            metrics_print(attr_metrics[0], attr['names'], metricss=metric)
        else:
            metrics_print(attr_metrics[0], attr_test['names'], metricss='precision')

    total_metrics(attr_metrics[0])
    iou_result = attr_metrics[1]
    print('\n','the mean of iou is: ',str(iou_result.mean().item()))
    if args.dataset == 'CA_Market' or args.dataset == 'Market_attribute' or args.dataset == 'PA100k':            
        iou_worst_plot(iou_result=iou_result, valid_idx=test_idx, main_path=main_path, attr=attr, num_worst = args.num_worst)
    else:    
        iou_worst_plot(iou_result=iou_result, valid_idx=test_idx, main_path=test_img_path, attr=attr, num_worst = args.num_worst)
 
    metrics_result = attr_metrics[0][:5]
    attr_metrics_pd = pd.DataFrame(data = np.array([result.numpy() for result in metrics_result]).T,
                                   index=attr['names'], columns=['precision','recall','accuracy','f1','MA'])
    attr_metrics_pd.to_excel(args.save_attr_metrcis)
    
    mean_metrics = attr_metrics[0][5:]
    mean_metrics.append(iou_result.mean().item())
    mean_metrics_pd = pd.DataFrame(data = np.array(mean_metrics), index=['precision','recall','accuracy','f1','MA', 'IOU'])
    peices = args.save_attr_metrcis.split('/')
    peices[-1] = 'mean_metrics.xlsx'
    path_mean_metrcis = '/'.join(peices)
    mean_metrics_pd.to_excel(path_mean_metrcis)
#%%
