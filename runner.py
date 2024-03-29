#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 20:07:29 2022

@author: hossein
"""
#%%
# repository imports
from utils import get_n_params, part_data_delivery, resampler, LGT, attr_weight, validation_idx, iou_worst_plot, common_attr, metrics_print, total_metrics, map_evaluation, resume_handler
from trainings import dict_training_multi_branch, take_out_multi_branch, dict_training_dynamic_loss, take_out_attr_retrieval
from models import attributes_model, Loss_weighting, mb_CA_auto_same_depth_build_model
from delivery import data_delivery, reid_delivery
from metrics import tensor_metrics, IOU
from loaders import CA_Loader, Simple_Loader
from rankings import rank_calculator
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


#%%
def parse_args():
    parser = argparse.ArgumentParser(description ='identify the most similar clothes to the input image')
    parser.add_argument('--dataset', type = str, help = 'one of dataset = [CA_Market,Market_attribute,CA_Duke,Duke_attribute,PA100k,CA_Duke_Market]', default='CA_Duke_Market')
    parser.add_argument('--mode', type = str, help = 'mode of runner = [train, eval]', default='eval')
    parser.add_argument('--eval_mode', type = str, help = '[re_id, attr, attr_retrieval]', default='attr_retrieval')
    parser.add_argument('--training_strategy',type = str,help = 'categorized or vectorized',default='vectorized')       
    parser.add_argument('--training_part',type = str,help = 'all, CA_Market: [age, head_colour, head, body, body_type, leg, foot, gender, bags, body_colour, leg_colour, foot_colour]'
                                                          +'Market_attribute: [age, bags, leg_colour, body_colour, leg_type, leg ,sleeve hair, hat, gender]'
                                                           +  'Duke_attribute: [bags, boot, gender, hat, foot_colour, body, leg_colour,body_colour]'
                                                           + 'CA_Duke:[gender,head,head_color,hat,cap_color,body,body_color,bags,face,leg,leg_color,foot,foot_color,accessories,position,race', default='all')
    parser.add_argument('--sampler_max',type = int,help = 'maxmimum iteration of images, if 1 nothing would change',default = 3)
    parser.add_argument('--num_worst',type = int,help = 'to plot how many of the worst images in eval mode',default = 10)
    parser.add_argument('--epoch',type = float,help = 'number epochs',default = 50)
    parser.add_argument('--lr',type = float,help = 'learning rate',default = 3.5e-5)
    parser.add_argument('--batch_size',type = int,help = 'training batch size',default = 32)
    parser.add_argument('--loss_weights',type = str,help = 'loss_weights if None without weighting [None,effective,dynamic]',default='None')
    parser.add_argument('--baseline',type = str,help = 'it should be one the [osnet_x1_0, osnet_ain_x1_0, lu_person]',default='osnet_x1_0')
    parser.add_argument('--baseline_path',type = str,help = 'path of network weights [osnet_x1_0_market, osnet_ain_x1_0_msmt17, osnet_x1_0_msmt17,osnet_x1_0_duke_softmax]',default='./checkpoints/osnet_x1_0_msmt17.pth')
    parser.add_argument('--branch_place',type = str,help = 'could be: conv1,maxpool,conv2,conv3,conv4,conv5',default='conv3')
    parser.add_argument('--cross_domain',type = str,help = 'y/n',default='y')
    parser.add_argument('--branch',type = str,help = 'y/n',default='n')
    parser.add_argument('--resume',type = str,help = 'y/n',default='y')
    args = parser.parse_args()
    return args


#%%

args = parse_args()
b_name = args.branch_place+'_' if args.training_strategy == 'categorized' else ''
branch = True if args.branch == 'y' else False
resume = True if args.resume == 'y' else False
vec_nam = '_'+args.branch_place if args.branch == 'y' else ''
version = args.dataset+'_'+b_name+args.training_strategy[:3]+vec_nam+'_'+re.search('/checkpoints/(.*).pth', args.baseline_path).group(1)
print('*** The Version is: ', version, '\n')
'''v = 0
while os.path.isdir('results/'+version):'''



save_attr_metrcis = './results/'+version+'/attr_metrics.xlsx'

if os.path.exists('./results/'+version+'/best_attr_net.pth'):
    # if resume:
    #     trained_multi_branch = os.path.join('./results', version, 'attr_net.pth')
    # else:
        trained_multi_branch = './results/'+version+'/best_attr_net.pth'
else:
    trained_multi_branch = None

if args.dataset == 'CA_Market':
    main_path = './datasets/Market1501/Market-1501-v15.09.15/gt_bbox/'
    path_attr = './attributes/CA_Market_with_id.npy'
    path_query = './datasets/Market1501/Market-1501-v15.09.15/query/'
    path_attr_test = './datasets/Market1501/Market-1501-v15.09.15/bounding_box_test'

elif args.dataset == 'Market_attribute':
    main_path = './datasets/Market1501/Market-1501-v15.09.15/gt_bbox/'
    path_attr = './attributes/Market_attribute_with_id.npy'
    path_query = './datasets/Market1501/Market-1501-v15.09.15/query/'
    path_attr_test = './datasets/Market1501/Market-1501-v15.09.15/bounding_box_test'

elif args.dataset == 'PA100k':
    main_path = './datasets/PA-100K/release_data/release_data/'
    path_attr = './attributes/PA100k_all_with_id.npy'

elif args.dataset == 'CA_Duke':
    train_img_path = './datasets/Dukemtmc/bounding_box_train'
    test_img_path = './datasets/Dukemtmc/bounding_box_test'
    path_attr_train = './attributes/CA_Duke_train_with_id.npy'
    path_attr_test = './attributes/CA_Duke_test_with_id.npy'
    path_query = './datasets/Dukemtmc/query/'

elif args.dataset == 'Duke_attribute':
    train_img_path = './datasets/Dukemtmc/bounding_box_train'
    test_img_path = './datasets/Dukemtmc/bounding_box_test'
    path_attr_train = './attributes/Duke_attribute_train_with_id.npy'
    path_attr_test = './attributes/Duke_attribute_test_with_id.npy'

elif args.dataset == 'CA_Duke_Market':
    train_img_path = './datasets/CA_Duke_Market/bbox_train'
    test_img_path = './datasets/CA_Duke_Market/bbox_test'
    path_attr_train = './attributes/CA_Duke_Market_train_with_id.npy'
    path_attr_test = './attributes/CA_Duke_Market_test_with_id.npy'


''' Delivering data as attr dictionaries '''
part_based = True if args.training_strategy == 'categorized' else False # if categotized, for each part one tesor will be genetrated
cross_domain = True if args.cross_domain == 'y' else False # if categotized, for each part one tesor will be genetrated
                                                                        # elif vectorize, only an attribute vector will be generated 

M_or_M_attr_or_PA = args.dataset == 'CA_Market' or args.dataset == 'Market_attribute' or args.dataset == 'PA100k'

if M_or_M_attr_or_PA:
    attr = data_delivery(main_path,
                  path_attr=path_attr,
                  need_parts=part_based,
                  need_attr=not part_based,
                  dataset = args.dataset)
    
    attr_train = attr
    for key , value in attr.items():
      try: print(key , 'size is: \t\t {}'.format((value.size())))
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
    attr_train = data_delivery(train_img_path, path_attr=path_attr_train,
                               need_parts=part_based, need_attr=not part_based, dataset=args.dataset)


    attr_test = data_delivery(test_img_path, path_attr=path_attr_test,
                              need_parts=part_based, need_attr=not part_based, dataset=args.dataset)
    
    train_idx = np.arange(len(attr_train['img_names']))
    if args.dataset == 'CA_Duke':
        test_idx = np.arange(len(attr_test['img_names']))
        valid_idx = test_idx
    else:
        test_idx = np.arange(len(attr_test['img_names']))
        valid_idx = validation_idx(test_idx)
        
    print('\n', 'train-set specifications') 
    for key , value in attr_train.items():   
        try: print(key , 'size is: \t\t {}'.format((value.size())))
        except:
          print(key)
   
    print('\n', 'test-set specifications') 
    for key , value in attr_test.items():
        try: print(key , 'size is: \t\t {}'.format((value.size())))
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


train_data = CA_Loader(img_path=main_path if M_or_M_attr_or_PA else train_img_path,
                            attr=attr if M_or_M_attr_or_PA else attr_train,
                            resolution=(256,128),
                            transform=train_transform,
                            indexes=train_idx,
                            dataset = args.dataset,
                            need_attr = not part_based,
                            need_collection = part_based,
                            need_id = False,
                            two_transforms = True) 

test_data = CA_Loader(img_path=main_path if M_or_M_attr_or_PA else test_img_path,
                            attr=attr if M_or_M_attr_or_PA else attr_test,
                            resolution=(256, 128),
                            indexes=test_idx,
                            dataset = args.dataset,
                            need_attr = not part_based,
                            need_collection = part_based,
                            need_id = True,
                            two_transforms = False) 

valid_data = CA_Loader(img_path=main_path if M_or_M_attr_or_PA else test_img_path,
                            attr=attr if M_or_M_attr_or_PA else attr_test,
                            resolution=(256, 128),
                            indexes=valid_idx,
                            dataset = args.dataset,
                            need_attr = not part_based,
                            need_collection = part_based,
                            need_id = False,
                            two_transforms = False)  

if args.training_part == 'all':      
    weights = attr_weight(attr=attr if M_or_M_attr_or_PA else attr_train, effective=args.loss_weights, device=device, beta=0.99)
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
elif args.baseline == 'lu_person':

    from fastreid.engine import DefaultTrainer, default_argument_parser, launch, default_setup
    from fastreid.config import get_cfg

    def setup(args):
        """
        Create configs and perform basic setups.
        """
        cfg = get_cfg()
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        cfg.freeze()
        default_setup(cfg, args)
        return cfg

    args = default_argument_parser().parse_args()

    args.config_file = './configs/CMDM/mgn_R50_moco.yml'
    args.eval_only = True
    args.opts = ['DATASETS.ROOT', 'datasets', 'DATASETS.KWARGS', 'data_name:duke', 
    'MODEL.WEIGHTS', 'C:/Users/ASUS/Downloads/duke.pth', 'MODEL.DEVICE', 'cuda:0', 
    'OUTPUT_DIR', 'logs/lup_moco/test/duke']
    print("Command Line Args:", args)
    
    cfg = launch(setup,
            args.num_gpus,
            num_machines=args.num_machines, 
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,),
        )

    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        si_calculate = False
        
    model = DefaultTrainer.build_model(cfg)
    
if args.loss_weights == 'dynamic':
    weight_nets = {}
    for key in weights:
        weight_nets.update({key:Loss_weighting(weights_dim=len(weights[key]))})

### freezing the network
if args.training_strategy == 'categorized' or branch:
    params = model.parameters()
    for idx, param in enumerate(params):
        param.requires_grad = False
        #if idx <= 214: param.requires_grad = False

branch_attrs_dims = {k: v.shape[1] for k, v in attr_train.items() if k not in ['id','cam_id','img_names','names']}

if part_based:
    attr_net = mb_CA_auto_same_depth_build_model(
                      model,
                      branch_place = args.branch_place,
                      dropout_p = 0.3,
                      branch_names=branch_attrs_dims,
                      feature_selection = None)

    # for idx, param in enumerate(attr_net.branch_age.parameters()):
    #     if param.requires_grad == False:
    #         print('1')

else:
    attr_dim = len(attr['names'] if M_or_M_attr_or_PA else attr_train['names'])
    if branch:
        attr_net = attributes_model(model, feature_dim = 512, 
                                    attr_dim = sum(branch_attrs_dims.values()) if cross_domain else attr_dim, branch_place=args.branch_place)
    else:
        attr_net = attributes_model(model, feature_dim = 512, attr_dim = sum(branch_attrs_dims.values()) if cross_domain else attr_dim)

if trained_multi_branch is not None:
    trained_net = torch.load(trained_multi_branch)
    attr_net.load_state_dict(trained_net)
else:
    print('\nthere is no trained branches', '\n')
    
attr_net = attr_net.to(device)
baseline_size = get_n_params(model)
mb_size = get_n_params(attr_net)
print('baseline has {} parameters'.format(baseline_size),'\n'
      'multi-branch net has {} parameters'.format(mb_size),'\n'
      'multi-branch is {:.2f} times bigger than baseline\n'.format(mb_size/baseline_size))
#%%

params = attr_net.parameters()

optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.99), eps=1e-08)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 10, 17], gamma=0.9)

#%%
save_path = './results/'
if args.mode == 'train':
    if args.loss_weights == 'dynamic':
        dict_training_dynamic_loss(num_epoch = args.epoch,
                              attr_net = attr_net,
                              dataset = args.dataset,
                              weight_nets = weight_nets,
                              weights = weights,
                              train_loader = train_loader,
                              test_loader = test_loader,
                              optimizer = optimizer,
                              scheduler = scheduler,
                              save_path = save_path,  
                              part_loss = part_loss,
                              device = device,
                              version = version,
                              categorical = part_based,
                              resume_dict = resume_handler(resume, version),
                              resume=False)
    else:
        dict_training_multi_branch(num_epoch = args.epoch,
                              attr_net = attr_net,
                              train_loader = train_loader,
                              test_loader = valid_loader,
                              optimizer = optimizer,
                              scheduler = scheduler,
                              save_path = save_path,  
                              part_loss = part_loss,
                              device = device,
                              version = version,
                              categorical = part_based,
                              resume_dict = resume_handler(resume, version),
                              resume=resume)
#%%
if args.mode == 'eval':
    import pandas as pd

    if cross_domain:
        predicts, targets = take_out_multi_branch(attr_net = attr_net,
                                       test_loader = test_loader,
                                       save_path = save_path,
                                       device = device,
                                       part_loss = part_loss,
                                       categorical = part_based) 

        attr_metrics = []
        predicts, targets, attr_names = common_attr(predicts, targets)
        attr_metrics.append(tensor_metrics(targets, predicts))
        attr_metrics.append(IOU(predicts, targets))
                
    else:
        
        if args.eval_mode == 're_id':
            query = reid_delivery(path_query)        
            query_data = Simple_Loader(img_path=path_query,
                                attr=query,
                                resolution=(256, 128))  

            query_loader = DataLoader(query_data,batch_size=100,shuffle=False)
        
            if args.dataset in ['CA_Market', 'Market_attribute']:
                
                gallery = reid_delivery(path_attr_test)                
                gallery_data = Simple_Loader(img_path=path_attr_test,
                                             attr=gallery,
                                             resolution=(256, 128)) 
                
                gallery_loader = DataLoader(gallery_data,batch_size=100,shuffle=False)
                
                cmc, mAP = rank_calculator(attr_net = attr_net,
                                           gallery_loader = gallery_loader,
                                           query_loader = query_loader,
                                           gallery = gallery, query = query,
                                           device = device, ratio = 0.1, activation=False)                
            else:
                cmc, mAP = rank_calculator(attr_net = attr_net,
                                           gallery_loader = test_loader,
                                           query_loader = query_loader,
                                           gallery = attr_test, query = query,
                                           device = device, ratio = 0, activation=False)
            print('** Re-Id Results **','\n','mAP: {:.2f}%'.format(100*mAP),
                  '\n', 'CMC curve','\n','Rank-1: {:.2f}%'.format(100*cmc[0]),'\n',
                  'Rank-5: {:.2f}%'.format(100*cmc[4]),'\n','Rank-10: {:.2f}%'.format(100*cmc[9]),
                  '\n','Rank-20: {:.2f}%'.format(100*cmc[19]))
        
        elif args.eval_mode == 'attr':
            predicts, targets = take_out_multi_branch(attr_net = attr_net,
                                           test_loader = test_loader,
                                           device = device,
                                           part_loss = part_loss,
                                           categorical = part_based)             
            attr_metrics = []
            attr_metrics.append(tensor_metrics(targets, predicts))
            attr_metrics.append(IOU(predicts, targets))
            if M_or_M_attr_or_PA: 
                if cross_domain:
                    pass
                else:
                    attr_names = attr['names']
                main_path = main_path
            else:
                if cross_domain:
                    pass
                else:
                    attr_names = attr_test['names']
                main_path = test_img_path
                attr = attr_test
            for metric in ['precision', 'recall', 'accuracy', 'f1', 'mean_accuracy']:
                metrics_print(attr_metrics[0], attr_names, metricss=metric)
        
            total_metrics(attr_metrics[0])
            iou_result = attr_metrics[1]
            print('\n','the mean of iou is: ',str(iou_result.mean().item()))
        
            iou_worst_plot(iou_result=iou_result, valid_idx=test_idx, main_path=main_path, attr=attr, num_worst = args.num_worst)
            
        
        
            metrics_result = attr_metrics[0][:5]
        
            attr_metrics_pd = pd.DataFrame(data = np.array([result.numpy() for result in metrics_result]).T,
                                           index=attr_names, columns=['precision','recall','accuracy','f1','MA'])  
            attr_metrics_pd.to_excel(save_attr_metrcis)
            
            mean_metrics = attr_metrics[0][5:]
            mean_metrics.append(iou_result.mean().item())
            mean_metrics_pd = pd.DataFrame(data = np.array(mean_metrics), index=['precision','recall','accuracy','f1','MA', 'IOU'])
            peices = save_attr_metrcis.split('/')
            peices[-1] = 'mean_metrics.xlsx'
            path_mean_metrcis = '/'.join(peices)
            mean_metrics_pd.to_excel(path_mean_metrcis)
        elif args.eval_mode == 'attr_retrieval':
            predicts, probability, targets = take_out_attr_retrieval(attr_net = attr_net,
                                           test_loader = test_loader,
                                           device = device,
                                           part_loss = part_loss,
                                           categorical = part_based) 
            names = attr['names'] if M_or_M_attr_or_PA else attr_test['names']
            average_precision, mean_average_precision = map_evaluation(names, probability, targets)            
            average_precision_pd = pd.DataFrame(data = average_precision,
                                           index=names, columns=['average precision'])
            peices = save_attr_metrcis.split('/')
            peices[-1] = 'attr_retrieval_average_precision.xlsx'
            path_average_precision = '/'.join(peices)
            average_precision_pd.to_excel(path_average_precision)

