#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 20:07:29 2022

@author: hossein
"""
#%%
# repository imports
from utils import get_n_params, part_data_delivery, resampler, attr_weight, validation_idx, LGT, iou_worst_plot, common_attr
from trainings import dict_training_multi_branch, dict_evaluating_multi_branch, take_out_multi_branch, dict_training_dynamic_loss
from models import attributes_model, Loss_weighting, mb_CA_auto_build_model, mb_CA_auto_same_depth_build_model
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


#%%
def parse_args():
    parser = argparse.ArgumentParser(description ='identify the most similar clothes to the input image')
    parser.add_argument('--dataset', type = str, help = 'one of dataset = [CA_Market,Market_attribute,CA_Duke,Duke_attribute,PA100k,CA_Duke_Market]', default='CA_Market')
    parser.add_argument('--mode', type = str, help = 'mode of runner = [train, eval]', default='train')
    parser.add_argument('--training_strategy',type = str,help = 'categorized or vectorized',default='categorized')       
    parser.add_argument('--training_part',type = str,help = 'all, CA_Market: [age, head_colour, head, body, body_type, leg, foot, gender, bags, body_colour, leg_colour, foot_colour]'
                                                          +'Market_attribute: [age, bags, leg_colour, body_colour, leg_type, leg ,sleeve hair, hat, gender]'
                                                           +  'Duke_attribute: [bags, boot, gender, hat, foot_colour, body, leg_colour,body_colour]',default='all')
    parser.add_argument('--sampler_max',type = int,help = 'maxmimum iteration of images, if 1 nothing would change',default = 1)
    parser.add_argument('--num_worst',type = int,help = 'to plot how many of the worst images in eval mode',default = 10)
    parser.add_argument('--epoch',type = float,help = 'number epochs',default = 30)
    parser.add_argument('--lr',type = float,help = 'learning rate',default = 3.5e-4)
    parser.add_argument('--batch_size',type = int,help = 'training batch size',default = 32)
    parser.add_argument('--loss_weights',type = str,help = 'loss_weights if None without weighting [None,effective,dynamic]',default='None')
    parser.add_argument('--baseline',type = str,help = 'it should be one the [osnet_x1_0, osnet_ain_x1_0, lu_person]',default='osnet_x1_0')
    parser.add_argument('--baseline_path',type = str,help = 'path of network weights [osnet_x1_0_market, osnet_ain_x1_0_msmt17, osnet_x1_0_msmt17,osnet_x1_0_duke_softmax]',default='./checkpoints/osnet_x1_0_market.pth')
    parser.add_argument('--branch_place',type = str,help = 'could be: conv1,maxpool,conv2,conv3,conv4,conv5',default='conv5')
    parser.add_argument('--cross_domain',type = str,help = 'y/n',default='n')
    args = parser.parse_args()
    return args


#%%

args = parse_args()

version = args.dataset+'_'+args.branch_place+'_'+args.training_strategy[:3]+'_'+re.search('/checkpoints/(.*).pth', args.baseline_path).group(1)
'''v = 0
while os.path.isdir('results/'+version):'''



save_attr_metrcis = './results/'+version+'/attr_metrics.xlsx'

if os.path.exists('./results/'+version+'/best_attr_net.pth'):
    trained_multi_branch = './results/'+version+'/best_attr_net.pth'
else:
    trained_multi_branch = None

if args.dataset == 'CA_Market':
    main_path = './datasets/Market1501/Market-1501-v15.09.15/gt_bbox/'
    path_attr = './attributes/CA_Market_with_id.npy'
    path_query = './datasets/Market1501/Market-1501-v15.09.15/query/'

elif args.dataset == 'Market_attribute':
    main_path = './datasets/Market1501/Market-1501-v15.09.15/gt_bbox/'
    path_attr = './attributes/Market_attribute_with_id.npy'

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


    '''train_img_path = './datasets/Dukemtmc/bounding_box_train'
    test_img_path = './datasets/Dukemtmc/bounding_box_test'
    path_attr_train = './attributes/CA_Duke_train_with_id.npy'
    path_attr_test = './attributes/CA_Duke_test_with_id.npy'
    path_query = './datasets/Dukemtmc/query/'
    attr_train_ca = data_delivery(train_img_path, path_attr=path_attr_train,
                               need_parts=part_based, need_attr=not part_based, dataset='CA_Duke')
    a = 0
    for i in range(attr_train['gender'].shape[0]):
        if attr_train_ca['gender'][i] != attr_train['gender'][i]:
            a += 1'''


    attr_test = data_delivery(test_img_path, path_attr=path_attr_test,
                              need_parts=part_based, need_attr=not part_based, dataset=args.dataset)
    
    train_idx = np.arange(len(attr_train['img_names']))
    if args.dataset == 'CA_Duke':
        test_idx = np.arange(len(attr_test['gender']))
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

#torchvision.transforms.RandomPerspective(distortion_scale, p)

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
if args.training_strategy == 'categorized':
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

    for idx, param in enumerate(attr_net.branch_age.parameters()):
        if param.requires_grad == False:
            print('1')
            
    '''attr_net = mb_CA_auto_build_model(
                      model,
                      main_cov_size = 384,
                      attr_dim = 64,
                      dropout_p = 0.3,
                      sep_conv_size = 64,
                      branch_names=branch_attrs_dims,
                      feature_selection = None)'''
    
else:
    attr_dim = len(attr['names'] if M_or_M_attr_or_PA else attr_train['names'])

    attr_net = attributes_model(model, feature_dim = 512, attr_dim = sum(branch_attrs_dims.values()) if cross_domain else attr_dim)

if trained_multi_branch is not None:
    trained_net = torch.load(trained_multi_branch)
    attr_net.load_state_dict(trained_net)
else:
    print('\nthere is no trained branches', '\n')
    

"""check freezing
modell = models.build_model(
    name=args.baseline,
    num_classes=751,
    loss='softmax',
    pretrained=False
)
utils.load_pretrained_weights(modell, args.baseline_path)
identical = True
for p1, p2 in zip(attr_net.model.parameters(), modell.parameters()):
    if p1.data.ne(p2.data).sum() > 0:
        identical = False
        break
"""
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
                              resume=False,
                              loss_train = None,
                              loss_test=None,
                              train_attr_F1=None,
                              test_attr_F1=None,
                              train_attr_acc=None,
                              test_attr_acc=None,  
                              stoped_epoch=None)
    else:
        dict_training_multi_branch(num_epoch = args.epoch,
                              attr_net = attr_net,
                              train_loader = train_loader,
                              test_loader = test_loader,
                              optimizer = optimizer,
                              scheduler = scheduler,
                              save_path = save_path,  
                              part_loss = part_loss,
                              device = device,
                              version = version,
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

        query = data_delivery(path_query,
                    path_attr=path_attr,
                    need_parts=part_based,
                    need_id=True,
                    need_attr=not part_based,
                    dataset = args.dataset)

        query_idx = np.arange(len(query['img_names']))
        
        query_data = CA_Loader(img_path=path_query,
                            attr=query,
                            resolution=(256, 128),
                            indexes=query_idx,
                            dataset = args.dataset,
                            need_attr = not part_based,
                            need_collection = part_based,
                            need_id = True,
                            two_transforms = False)  

        query_loader = DataLoader(query_data,batch_size=100,shuffle=False)

        attr_metrics, dist_matrix = dict_evaluating_multi_branch(attr_net = attr_net,
                                                    test_loader = test_loader,
                                                    query_loader=query_loader,
                                                    save_path = save_path,
                                                    device = device,
                                                    part_loss = part_loss,
                                                    categorical = part_based)

        from evaluation import cmc_map_fromdist
        concat_ranks = cmc_map_fromdist(query_data, test_data, dist_matrix, feature_mode='concat', max_rank=10)

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
#%%

# if args.mode == 'eval':
#     import pandas as pd
    
#     attr_metrics = dict_evaluating_multi_branch(attr_net = attr_net,
#                                    test_loader = test_loader,
#                                    save_path = save_path,
#                                    device = device,
#                                    part_loss = part_loss,
#                                    categorical = part_based,
#                                    loss_train=None,
#                                    loss_test=None,
#                                    train_attr_F1=None,
#                                    test_attr_F1=None,
#                                    train_attr_acc=None,
#                                    test_attr_acc=None,
#                                    stoped_epoch=None)

#     for metric in ['precision', 'recall', 'accuracy', 'f1', 'mean_accuracy']:
#         if args.dataset == 'CA_Market' or args.dataset == 'Market_attribute' or args.dataset == 'PA100k':
#             metrics_print(attr_metrics[0], attr['names'], metricss=metric)
#         else:
#             metrics_print(attr_metrics[0], attr_test['names'], metricss='precision')

#     total_metrics(attr_metrics[0])
#     iou_result = attr_metrics[1]
#     print('\n','the mean of iou is: ',str(iou_result.mean().item()))
#     if args.dataset == 'CA_Market' or args.dataset == 'Market_attribute' or args.dataset == 'PA100k':            
#         iou_worst_plot(iou_result=iou_result, valid_idx=test_idx, main_path=main_path, attr=attr, num_worst = args.num_worst)
#     else:    
#         iou_worst_plot(iou_result=iou_result, valid_idx=test_idx, main_path=test_img_path, attr=attr, num_worst = args.num_worst)
 
#     metrics_result = attr_metrics[0][:5]
#     attr_metrics_pd = pd.DataFrame(data = np.array([result.numpy() for result in metrics_result]).T,
#                                    index=attr['names'], columns=['precision','recall','accuracy','f1','MA'])
#     attr_metrics_pd.to_excel(args.save_attr_metrcis)
    
#     mean_metrics = attr_metrics[0][5:]
#     mean_metrics.append(iou_result.mean().item())
#     mean_metrics_pd = pd.DataFrame(data = np.array(mean_metrics), index=['precision','recall','accuracy','f1','MA', 'IOU'])
#     peices = args.save_attr_metrcis.split('/')
#     peices[-1] = 'mean_metrics.xlsx'
#     path_mean_metrcis = '/'.join(peices)
#     mean_metrics_pd.to_excel(path_mean_metrcis)

