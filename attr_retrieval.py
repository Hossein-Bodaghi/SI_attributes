#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 12:51:17 2022

@author: hossein
"""
import os
import argparse
from models import CA_market_model2
from preprocess import get_image, attr_evaluation
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import skimage.io
import matplotlib.pyplot as plt
from torchreid import models
import torch
import torch.nn as nn 
import time
from torchvision import transforms

# import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print('calculation is on:',device)
torch.cuda.empty_cache()

#%%
# repository imports
from utils import get_n_params, part_data_delivery, resampler, LGT, attr_weight, validation_idx, iou_worst_plot, common_attr, metrics_print, total_metrics
from trainings import dict_training_multi_branch, take_out_multi_branch, dict_training_dynamic_loss
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
    parser.add_argument('--dataset', type = str, help = 'one of dataset = [CA_Market,Market_attribute,CA_Duke,Duke_attribute,PA100k,CA_Duke_Market]', default='CA_Market')
    parser.add_argument('--mode', type = str, help = 'mode of runner = [train, eval, fine_tune]', default='eval')
    parser.add_argument('--eval_mode', type = str, help = '[re_id, attr]', default='re_id')
    parser.add_argument('--training_strategy',type = str,help = 'categorized or vectorized',default='categorized')       
    parser.add_argument('--training_part',type = str,help = 'all, CA_Market: [age, head_colour, head, body, body_type, leg, foot, gender, bags, body_colour, leg_colour, foot_colour]'
                                                          +'Market_attribute: [age, bags, leg_colour, body_colour, leg_type, leg ,sleeve hair, hat, gender]'
                                                           +  'Duke_attribute: [bags, boot, gender, hat, foot_colour, body, leg_colour,body_colour]'
                                                           + 'CA_Duke:[gender,head,head_color,hat,cap_color,body,body_color,bags,face,leg,leg_color,foot,foot_color,accessories,position,race', default='all')
    parser.add_argument('--sampler_max',type = int,help = 'maxmimum iteration of images, if 1 nothing would change',default = 3)
    parser.add_argument('--num_worst',type = int,help = 'to plot how many of the worst images in eval mode',default = 10)
    parser.add_argument('--epoch',type = float,help = 'number epochs',default = 30)
    parser.add_argument('--lr',type = float,help = 'learning rate',default = 3.5e-5)
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
b_name = args.branch_place+'_' if args.training_strategy == 'categorized' else ''
version = args.dataset+'_'+b_name+args.training_strategy[:3]+'_'+re.search('/checkpoints/(.*).pth', args.baseline_path).group(1)
print('*** The Version is: ', version, '\n')
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

    # for idx, param in enumerate(attr_net.branch_age.parameters()):
    #     if param.requires_grad == False:
    #         print('1')

else:
    attr_dim = len(attr['names'] if M_or_M_attr_or_PA else attr_train['names'])

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

def main():
    
    attr_colomns = ['gender','cap','hairless','short hair','long hair',
           'knot', 'T-shirt','coat',
           'top','simple/patterned','b_white','b_red',
           'b_orange','b_yellow','b_green','b_blue',
           'b_gray','b_pink','b_black','backpack',
           'hand bag','no bag','pants',
           'shorts','skirt','l_white','l_red','l_orange','l_yellow','l_green','l_blue',
           'l_gray','l_pink','l_black','shoes','sandal',
           'hidden','f_white','f_red','f_orange','f_yellow','f_green','f_blue',
           'f_gray','f_pink','f_black']
    
    args = parse_args()
    
    # loading the model
    if args.model == 'osnet':        
        
        model = models.build_model(
            name='osnet_x1_0',
            num_classes=751,
            loss='softmax',
            pretrained=False
        )
        
        attr_net_camarket = CA_market_model2(model=model,
                          feature_dim = 512,
                          num_id = 751,
                          attr_dim = 46,
                          need_id = False,
                          need_attr = True,
                          need_collection = False)
        
        model_path = args.checkpoint
        trained_net = torch.load(model_path)
        attr_net_camarket.load_state_dict(trained_net.state_dict())
        
        attr_net_camarket = attr_net_camarket.to(device)
        resolution = (256,128)
        # del model
        
    
    keepgoing = 'y'
    while keepgoing == 'y':
        keepgoing = input('do you want to continue?[y/n] ')
        if keepgoing == 'n':
            break
        args = parse_args()        
        #'C:/Users/ASUS/Desktop/Taarlab/Datasetes/Market-1501-v15.09.15/gt_bbox/0010_c6s4_002427_00.jpg'
        ar = input('put your photo here: ')
        args.input = ar[1:-2]
        img = get_image(args.input , resolution[0], resolution[1])
        img = transform_simple(img)
        img = img.to(device)
        attr_net_camarket.eval()
        attr_net_camarket = attr_net_camarket.to(device)
        pred_array = attr_net_camarket.get_feature(torch.unsqueeze(img, dim=0), get_attr=True, get_feature=False)    
        pred_array = torch.squeeze(torch.sigmoid(pred_array['attr']))
        pred_array = pred_array.to('cpu')

        #print('The results are:'+'\n')
        #for idx, m in enumerate(attr_colomns):
        #    print(idx, ')', m, '-->', pred_array[idx].item()) 
        

        fig , ax = plt.subplots(nrows=1,ncols=2)
           
        img = skimage.io.imread(args.input)
          #Show images
        #fig , ax = plt.subplots(nrows=1,ncols=2)
          #Show images
        ax.ravel()[0].imshow(img)
        ax.ravel()[0].set_axis_off()
        
        gender = tensor_thresh(pred_array[0], 0.5)
        hair = torch.argmax(pred_array[1:6])
        body = torch.argmax(pred_array[6:10])
        body_color = torch.argmax(pred_array[10:19])
        bags = torch.argmax(pred_array[19:22])
        leg = torch.argmax(pred_array[22:25])
        leg_color = torch.argmax(pred_array[25:34])

        foot = torch.argmax(pred_array[34:37])
        foot_color = torch.argmax(pred_array[37:])

        ax.ravel()[1].text(0, 0.9, 'Attributes:', style='italic', fontsize='xx-large', fontweight='bold')

        if gender == 0: 
            ax.ravel()[1].text(0, 0.8, 'Male', style='italic', fontsize='xx-large')
        else: 
            ax.ravel()[1].text(0, 0.8, 'Female', style='italic', fontsize='xx-large')

        ax.ravel()[1].text(0, 0.7, attr_colomns[hair+1], style='italic', fontsize='xx-large')
                
        ax.ravel()[1].text(0, 0.6, attr_colomns[body_color+10].split("_")[1]+' '+attr_colomns[body+6], style='italic', fontsize='xx-large')
                        
        ax.ravel()[1].text(0, 0.5, attr_colomns[bags+19], style='italic', fontsize='xx-large')

        ax.ravel()[1].text(0, 0.4, attr_colomns[leg_color+25].split("_")[1]+' '+attr_colomns[leg+22], style='italic', fontsize='xx-large')
        
        if attr_colomns[foot+34] == 'sandal':
            ax.ravel()[1].text(0, 0.3, attr_colomns[foot+34], style='italic', fontsize='xx-large')
        else:
            ax.ravel()[1].text(0, 0.3, attr_colomns[foot_color+37].split("_")[1]+' '+attr_colomns[foot+34], style='italic', fontsize='xx-large')


        ax.ravel()[1].set_axis_off()
        plt.show()    

if __name__ == '__main__':
      main()
#args = parse_args()
#feat_model = load_model(args.checkpoint,compile = False)
#feat_model = Model(inputs=feat_model.input, outputs=feat_model.layers[-3].output)



