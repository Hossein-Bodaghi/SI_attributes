
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:02:43 2020

@author: Hossein
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

def parse_args():
    parser = argparse.ArgumentParser(
        description ='identify the most similar clothes to the input image')
    parser.add_argument(
        '--input',
        type = str,
        help = 'input image path',
        default='C:/Users/ASUS/Desktop/Taarlab/Datasetes/Market-1501-v15.09.15/gt_bbox/0010_c6s4_002427_00.jpg')
    
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
        default = '/home/hossein/deep-person-reid/datasets/Market-1501-v15.09.15/bounding_box_test')
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default = '/home/hossein/anaconda3/envs/torchreid/deep-person-reid/my_osnet/result/V8_01/best_attr_net.pth',
        help='the checkpoint network to resume from')
    
    parser.add_argument(
        '--use_ready_features', action='store_true')
    
    parser.add_argument(
        '--need_attr', action='store_true')
    parser.set_defaults(use_ready_features=False, need_attr=True)
    
    parser.add_argument(
        '--all_features',
        type=str,
        default='C:/Users/ASUS/Desktop/demo/features/market_gallery_osnet.npy',
        help='features of whole data that we want to retrieve')
    
    parser.add_argument(
        '--save_features',
        type=str,
        default='C:/Users/ASUS/Desktop/demo/features',
        help='features of whole data that we want to retrieve')

    args = parser.parse_args()
    return args

#%%

transform_simple = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def tensor_thresh(tensor, thr=0.5):
    out = (tensor>thr).float()
    return out
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



