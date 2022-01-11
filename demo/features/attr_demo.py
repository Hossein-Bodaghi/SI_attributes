
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
        default='/home/dr/Downloads/Iran Test/Iran Fashion Dataset 990 Resized/002.jpg')
    
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
        default='/home/hossein/Downloads/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth',
        help='the checkpoint network to resume from')
    
    parser.add_argument(
        '--use_ready_features', action='store_true')
    
    parser.add_argument(
        '--need_attr', action='store_true')
    parser.set_defaults(use_ready_features=False, need_attr=True)
    
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

transform_simple = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def tensor_thresh(tensor, thr=0.5):
    out = (tensor>thr).float()
    return out
#%%

def main():
    
    attr_colomns = ['gender','cap','hairless','short hair','long hair',
           'knot', 'Tshirt/shirt','coat',
           'top','simple/patterned','b_w','b_r',
           'b_o','b_y','b_green','b_b',
           'b_gray','b_p','b_black','backpack',
           'hand bag','no bag','pants',
           'short','skirt','l_w','l_r','l_o','l_y','l_green','l_b',
           'l_gray','l_p','l_black','shoes','sandal',
           'hidden','f_w','f_r','f_o','f_y','f_green','f_b',
           'f_gray','f_p','f_black']
    
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
        
        model_path = '/home/hossein/anaconda3/envs/torchreid/deep-person-reid/my_osnet/result/V8_01/best_attr_net.pth'
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
        
        gender = tensor_thresh(pred_array[0], 0.5)


        print('The results are:'+'\n')

        for idx, m in enumerate(attr_colomns):

            print(idx, ')', m, '-->', pred_array[idx].item()) 
        

        fig , ax = plt.subplots(nrows=1,ncols=2)
           
        img = skimage.io.imread(args.input)
          #Show images
        fig , ax = plt.subplots(nrows=1,ncols=2)
          #Show images
        ax.ravel()[0].imshow(img)
        ax.ravel()[0].set_axis_off()
        
        if gender == 0: 
            ax.ravel()[1].text(0, 0.8, 'Male', style='italic', fontsize='xx-large')
        else: 
            ax.ravel()[1].text(0, 0.8, 'Female', style='italic', fontsize='xx-large')
        ax.ravel()[1].text(1, 0.8, 'Handbag', style='italic', fontsize='xx-large')
        
        ax.ravel()[1].text(0.4, 0.7, 'Knot Hair', style='italic', fontsize='xx-large')
        
        ax.ravel()[1].text(0.3, 0.55, 'White Shirt', style='italic', fontsize='xx-large')
        ax.ravel()[1].text(0.45, 0.59, 'Simple', style='italic', fontsize='xx-large')
        # ax.ravel()[1].text(0.4, 0.5, 'leg', style='italic', fontsize='xx-large')
        
        ax.ravel()[1].text(0.3, 0.4, 'Green Shorts', style='italic', fontsize='xx-large')
        
        ax.ravel()[1].text(0.3, 0.3, 'White shoes', style='italic', fontsize='xx-large')
        
        ax.ravel()[1].set_axis_off()
        plt.show()    

if __name__ == '__main__':
      main()
#args = parse_args()
#feat_model = load_model(args.checkpoint,compile = False)
#feat_model = Model(inputs=feat_model.input, outputs=feat_model.layers[-3].output)



