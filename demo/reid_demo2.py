#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:02:43 2020

@author: Hossein
"""

#%%
import os
import argparse
from preprocess import get_image, Demo_Market_Loader
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import skimage.io
import matplotlib.pyplot as plt
from torchreid import models
from torchreid import utils
import torch
from torch.utils.data import DataLoader
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

transform_simple = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
# '/home/dr/Downloads/Iran Test/Iran Fashion Dataset 990 Resized/001.jpg'
# '/home/dr/Pictures/m&r'
# '/home/dr/Pictures'
# python demo2.py --use_ready_features False --source_clothes '/home/dr/Pictures/m&r' --save_features '/home/dr/Pictures'
# '/home/dr/Pictures/all_features.npy'
# python demo2.py --use_ready_features False --source_clothes '/home/dr/Downloads/Insta Fashion/kafe_rosari_avat Resized' --save_features '/home/dr/Downloads/Insta Fashion/kafe_rosari_avat'
# python demo2.py --source_clothes '/home/dr/Downloads/Insta Fashion/kafe_rosari_avat Resized' --all_features '/home/dr/Downloads/Insta Fashion/kafe_rosari_avatall_features.npy'
# '/home/dr/Downloads/Insta Fashion/kafe_rosari_avat Resized' '/home/dr/Downloads/Insta Fashion/kafe_rosari_avatall_features.npy'


def main():
    torch.cuda.empty_cache()
    
    args = parse_args()
    
    # loading the model
    if args.model == 'osnet':        
        feat_model = models.build_model(
            name='osnet_x1_0',
            num_classes=751,
            loss='softmax',
            pretrained=False
        )
        
        weight_path = args.checkpoint
        utils.load_pretrained_weights(feat_model, weight_path)
        feat_model = feat_model.to(device)
        resolution = (256,128)
        
    
    keepgoing = 'y'
    while keepgoing == 'y':
        # keepgoing = input('do you want to continue?[y/n]')
        # print('in hamon vorodiast', keepgoing)
        # if keepgoing == 'n':
        #     break
        args = parse_args()    
        if args.use_ready_features == False:                
            all_features = []
            products_names = os.listdir(args.source_clothes)
            products_names.sort()
            gallery_data = Demo_Market_Loader(img_path = args.source_clothes, img_names = products_names,
                                                resolution = resolution)
            batch_size = 200
            gallery_loader = DataLoader(gallery_data,batch_size=batch_size,shuffle=False)
            feat_model.eval()
            with torch.no_grad():
                start = time.time()
                for data in gallery_loader:
                    features = feat_model(data['img'].to(device))
                    all_features.append(features) 
                all_features = torch.cat(all_features)
                all_features = all_features.to('cpu')
                all_features = all_features.detach().numpy()
                feat_save_path = os.path.join(args.save_features, 'market_gallery_' + args.model + '.npy')
                np.save(feat_save_path, all_features)
                finish = time.time()    
            print('the time of getting feature for {} images is:'.format(len(gallery_data)), finish - start)
            
        else:
            ar = input('put your photo here: ')
            args.input = ar[1:-2]
            img = get_image(args.input , resolution[0], resolution[1])
            fig , ax = plt.subplots(nrows=5,ncols=5, figsize=(5,8))
            
            ax.ravel()[2].set_title('Input Image',style='italic', fontsize='medium')
            ax.ravel()[2].imshow(img)
            
            for i in range(5):
                ax.ravel()[i].set_axis_off()

            img = transform_simple(img)
            img = img.to(device)
            feat_model.eval()
            pred_array = feat_model(torch.unsqueeze(img, dim=0))  
            pred_array = pred_array.to('cpu')
            pred_array = pred_array.detach().numpy()
            all_feat = np.load(args.all_features)
            similarity = cosine_similarity(np.squeeze(all_feat),pred_array.reshape(1,512))
            # similarity = my_cosin(all_feat , pred_array)
                #Sort all and Get Top N best silmilarities
            top_items_index = np.argsort(-similarity, axis=0)[:args.topN] #Get index of top N similar Items
            images_names = os.listdir(args.source_clothes)
            images_names.sort()
            for i in range(len(top_items_index)):
                path = os.path.join(args.source_clothes, images_names[int(top_items_index[i])])
                img = skimage.io.imread(path)
                #Show images
                ax.ravel()[i+5].imshow(img)
                ax.ravel()[i+5].set_axis_off()
                plt.imshow(img) 
            
            ax.ravel()[7].set_title('Results',style='italic', fontsize='medium')
            line = plt.Line2D((0.1,0.4),(0.735,0.735), color="k", linewidth=1)
            fig.add_artist(line)
            line2 = plt.Line2D((0.6,0.9),(0.735,0.735), color="k", linewidth=1)
            fig.add_artist(line2)
            #for input image lines
            line3 = plt.Line2D((0.1,0.35),(0.9,0.9), color="k", linewidth=1)
            fig.add_artist(line3)
            line4 = plt.Line2D((0.65,0.9),(0.9,0.9), color="k", linewidth=1)
            fig.add_artist(line4)

            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)
            plt.show()

if __name__ == '__main__':
      main()

#args = parse_args()
#feat_model = load_model(args.checkpoint,compile = False)
#feat_model = Model(inputs=feat_model.input, outputs=feat_model.layers[-3].output)
