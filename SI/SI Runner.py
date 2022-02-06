#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 14:15:34 2021

@author: hossein
""" 

from functions import load_model, latent_feat_extractor, si_calculator, load_saved_features, forward_selection_SI, Plot_SI
from loaders import CA_Loader
import torch
from torch.utils.data import DataLoader
from delivery import data_delivery
import numpy as np
import argparse

def default_argument_parser():

    parser = argparse.ArgumentParser(description="SI Runner")
    parser.add_argument("--features-ready", action="store_true", help="features were saved or not")
    parser.add_argument("--layer", default="out_featuremap", help="layer to extract")
    parser.add_argument("--si-all", action="store_true", help="run si based on all")
    parser.add_argument("--clss", default="gender", help="which class to do forward select on")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    return parser


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = ['osnet', 'attr_net']
    weight_path = './SI/osnet_x1_0_msmt17.pth'
    train_img_path = './datasets/Dukemtmc/bounding_box_train'
    path_attr_train = './attributes/CA_Duke_train_with_id.npy'
    
    network = load_model(model=models[0], weight_path=weight_path)

    attr = data_delivery(train_img_path,
                    path_attr=path_attr_train,
                    need_parts=True,
                    need_attr=True,
                    dataset = 'CA_Duke')
    
    train_idx = np.arange(len(attr['img_names']))

    train_data = CA_Loader(img_path=train_img_path,
                            attr=attr,
                            resolution=(256,128),
                            indexes=train_idx,
                            dataset = 'CA_Duke',
                            need_id = False,
                            two_transforms = False) 

    if args.features_ready == True:
        out_layers = load_saved_features(args.layer)
    else:
        test_loader = DataLoader(train_data, batch_size=128, shuffle=False)
        si = latent_feat_extractor(net=network, test_loader=test_loader,
                                            layer=args.layer, save_path='./saving/', 
                                            device=device, use_adapt=False,
                                            final_size = (8, 8), mode=None)
        import pickle
        with open(args.layer+'.pkl', 'wb') as f:
            pickle.dump(si, f)


    if args.si_all == False:
        si = si_calculator(out_layers, train_data)

    trend, layer_nums = forward_selection_SI(out_layers.to('cuda'), train_data.gender.to('cuda'), args.clss)

    Plot_SI(layer_nums, trend)