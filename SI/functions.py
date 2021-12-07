#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 13:07:05 2021

@author: hossein
"""

import torch
from torchreid import models, utils    
import os
from models import CA_market_model2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model, weight_path,
                need_id = False,
                need_attr = True,
                need_collection = False ):
    '''
    models = ['osnet', 'attr_net']
    model_path = './result/V8_01/best_attr_net.pth'
    weight_path1 = '/home/hossein/Downloads/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth'

    '''    
    network = models.build_model(
        name='osnet_x1_0',
        num_classes=751,
        loss='softmax',
        pretrained=False
    )
    if model == 'osnet':
        utils.load_pretrained_weights(network, weight_path)
        return network
        
    elif model == 'attr_net':        
        attr_net_camarket = CA_market_model2(model=network,
                          feature_dim = 512,
                          num_id = 751,
                          attr_dim = 46,
                          need_id = need_id,
                          need_attr = need_attr,
                          need_collection = need_collection)
        
        
        trained_net = torch.load(weight_path)
        attr_net_camarket.load_state_dict(trained_net.state_dict())


def latent_feat_extractor(net, test_loader,
                          layer, save_path, device, use_adapt = True,
                          final_size = (8, 8), mode='return'):
    '''
    layers = ['out_maxpool', 'out_conv2','out_conv3','out_conv4','out_featuremap','out_globalavg','out_fc']        
    mode = ['saving', 'return']
    '''
    net = net.to(device)
    features = torch.Tensor().to('cpu')
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data['img'] = data['img'].to(device)    
            out_layer = net.layer_extractor(data['img'], layer=layer) # extract features from different latent layers of osnet
            if use_adapt == True:
                madapt = torch.nn.AdaptiveAvgPool2d(final_size) # define adaptive pooling
                out_layer = madapt(out_layer) # dimention reduction with adaptive pooling
            
            if mode == 'return':
                features = torch.cat((features, out_layer.to('cpu')), 0)
            
            if mode == 'saving':
                # save features differently
                saving_path = os.path.join(save_path, layer+'_Part_'+str(i)+'.pt')
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                torch.save(out_layer, saving_path)

    return features


def si_calculator(X, Y):
    #X(12000,30,30,3)  Y(12000, 43)  flatx (12000, 2700)
    #print('Computing SI')
    X = torch.flatten(X, start_dim=1)
    idxs2 = torch.sort(torch.cdist(X, X)).indices[:,1:2]
    si = 0
    for i in range(X.shape[0]):
            if torch.equal(torch.reshape(Y[idxs2[i]], (-1,)), Y[i]):
                si += 1
    total = X.shape[0]
    del X
    del idxs2
    del Y
    torch.cuda.empty_cache()
    return si/total
        

def load_saved_features(layer, end):
    import glob
    for i, path in enumerate(glob.glob('./saving/'+layer+'_*.pt')):
            if i == end:
                break
            if i == 0:
                    out_layers = torch.load(path).to('cpu')
            else:
                temp = torch.load(path).to('cpu')
                out_layers = torch.cat((out_layers, temp), 0)
    
    del temp
    return out_layers


def pca_converter(out_layers):
    import numpy as np
    from sklearn.decomposition import PCA
    pca = PCA()
    out_layers = torch.flatten(out_layers, start_dim=2)
    for i in range(out_layers.shape[1]):
        temp = pca.fit_transform(out_layers[:,i,:].cpu().detach().numpy())
        temp = temp[:, np.newaxis, :]
        if i == 0:
            new_features = temp
        else:
            new_features = np.concatenate((new_features, temp), axis=1) 

    out_layers = torch.from_numpy(new_features).to(device)
    del new_features
    return out_layers


def forward_selection_SI(out_layers, label):
    import time

    si = []
    layer_nums = []
    trend = []
    j = 0

    while True:
        t = time.time()
        for i in range(out_layers.shape[1]):
            if j == 0:
                si.append(si_calculator(out_layers[:,i,:], label)) 
            else:
                si.append(si_calculator(torch.cat((best_layers, out_layers[:,i,:]), dim=1), label))
            if i%50 == 0:
                print('calculated SI on '+str(i)+' layers')

        max_value  = max(si)
        trend.append(max_value)
        max_index = si.index(max_value)
        layer_nums.append(max_index)
        torch.cuda.empty_cache()
        si = []
        j += 1

        if j == 1:
                best_layers = out_layers[:,max_index,:]
        else:
                best_layers = torch.cat((out_layers[:,max_index,:],best_layers), dim=1)

        out_layers = out_layers.transpose(0, 1)
        out_layers = torch.cat((out_layers[:max_index],out_layers[max_index+1:]))
        out_layers = out_layers.transpose(0, 1)

        if (len(trend) > 2) and (trend[-1] < trend[-2]):
            break

        print('time: '+"{:.2f}".format(time.time()-t)+'s')
        torch.cuda.empty_cache()

    return trend, layer_nums