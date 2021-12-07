from delivery import data_delivery 
import torch
from torchreid import models, utils    
import os
import numpy as np

from torchreid.utils.torchtools import open_specified_layers
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from models import CA_market_model2

attr_net_camarket = models.build_model(
    name='osnet_x1_0',
    num_classes=751,
    loss='softmax',
    pretrained=False
)

pretrain_path = './osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth'
utils.load_pretrained_weights(attr_net_camarket, pretrain_path)

"""
attr_net_camarket = CA_market_model2(model=model,
                  feature_dim = 512,
                  num_id = 751,
                  attr_dim = 46,
                  need_id = False,
                  need_attr = True,
                  need_collection = False)

model_path = './result/V8_01/best_attr_net.pth'
trained_net = torch.load(model_path)
attr_net_camarket.load_state_dict(trained_net.state_dict())
"""

attr_net_camarket = attr_net_camarket.to(device)

torch.cuda.empty_cache()

main_path = './Market-1501-v15.09.15/gt_bbox/'
path_attr = './attributes/total_attr.npy'
test_idx = torch.load('./attributes/test_idx_full.pth')

indices = np.load('./results/layersleg.npy')
indices = torch.from_numpy(indices).to(device)

import random
test_idx = random.sample(test_idx, 300)

attr = data_delivery(main_path=main_path,
                     path_attr=path_attr,
                     need_collection=True,
                     double=False,
                     need_attr=False)

from loaders import MarketLoader4
from torch.utils.data import DataLoader

test_data = MarketLoader4(img_path=main_path,
                          attr=attr,
                          resolution=(256, 128),
                          indexes=test_idx,
                          need_attr = False,
                          need_collection=True,
                          need_id = False,
                          two_transforms = False,                          
                          ) 

test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

attr_net_camarket.eval()

import torch.nn.functional as F
import cv2
import numpy as np 
height = 256
width = 128
GRID_SPACING = 10
import os.path as osp
img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]

#os.mkdir('./Feature visualized')
#for i in range(300):
#    os.mkdir(os.path.join('./Feature visualized', 'person_'+str(i)))


attr_net_camarket.eval()

with torch.no_grad():

    for batch_idx, data in enumerate(test_loader):

            #os.mkdir('C:/Users/ASUS/Desktop/Feature visualized/'+ str(batch_idx))

            data['img'] = data['img'].to(device)
            #outputs = attr_net_camarket(data['img'], return_featuremaps=True)
            outputs = attr_net_camarket.layer_extractor(data['img'], layer='out_conv4')

            outputs = torch.index_select(outputs, 1, indices)

            outputs = (outputs**2).sum(1)
            b, h, w = outputs.size()
            outputs = outputs.view(b, h * w)
            outputs = F.normalize(outputs, p=2, dim=1)
            outputs = outputs.view(b, h, w)

            if device:
                    img, outputs = data['img'].sum(0).cpu(), outputs.cpu()

            # RGB image
            for t, m, s in zip(img, img_mean, img_std):
                t.mul_(s).add_(m).clamp_(0, 1)
            img_np = np.uint8(np.floor(img.numpy() * 255))
            img_np = img_np.transpose((1, 2, 0)) # (c, h, w) -> (h, w, c)

            for j in range(outputs.size(0)):
                        # activation map
                        am = outputs[j].detach().numpy()
                        am = cv2.resize(am, (width, height))
                        am = 255 * (am - np.min(am)) / (
                            np.max(am) - np.min(am) + 1e-12
                        )
                        am = np.uint8(np.floor(am))
                        am = cv2.applyColorMap(am, cv2.COLORMAP_JET)

                        # overlapped
                        overlapped = img_np*0.3 + am*0.7
                        overlapped[overlapped > 255] = 255
                        overlapped = overlapped.astype(np.uint8)
                        # save images in a single figure (add white spacing between images)
                        # from left to right: original image, activation map, overlapped image
                        grid_img = 255 * np.ones(
                            (height, 3*width + 2*GRID_SPACING, 3), dtype=np.uint8
                        )
                        grid_img[:, :width, :] = img_np[:, :, ::-1]
                        grid_img[:,
                                width + GRID_SPACING:2*width + GRID_SPACING, :] = am
                        grid_img[:, 2*width + 2*GRID_SPACING:, :] = overlapped

                        cv2.imwrite(osp.join('./Feature visualized/', 'person_'+str(batch_idx) +'.jpg'), overlapped)

            if (batch_idx+1) % 10 == 0:
                print(
                    '- done batch {}/{}'.format(batch_idx + 1, len(test_loader))
                    )
