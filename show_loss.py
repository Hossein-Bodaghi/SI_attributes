import torch
#import numpy as np
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def show_loss(name,train_path,test_path,device):

    im = torch.load(test_path,map_location= torch.device(device))[:]
    im2 = torch.load(train_path,map_location= torch.device(device))[:]
    plt.figure('train')
    
    plt.xlabel('Epoch')
    plt.ylabel(name)
    plt.plot(im2, label='train')
    plt.plot(im, label = 'test')
    # plt.legend([plt_train, plt_test], ['train_'+name , 'val_'+name])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    for e, v in enumerate(im):
        if e%1 ==0:
            plt.text(e, v, '{:.2}'.format(v), color='g', fontsize= 'large') 
            plt.text(e, im2[e], '{:.2}'.format(im2[e]), color='r', fontsize= 'large')
    plt.show()
    

#%%

train_loss_path = '/home/hossein/anaconda3/envs/torchreid/deep-person-reid/my_osnet/result/V8_03/train_loss.pth'
test_loss_path = '/home/hossein/anaconda3/envs/torchreid/deep-person-reid/my_osnet/result/V8_03/test_attr_loss.pth'

train_F1_path = '/home/hossein/anaconda3/envs/torchreid/deep-person-reid/my_osnet/result/V8_03/train_attr_f1.pth'
test_F1_path = '/home/hossein/anaconda3/envs/torchreid/deep-person-reid/my_osnet/result/V8_03/test_attr_f1.pth'
show_loss(name='loss', train_path=train_loss_path, test_path=test_loss_path, device=device)
show_loss(name='F1', train_path=train_F1_path, test_path=test_F1_path, device=device)

#%%
train_loss_path = '/home/hossein/anaconda3/envs/torchreid/HydraPlusNet/result/v2.02/train_loss_MNet.pth'
test_loss_path = '/home/hossein/anaconda3/envs/torchreid/HydraPlusNet/result/v2.02/val_loss_MNet.pth'
show_loss(name='v2.02_MNet_loss', train_path=train_loss_path, test_path=test_loss_path, device=device)

# AF1 F1
train_F1_path = '/home/hossein/anaconda3/envs/torchreid/HydraPlusNet/result/v2.02/train_F1_MNet.pth'
test_F1_path = '/home/hossein/anaconda3/envs/torchreid/HydraPlusNet/result/v2.02/val_f1_MNet.pth'
show_loss(name='v2.02_MNet_F1', train_path=train_F1_path, test_path=test_F1_path, device=device)

# af1 acc 
train_acc = '/home/hossein/anaconda3/envs/torchreid/HydraPlusNet/result/v2.02/train_acc_MNet.pth'
test_acc = '/home/hossein/anaconda3/envs/torchreid/HydraPlusNet/result/v2.02/val_acc_MNet.pth'
show_loss(name='v2.02_MNet_acc', train_path=train_acc, test_path=test_acc, device=device)

#%%

im = torch.load(test_loss_path,map_location= torch.device(device))

