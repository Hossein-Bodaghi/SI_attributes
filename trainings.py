#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:28:41 2021

@author: hossein

here we can find different types of trainigs 
that are define for person-attribute detection.
this is Hossein Bodaghies thesis
"""
import torch
import numpy as np
import torch.nn as nn
from metrics import tensor_metrics, IOU
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import part_data_delivery
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
'''
*
functions which are needed for training proccess.

tensor_max: takes a matrix and return a matrix with one hot vectors for max argument
lis2tensor: takes a list containing torch tensors and return a torch matrix  
id_onehot: takes id tensors and make them one hoted 
'''

class IOU_Loss(torch.nn.Module):
    
    def __init__(self):
        super(IOU_Loss,  self).__init__()
        self.eps=1e-6
    def forward(self, y_true, y_pred):
        intersection = torch.sum(torch.abs(y_true * y_pred), axis=-1)
        union = torch.sum(y_true,-1) + torch.sum(y_pred,-1) - intersection
        IOU = intersection / union
      
        Loss = - torch.mean( 1 - IOU )            
        return Loss  

def tensor_max(tensor):

    idx = torch.argmax(tensor, dim=1, keepdim=True)
    y = torch.zeros(tensor.size(),device=device).scatter_(1, idx, 1.)
    return y

def tensor_thresh(tensor, thr=0.5):
    out = (tensor>thr).float()
    return out
        

def list2tensor(list1):
    tensor = torch.zeros((len(list1),list1[0].size()[0]))
    for i in range(len(list1)):
        tensor[i] = list1[i]
    return tensor   

def id_onehot(id_,num_id):
    # one hot id vectors
    id1 = torch.zeros((len(id_),num_id))
    for i in range(len(id1)):
        a = id_[i]
        id1[i,a-1] = 1
    return id1

def CA_part_loss_calculator(out_data, data, part_loss, categorical = True, dynamic = False):
    attr_loss = []
    if dynamic:
        for key, loss in part_loss.items():
            loss_part = loss(torch.sigmoid(out_data[key]), data[key].float())
            attr_loss.append(loss_part)
            # print(key, '-->', out_data[key].size(), data[key].size(), loss_part)
    else:
        if categorical:
            for key, loss in part_loss.items():
                if key == 'body_type' or key == 'gender' or key == 'body_colour' or key == 'position' or key == 'accessories':
                    loss_part = loss(out_data[key], data[key].float())
                else :
                    loss_part = loss(out_data[key], data[key].argmax(dim=1))
                attr_loss.append(loss_part)
        else:
            for key, loss in part_loss.items():
                loss_part = loss(out_data[key], data[key].float())
                attr_loss.append(loss_part)
            
    loss_total = sum(attr_loss)
    attr_loss = torch.tensor(attr_loss)
    return attr_loss, loss_total

def CA_target_attributes_12(out_data, data, part_loss, need_tensor_max = False, categorical = True):
    'calculte y_attr and y_target for categorical and vectorize formats'
    if categorical:
        m = 0
        for key in part_loss:
            if key == 'body_type' or key == 'gender' or key == 'position' or key == 'accessories':
                y = tensor_thresh(torch.sigmoid(out_data[key]), 0.5)
                if m == 0:
                    y_target = data[key]
                else:
                    y_target = torch.cat((y_target, data[key]),dim = 1)
            elif key == 'body_colour':
                y = tensor_thresh(torch.sigmoid(out_data[key]), 0.5)
                if m == 0:
                    y_target = data[key]
                else:
                    y_target = torch.cat((y_target, data[key]), dim = 1)
            else :
                out_data[key] = softmax(out_data[key])
                if m == 0:
                    y_target = data[key]
                else:
                    y_target = torch.cat((y_target, data[key]), dim = 1)
                if need_tensor_max:
                    y = tensor_max(out_data[key])
                else:
                    y = tensor_thresh(out_data[key])
            if m == 0:
                y_attr = y
            else:
                y_attr = torch.cat((y_attr, y), dim=1)
            m += 1
    else:
        m = 0
        for key in part_loss:
            if key == 'body_type' or key == 'gender' or key == 'body_colour' or key == 'attributes':
                y = tensor_thresh(torch.sigmoid(out_data[key]), 0.5)
            else :
                out_data[key] = torch.sigmoid(out_data[key])
                if need_tensor_max:
                    y = tensor_max(out_data[key])
                else:
                    y = tensor_thresh(out_data[key])
            if m == 0:
                y_attr = y
                y_target = data[key]
            else :
                y_target = torch.cat((y_target, data[key]), dim = 1)
                y_attr = torch.cat(y_attr, y, dim=1)
            m += 1
    return y_attr, y_target
softmax = torch.nn.Softmax(dim=1)
#%%
def dict_training_multi_branch(num_epoch,
                               attr_net,
                               train_loader,
                               test_loader,
                               optimizer,
                               scheduler,
                               save_path,
                               device,
                               version,
                               part_loss,
                               categorical,
                               resume=False,
                               loss_train=None,
                               loss_test=None,
                               train_attr_F1=None,
                               test_attr_F1=None,
                               train_attr_acc=None,
                               test_attr_acc=None,
                               stoped_epoch=None):
    print('this is start')
    loss_min = 10000
    f1_best = 0
    if resume:
        train_loss = loss_train
        test_loss = loss_test
        F1_train = train_attr_F1
        F1_test = test_attr_F1
        Acc_train = train_attr_acc
        Acc_test = test_attr_acc
    else:
        train_loss = []
        test_loss = []
        # attributes metrics lists
        F1_train = []
        F1_test = []
        Acc_train = []
        Acc_test = []
    attr_loss_train = torch.zeros((num_epoch,len(part_loss)))
    attr_loss_test = torch.zeros((num_epoch,len(part_loss)))

    print('epoches started')
    if resume:
        start_epoch = stoped_epoch + 1
    else:
        start_epoch = 1
    
    tb = SummaryWriter()
    
    for epoch in range(start_epoch, num_epoch + 1):

        # torch.cuda.empty_cache()
        attr_net = attr_net.to(device)
        attr_net.train()
        loss_e = []
        loss_t = []
        loss_parts_train = torch.zeros(len(part_loss))
        loss_parts_test = torch.zeros(len(part_loss))

        # attributes temporary metrics lists
        ft_train = []
        ft_test = []
        acc_train = []
        acc_test = []
        with tqdm(train_loader, unit="batch") as tepoch:
            for idx, data in enumerate(tepoch):
                for key, _ in data.items():
                    data[key] = data[key].to(device)
                # forward step
                optimizer.zero_grad()
                out_data = attr_net.forward(data['img'])

                # compute losses and evaluation metrics:
                attr_loss, loss_total = CA_part_loss_calculator(out_data = out_data,
                                                                data = data,
                                                                part_loss = part_loss,
                                                                categorical = categorical)

                loss_parts_train += attr_loss

                y_attr, y_target = CA_target_attributes_12(out_data, data, part_loss, need_tensor_max=categorical, categorical=categorical)
                # evaluation    
                train_attr_metrics = tensor_metrics(y_target.float(), y_attr)
                # append results
                ft_train.append(train_attr_metrics[-2])
                acc_train.append(train_attr_metrics[-3])
                loss_e.append(loss_total.item())
                
                # backward step
                loss_total.backward()
                # optimization step
                optimizer.step()
                
                # new loss functions
                
                # print log
                tb.add_scalar('Loss/train', loss_total.item(), idx)
                tb.add_scalar('Accuracy/train', acc_train[-1], idx)
                tb.add_scalar('F1/train', ft_train[-1], idx)

                if idx % 1 == 0:
                    tepoch.set_postfix(loss=loss_total.item(), F1=100*ft_train[-1], Acc=100.*acc_train[-1])
                    """
                    print('\nTrain Epoch: {} [{}/{} , lr {}] \t Loss: {:.6f} \nattr_metrics: F1_attr: {:.3f} acc_attr{:.3f}'.format(
                        epoch, idx , len(train_loader),
                        optimizer.param_groups[0]['lr'],
                        loss_total.item(),ft_train[-1], acc_train[-1]))
                    """

        train_loss.append(np.mean(loss_e))
        F1_train.append(np.mean(ft_train))
        Acc_train.append(np.mean(acc_train))

        ##calculte mean of loss for each iteration for each part
        loss_parts_train = loss_parts_train / (idx + 1)
        ##create a matrix of loss in each epoch for each part
        attr_loss_train[epoch-1, :] = loss_parts_train

        # evaluation:     
        attr_net.eval()
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                
                for key, _ in data.items():
                    data[key] = data[key].to(device)
                    
                # forward step
                out_data = attr_net.forward(data['img'])           
                
                # compute losses and evaluation metrics:
                attr_loss, loss_total = CA_part_loss_calculator(out_data=out_data,
                                                                data=data,
                                                                part_loss=part_loss,
                                                                categorical=categorical)
                loss_parts_test += attr_loss
                
                y_attr, y_target = CA_target_attributes_12(out_data, data, part_loss, need_tensor_max=categorical, categorical=categorical)

                test_attr_metrics = tensor_metrics(y_target.float(), y_attr)
                ft_test.append(test_attr_metrics[-2])
                acc_test.append(test_attr_metrics[-3]) 
                loss_t.append(loss_total.item())
                
        test_loss.append(np.mean(loss_t))
        F1_test.append(np.mean(ft_test))
        Acc_test.append(np.mean(acc_test))

        #calculte mean of loss for each iteration for each part
        loss_parts_test = loss_parts_test / (idx + 1)
        #create a matrix of loss in each epoch for each part
        attr_loss_test[epoch-1, :] = loss_parts_test

        tb.add_scalar("Loss", test_loss[-1], epoch)

        print('Epoch: {}\ntrain loss: {:.6f}\ntest loss: {:.6f}\n\nF1 train: {:.4f}\nF1 test: {:.4f}\n\nacc_train: {:.4f}\nacc_test: {:.4f}\n'.format(
                    epoch,train_loss[-1],test_loss[-1],F1_train[-1],F1_test[-1],Acc_train[-1],Acc_test[-1]))

        scheduler.step()
        
        # saving training results
        saving_path = os.path.join(save_path, version)
        if not os.path.exists(saving_path):
            os.mkdir(saving_path)
            
        torch.save(train_loss, os.path.join(saving_path, 'train_loss.pth'))
        torch.save(F1_train, os.path.join(saving_path, 'train_attr_f1.pth'))
        torch.save(Acc_train, os.path.join(saving_path, 'train_attr_acc.pth'))
                   
        torch.save(test_loss, os.path.join(saving_path, 'test_attr_loss.pth'))
        torch.save(F1_test, os.path.join(saving_path, 'test_attr_f1.pth'))
        torch.save(Acc_test, os.path.join(saving_path, 'test_attr_acc.pth'))           

        torch.save(attr_net, os.path.join(saving_path, 'attr_net.pth'))
        torch.save(optimizer.state_dict(), os.path.join(saving_path, 'optimizer.pth')) 
        torch.save(scheduler.state_dict(), os.path.join(saving_path, 'scheduler.pth'))
        torch.save(epoch, os.path.join(saving_path, 'training_epoch.pth'))            

        #attr losses saving
        
        torch.save(attr_loss_train[:epoch], os.path.join(saving_path, 'train_part_loss.pth'))
        torch.save(attr_loss_test[:epoch], os.path.join(saving_path, 'test_part_loss.pth'))

        d = 0
        if test_loss[-1] < loss_min: 
            loss_min = test_loss[-1]
            d += 1
            torch.save(attr_net.state_dict(), os.path.join(saving_path, 'best_attr_net.pth'))
            torch.save(epoch, os.path.join(saving_path, 'best_epoch.pth'))
            print('test loss improved')
            
        if F1_test[-1] > f1_best: 
            f1_best = F1_test[-1]
            if d ==0:
                print('test f1 improved','\n')
            else:
                print('test f1 improved','\n')
#%%
    
def dict_training_dynamic_loss(num_epoch,
                               dataset,
                               attr_net,
                               weight_nets,
                               weights,
                               train_loader,
                               test_loader,
                               optimizer,
                               scheduler,
                               save_path,
                               device,
                               version,
                               part_loss,
                               categorical,
                               resume=False,
                               loss_train=None,
                               loss_test=None,
                               train_attr_F1=None,
                               test_attr_F1=None,
                               train_attr_acc=None,
                               test_attr_acc=None,
                               stoped_epoch=None):
    print('this is start')
    loss_min = 10000
    f1_best = 0
    if resume:
        train_loss = loss_train
        test_loss = loss_test
        F1_train = train_attr_F1
        F1_test = test_attr_F1
        Acc_train = train_attr_acc
        Acc_test = test_attr_acc
    else:
        train_loss = []
        test_loss = []
        # attributes metrics lists
        F1_train = []
        F1_test = []
        Acc_train = []
        Acc_test = []
    attr_loss_train = torch.zeros((num_epoch,len(part_loss)))
    attr_loss_test = torch.zeros((num_epoch,len(part_loss)))

    print('epoches started')
    if resume:
        start_epoch = stoped_epoch + 1
    else:
        start_epoch = 1
    
    tb = SummaryWriter()
    
    for epoch in range(start_epoch, num_epoch + 1):

        # torch.cuda.empty_cache()
        attr_net = attr_net.to(device)
        attr_net.train()
        loss_e = []
        loss_t = []
        loss_parts_train = torch.zeros(len(part_loss))
        loss_parts_test = torch.zeros(len(part_loss))

        # attributes temporary metrics lists
        ft_train = []
        ft_test = []
        acc_train = []
        acc_test = []
        new_weights = {}
        with tqdm(train_loader, unit="batch") as tepoch:
            for idx, data in enumerate(tepoch):
                for key in data:
                    data[key] = data[key].to(device)
                for key in weights:    
                    weight_nets[key] = weight_nets[key].to(device)
                    weight_net = weight_nets[key](weights[key])
                    new_weights.update({key:weight_nets[key](weights[key])})
                    # creating new part_loss
                part_loss = part_data_delivery(new_weights, device, dataset = dataset, dynamic=True)
                
                # forward step
                optimizer.zero_grad()
                out_data = attr_net.forward(data['img'])

                # compute losses and evaluation metrics:
                attr_loss, loss_total = CA_part_loss_calculator(out_data = out_data,
                                                                data = data,
                                                                part_loss = part_loss,
                                                                categorical = categorical, dynamic=True)

                loss_parts_train += attr_loss

                y_attr, y_target = CA_target_attributes_12(out_data, data, part_loss, need_tensor_max=categorical, categorical=categorical)
                # evaluation    
                train_attr_metrics = tensor_metrics(y_target.float(), y_attr)
                # append results
                ft_train.append(train_attr_metrics[-2])
                acc_train.append(train_attr_metrics[-3])
                loss_e.append(loss_total.item())
                
                # backward step
                loss_total.backward()
                # optimization step
                optimizer.step()
                
                # new loss functions
                
                # print log
                tb.add_scalar('Loss/train', loss_total.item(), idx)
                tb.add_scalar('Accuracy/train', acc_train[-1], idx)
                tb.add_scalar('F1/train', ft_train[-1], idx)

                if idx % 1 == 0:
                    tepoch.set_postfix(loss=loss_total.item(), F1=100*ft_train[-1], Acc=100.*acc_train[-1])
                    """
                    print('\nTrain Epoch: {} [{}/{} , lr {}] \t Loss: {:.6f} \nattr_metrics: F1_attr: {:.3f} acc_attr{:.3f}'.format(
                        epoch, idx , len(train_loader),
                        optimizer.param_groups[0]['lr'],
                        loss_total.item(),ft_train[-1], acc_train[-1]))
                    """

        train_loss.append(np.mean(loss_e))
        F1_train.append(np.mean(ft_train))
        Acc_train.append(np.mean(acc_train))

        ##calculte mean of loss for each iteration for each part
        loss_parts_train = loss_parts_train / (idx + 1)
        ##create a matrix of loss in each epoch for each part
        attr_loss_train[epoch-1, :] = loss_parts_train

        # evaluation:     
        attr_net.eval()
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                
                for key, _ in data.items():
                    data[key] = data[key].to(device)
                    
                # forward step
                out_data = attr_net.forward(data['img'])           
                
                # compute losses and evaluation metrics:
                attr_loss, loss_total = CA_part_loss_calculator(out_data=out_data,
                                                                data=data,
                                                                part_loss=part_loss,
                                                                categorical=categorical, dynamic=True)
                
                loss_parts_test += attr_loss
                
                y_attr, y_target = CA_target_attributes_12(out_data, data, part_loss, need_tensor_max=categorical, categorical=categorical)

                test_attr_metrics = tensor_metrics(y_target.float(), y_attr)
                ft_test.append(test_attr_metrics[-2])
                acc_test.append(test_attr_metrics[-3]) 
                loss_t.append(loss_total.item())
                
        test_loss.append(np.mean(loss_t))
        F1_test.append(np.mean(ft_test))
        Acc_test.append(np.mean(acc_test))

        #calculte mean of loss for each iteration for each part
        loss_parts_test = loss_parts_test / (idx + 1)
        #create a matrix of loss in each epoch for each part
        attr_loss_test[epoch-1, :] = loss_parts_test

        tb.add_scalar("Loss", test_loss[-1], epoch)

        print('Epoch: {}\ntrain loss: {:.6f}\ntest loss: {:.6f}\n\nF1 train: {:.4f}\nF1 test: {:.4f}\n\nacc_train: {:.4f}\nacc_test: {:.4f}\n'.format(
                    epoch,train_loss[-1],test_loss[-1],F1_train[-1],F1_test[-1],Acc_train[-1],Acc_test[-1]))

        scheduler.step()
        
        # saving training results
        saving_path = os.path.join(save_path, version)
        if not os.path.exists(saving_path):
            os.mkdir(saving_path)
            
        torch.save(train_loss, os.path.join(saving_path, 'train_loss.pth'))
        torch.save(F1_train, os.path.join(saving_path, 'train_attr_f1.pth'))
        torch.save(Acc_train, os.path.join(saving_path, 'train_attr_acc.pth'))
                   
        torch.save(test_loss, os.path.join(saving_path, 'test_attr_loss.pth'))
        torch.save(F1_test, os.path.join(saving_path, 'test_attr_f1.pth'))
        torch.save(Acc_test, os.path.join(saving_path, 'test_attr_acc.pth'))           

        torch.save(attr_net, os.path.join(saving_path, 'attr_net.pth'))
        torch.save(optimizer.state_dict(), os.path.join(saving_path, 'optimizer.pth')) 
        torch.save(scheduler.state_dict(), os.path.join(saving_path, 'scheduler.pth'))
        torch.save(epoch, os.path.join(saving_path, 'training_epoch.pth'))          
        torch.save(new_weights, os.path.join(saving_path, 'loss_weights.pth'))

        #attr losses saving
        
        torch.save(attr_loss_train[:epoch], os.path.join(saving_path, 'train_part_loss.pth'))
        torch.save(attr_loss_test[:epoch], os.path.join(saving_path, 'test_part_loss.pth'))

        d = 0
        if test_loss[-1] < loss_min: 
            loss_min = test_loss[-1]
            d += 1
            torch.save(attr_net.state_dict(), os.path.join(saving_path, 'best_attr_net.pth'))
            torch.save(epoch, os.path.join(saving_path, 'best_epoch.pth'))
            print('test loss improved')
            
        if F1_test[-1] > f1_best: 
            f1_best = F1_test[-1]
            if d ==0:
                print('test f1 improved','\n')
            else:
                print('test f1 improved','\n')                
#%%
def dict_evaluating_multi_branch(attr_net, test_loader, save_path, device, part_loss, categorical):

    attr_net = attr_net.to(device)
    # evaluation:     
    attr_net.eval()
    with torch.no_grad():
        targets = []
        predicts = []
        for idx, data in enumerate(test_loader):
            
            for key, _ in data.items():
                data[key] = data[key].to(device)
                
            # forward step
            out_data = attr_net.forward(data['img'])           
            y_attr, y_target = CA_target_attributes_12(out_data, data, part_loss, need_tensor_max=categorical, categorical=categorical)
            predicts.append(y_attr.to('cpu'))
            targets.append(y_target.to('cpu'))
    predicts = torch.cat(predicts)
    targets = torch.cat(targets)   
    iou_result = IOU(predicts, targets)
    test_attr_metrics = tensor_metrics(y_target.float(), y_attr)           
    return [test_attr_metrics, iou_result]
#%%
def take_out_multi_branch(attr_net,test_loader,save_path,device,part_loss,categorical):

    attr_net = attr_net.to(device)
    # evaluation:     
    attr_net.eval()
    with torch.no_grad():
        targets = []
        predicts = []
        for idx, data in enumerate(test_loader):
            
            for key, _ in data.items():
                data[key] = data[key].to(device)
                
            # forward step
            out_data = attr_net.forward(data['img'])           
            y_attr, y_target = CA_target_attributes_12(out_data, data, part_loss, need_tensor_max=categorical, categorical=categorical)
            predicts.append(y_attr.to('cpu'))
            targets.append(y_target.to('cpu'))
    predicts = torch.cat(predicts)
    targets = torch.cat(targets)   
         
    return [predicts, targets]