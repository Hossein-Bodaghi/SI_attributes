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
from metrics import tensor_metrics, boolian_metrics, category_metrics
import gc
import time
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

#%%
'''
*
functions which are needed for training proccess.

tensor_max: takes a matrix and return a matrix with one hot vectors for max argument
lis2tensor: takes a list containing torch tensors and return a torch matrix  
id_onehot: takes id tensors and make them one hoted 
'''
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

softmax = torch.nn.Softmax(dim=1)
#%%
def train_collection(num_epoch,
                     attr_net,
                     train_loader,
                     test_loader,
                     optimizer,
                     scheduler,
                     criterion1,
                     criterion2,
                     saving_path,
                     version): 
    # def training(train_loader,test_loader,generator,classifier,num_epoch,optimizer,criterion1,criterion2,scheduler,device):
    train_loss = []
    test_loss = []
    F1_train = []
    F1_test = []
    gc.collect()
    torch.cuda.empty_cache()
    attr_net = attr_net.to(device)
    
    for epoch in range(1,num_epoch+1):
        
        attr_net.train()
        loss_e = []
        loss_t = []
        ft_train = []
        ft_test = []
        
        for idx, data in enumerate(train_loader):
            
            # forward step
            optimizer.zero_grad()
            out_data = attr_net(data[0].to(device))
            
            for i in range(len(data)):
                data[i] = data[i].to(device)
            # compute losses and evaluation metrics:
                
            # head 
            loss0 = criterion1(out_data[0],data[2].argmax(dim=1))        
            y = tensor_max(out_data[0])
            metrics = tensor_metrics(data[2].float(),y)
            ft_train.append(metrics[7])
            
            # body
            loss1 = criterion1(out_data[1],data[3].argmax(dim=1))
            y = tensor_max(out_data[1])
            metrics = tensor_metrics(data[3].float(),y)
            ft_train.append(metrics[7])
            
            # body type
            loss2 = criterion2(out_data[2].squeeze(),data[4].float())    
            y = tensor_thresh(out_data[2], 0.5)
            metrics = boolian_metrics(data[4].float(),y)
            ft_train.append(metrics[3])
            
            # leg
            loss3 = criterion1(out_data[3],data[5].argmax(dim=1))
            y = tensor_max(out_data[3])
            metrics = tensor_metrics(data[5].float(),y)
            ft_train.append(metrics[7])
            
            # foot 
            loss4 = criterion1(out_data[4],data[6].argmax(dim=1))      
            y = tensor_max(out_data[4])
            metrics = tensor_metrics(data[6].float(),y)  
            ft_train.append(metrics[7])
            
            # gender
            loss5 = criterion2(out_data[5].squeeze(),data[7].float())
            y = tensor_thresh(out_data[5], 0.5)
            metrics = boolian_metrics(data[7].float(),y)  
            ft_train.append(metrics[3])
            
            # bags
            loss6 = criterion1(out_data[6],data[8].argmax(dim=1))
            y = tensor_max(out_data[6])
            metrics = tensor_metrics(data[8].float(),y)
            ft_train.append(metrics[7])
            
            # body colour
            loss7 = criterion1(out_data[7],data[9].argmax(dim=1))      
            y = tensor_max(out_data[7])
            metrics = tensor_metrics(data[9].float(),y)
            ft_train.append(metrics[7])
            
            # leg colour
            loss8 = criterion1(out_data[8],data[10].argmax(dim=1))
            y = tensor_max(out_data[8])
            metrics = tensor_metrics(data[10].float(),y)
            ft_train.append(metrics[7])
            
            # foot colour
            loss9 = criterion1(out_data[9],data[11].argmax(dim=1))
            y = tensor_max(out_data[9])
            metrics = tensor_metrics(data[11].float(),y)      
            ft_train.append(metrics[7])
            
            # total loss
            loss = loss0+loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9
            loss_e.append(loss.item())
            
            # backward step
            loss.backward()
            
            # optimization step
            optimizer.step()
            scheduler.step()
            # print log
            if idx % 1 == 0:
                print('Train Epoch: {} [{}/{} , lr {}] \t Loss: {:.6f} , F1: {:.3f}'.format(
                    epoch, idx , len(train_loader),
                     optimizer.param_groups[0]['lr'],
                      loss.item(),np.mean(ft_train)))
       
        train_loss.append(np.mean(loss_e))
        F1_train.append(np.mean(ft_train))
        torch.save(train_loss,saving_path+'trainloss_'+version+'.pth')
        # evaluation:     
        attr_net.eval()
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                
                # data = data.to(device) 'list' object has no attribute 'to'
                out_data = attr_net(data[0].to(device))
                for i in range(len(data)):
                    data[i] = data[i].to(device)
                # compute losses and evaluation metrics:
                    
                # head 
                loss0 = criterion1(out_data[0],data[2].argmax(dim=1))        
                y = tensor_max(out_data[0])
                metrics = tensor_metrics(data[2].float(),y)
                ft_test.append(metrics[7])
                
                # body
                loss1 = criterion1(out_data[1],data[3].argmax(dim=1))
                y = tensor_max(out_data[1])
                metrics = tensor_metrics(data[3].float(),y)
                ft_test.append(metrics[7])
                
                # body type
                loss2 = criterion2(out_data[2].squeeze(),data[4].float())    
                y = tensor_thresh(out_data[2], 0.5)
                metrics = boolian_metrics(data[4].float(),y)
                ft_test.append(metrics[3])
                
                # leg
                loss3 = criterion1(out_data[3],data[5].argmax(dim=1))
                y = tensor_max(out_data[3])
                metrics = tensor_metrics(data[5].float(),y)
                ft_test.append(metrics[7])
                
                # foot 
                loss4 = criterion1(out_data[4],data[6].argmax(dim=1))      
                y = tensor_max(out_data[4])
                metrics = tensor_metrics(data[6].float(),y)  
                ft_test.append(metrics[7])
                
                # gender
                loss5 = criterion2(out_data[5].squeeze(),data[7].float())
                y = tensor_thresh(out_data[5], 0.5)
                metrics = boolian_metrics(data[7].float(),y)  
                ft_test.append(metrics[3])
                
                # bags
                loss6 = criterion1(out_data[6],data[8].argmax(dim=1))
                y = tensor_max(out_data[6])
                metrics = tensor_metrics(data[8].float(),y)
                ft_test.append(metrics[7])
                
                # body colour
                loss7 = criterion1(out_data[7],data[9].argmax(dim=1))      
                y = tensor_thresh(out_data[7], 0.5)
                metrics = tensor_metrics(data[9].float(),y)
                ft_test.append(metrics[7])
                
                # leg colour
                loss8 = criterion1(out_data[8],data[10].argmax(dim=1))
                y = tensor_max(out_data[8])
                metrics = tensor_metrics(data[10].float(),y)
                ft_test.append(metrics[7])
                
                # foot colour
                loss9 = criterion1(out_data[9],data[11].argmax(dim=1))
                y = tensor_max(out_data[9])
                metrics = tensor_metrics(data[11].float(),y)      
                ft_test.append(metrics[7])
                
                # total loss
                loss = loss0+loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9
                loss_t.append(loss.item())
        test_loss.append(np.mean(loss_t))
        F1_test.append(np.mean(ft_test))
        print('Epoch: {}\ntrain loss: {:.6f}\ntest loss: {:.6f}\n\nF1 train: {:.4f}\nF1 test: {:.4f} '.format(
                    epoch,train_loss[-1],test_loss[-1],F1_train[-1],F1_test[-1]))
        torch.save(test_loss,saving_path+'testloss_'+version+'.pth')
        torch.save(F1_test,saving_path+'testF1_'+version+'.pth')
        
        scheduler.step()
        if len(F1_test)>2: 
            if F1_test[-1] > F1_test[-2]:
                print('our net improved')
                torch.save(attr_net , saving_path+'attrnet_'+version+'.pth')
                torch.save(optimizer, saving_path+'optimizer_'+version+'.pth')
                
  
#%%
def train_collection_id(num_epoch,
                     attr_net,
                     train_loader,
                     test_loader,
                     optimizer,
                     scheduler,
                     criterion1,
                     criterion2,
                     saving_path,
                     num_id,
                     device,
                     version,
                     resume=False,
                     loss_train = None,
                     loss_test=None,
                     train_F1=None,
                     test_F1=None,
                     stop_epoch=None): 
    # model_output (tuple): (out_head,out_body,out_body_type,out_leg,out_foot,out_gender,out_bags,out_body_colour,out_leg_colour,out_foot_colour,out_id)
    # loader_outpu (tuple): (img,id,head,body,body_type,leg,foot,gender,bags,body_colour,leg_colour,foot_colour)
    print('this is start')
    if resume:
        train_loss = loss_train
        test_loss = loss_test
        F1_train = train_F1
        F1_test = test_F1
    else:
        train_loss = []
        test_loss = []
        F1_train = []
        F1_test = []
    # gc.collect()
    # torch.cuda.empty_cache()
    # attr_net = attr_net.to(device)
    print('epoches started')
    if resume:
        start_epoch = stop_epoch+1
    else:
        start_epoch = 1    
    for epoch in range(start_epoch,num_epoch+1):
        
        torch.cuda.empty_cache()
        attr_net = attr_net.to(device)        
        attr_net.train()
        loss_e = []
        loss_t = []
        ft_train = []
        ft_test = []
        
        for idx, data in enumerate(train_loader):
            
            # forward step
            optimizer.zero_grad()
            out_data = attr_net.forward(data[0].to(device))
            
            for i in range(len(data)):
                data[i] = data[i].to(device)
            # compute losses and evaluation metrics:
                
            # head 
            loss0 = criterion1(out_data[0],data[2].argmax(dim=1))        
            y = tensor_max(out_data[0])
            metrics = tensor_metrics(data[2].float(),y)
            ft_train.append(metrics[7])
            
            # body
            loss1 = criterion1(out_data[1],data[3].argmax(dim=1))
            y = tensor_max(out_data[1])
            metrics = tensor_metrics(data[3].float(),y)
            ft_train.append(metrics[7])
            
            # body type
            loss2 = criterion2(out_data[2].squeeze(),data[4].float())    
            y = tensor_thresh(out_data[2], 0.5)
            metrics = boolian_metrics(data[4].float(),y)
            ft_train.append(metrics[3])
            
            # leg
            loss3 = criterion1(out_data[3],data[5].argmax(dim=1))
            y = tensor_max(out_data[3])
            metrics = tensor_metrics(data[5].float(),y)
            ft_train.append(metrics[7])
            
            # foot 
            loss4 = criterion1(out_data[4],data[6].argmax(dim=1))      
            y = tensor_max(out_data[4])
            metrics = tensor_metrics(data[6].float(),y)  
            ft_train.append(metrics[7])
            
            # gender
            loss5 = criterion2(out_data[5].squeeze(),data[7].float())
            y = tensor_thresh(out_data[5], 0.5)
            metrics = boolian_metrics(data[7].float(),y)  
            ft_train.append(metrics[3])
            
            # bags
            loss6 = criterion1(out_data[6],data[8].argmax(dim=1))
            y = tensor_max(out_data[6])
            metrics = tensor_metrics(data[8].float(),y)
            ft_train.append(metrics[7])
            
            # body colour
            loss7 = criterion1(out_data[7],data[9].argmax(dim=1))      
            y = tensor_max(out_data[7])
            metrics = tensor_metrics(data[9].float(),y)
            ft_train.append(metrics[7])
            
            # leg colour
            loss8 = criterion1(out_data[8],data[10].argmax(dim=1))
            y = tensor_max(out_data[8])
            metrics = tensor_metrics(data[10].float(),y)
            ft_train.append(metrics[7])
            
            # foot colour
            loss9 = criterion1(out_data[9],data[11].argmax(dim=1))
            y = tensor_max(out_data[9])
            metrics = tensor_metrics(data[11].float(),y)      
            ft_train.append(metrics[7])
            
            # id
            loss10 = criterion1(out_data[-1],data[1].argmax(dim=1))
            y = tensor_max(out_data[-1])
            metrics = tensor_metrics(data[1].float(),y)      
            ft_train.append(metrics[7])            
            
            # total loss
            loss = loss0+loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss10
            loss_e.append(loss.item())
            
            # backward step
            loss.backward()
            
            # optimization step
            optimizer.step()
            
            # print log
            if idx % 1 == 0:
                print('Train Epoch: {} [{}/{} , lr {}] \t Loss: {:.6f} , F1: {:.3f}'.format(
                    epoch, idx , len(train_loader),
                     optimizer.param_groups[0]['lr'],
                      loss.item(),np.mean(ft_train)))
       
        train_loss.append(np.mean(loss_e))
        F1_train.append(np.mean(ft_train))
        torch.save(train_loss,saving_path+'trainloss_'+version+'.pth')
        torch.save(F1_train,saving_path+'trainF1_'+version+'.pth')
        
        # evaluation:     
        attr_net.eval()
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                
                # data = data.to(device) 'list' object has no attribute 'to'
                # out_data = attr_net.predict(data[0].to(device))
                out_data = attr_net(data[0].to(device))
                for i in range(len(data)):
                    data[i] = data[i].to(device)
                # compute losses and evaluation metrics:
                    
                # head 
                loss0 = criterion1(out_data[0],data[2].argmax(dim=1))        
                y = tensor_max(out_data[0])
                metrics = tensor_metrics(data[2].float(),y)
                ft_test.append(metrics[7])
                
                # body
                loss1 = criterion1(out_data[1],data[3].argmax(dim=1))
                y = tensor_max(out_data[1])
                metrics = tensor_metrics(data[3].float(),y)
                ft_test.append(metrics[7])
                
                # body type
                loss2 = criterion2(out_data[2].squeeze(),data[4].float())    
                y = tensor_thresh(out_data[2], 0.5)
                metrics = boolian_metrics(data[4].float(),y)
                ft_test.append(metrics[3])
                
                # leg
                loss3 = criterion1(out_data[3],data[5].argmax(dim=1))
                y = tensor_max(out_data[3])
                metrics = tensor_metrics(data[5].float(),y)
                ft_test.append(metrics[7])
                
                # foot 
                loss4 = criterion1(out_data[4],data[6].argmax(dim=1))      
                y = tensor_max(out_data[4])
                metrics = tensor_metrics(data[6].float(),y)  
                ft_test.append(metrics[7])
                
                # gender
                loss5 = criterion2(out_data[5].squeeze(),data[7].float())
                y = tensor_thresh(out_data[5], 0.5)
                metrics = boolian_metrics(data[7].float(),y)  
                ft_test.append(metrics[3])
                
                # bags
                loss6 = criterion1(out_data[6],data[8].argmax(dim=1))
                y = tensor_max(out_data[6])
                metrics = tensor_metrics(data[8].float(),y)
                ft_test.append(metrics[7])
                
                # body colour
                loss7 = criterion1(out_data[7],data[9].argmax(dim=1))      
                y = tensor_thresh(out_data[7], 0.5)
                metrics = tensor_metrics(data[9].float(),y)
                ft_test.append(metrics[7])
                
                # leg colour
                loss8 = criterion1(out_data[8],data[10].argmax(dim=1))
                y = tensor_max(out_data[8])
                metrics = tensor_metrics(data[10].float(),y)
                ft_test.append(metrics[7])
                
                # foot colour
                loss9 = criterion1(out_data[9],data[11].argmax(dim=1))
                y = tensor_max(out_data[9])
                metrics = tensor_metrics(data[11].float(),y)      
                ft_test.append(metrics[7])

                # id
                loss10 = criterion1(out_data[-1],data[1].argmax(dim=1))
                y = tensor_max(out_data[-1])
                metrics = tensor_metrics(data[1].float(),y)      
                ft_train.append(metrics[7])          
                
                # total loss
                loss = loss0+loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9
                loss_t.append(loss.item())
        test_loss.append(np.mean(loss_t))
        F1_test.append(np.mean(ft_test))
        print('Epoch: {}\ntrain loss: {:.6f}\ntest loss: {:.6f}\n\nF1 train: {:.4f}\nF1 test: {:.4f} '.format(
                    epoch,train_loss[-1],test_loss[-1],F1_train[-1],F1_test[-1]))
        
        torch.save(test_loss,saving_path+'testloss_'+version+'.pth')
        torch.save(F1_test,saving_path+'testF1_'+version+'.pth')
        scheduler.step()
        if len(F1_test)>2: 
            if F1_test[-1] > F1_test[-2]:
                print('our net improved')
                torch.save(attr_net, saving_path+'attrnet_'+version+'_epoch'+str(epoch)+'.pth')
                torch.save(optimizer.state_dict(), saving_path+'optimizer_'+version+'_epoch'+str(epoch)+'.pth')
#%%
def train_collection_id2(num_epoch,
                     attr_net,
                     train_loader,
                     test_loader,
                     optimizer,
                     scheduler,
                     criterion1,
                     criterion2,
                     criterion3,
                     saving_path,
                     num_id,
                     device,
                     version,
                     resume=False,
                     loss_train = None,
                     loss_test=None,
                     train_F1=None,
                     test_F1=None,
                     stop_epoch=None): 
    # model_output (tuple): (out_head,out_body,out_body_type,out_leg,out_foot,out_gender,out_bags,out_body_colour,out_leg_colour,out_foot_colour,out_id)
    # loader_outpu (tuple): (img,id,head,body,body_type,leg,foot,gender,bags,body_colour,leg_colour,foot_colour)
    print('this is start')
    if resume:
        train_loss = loss_train
        test_loss = loss_test
        F1_train = train_F1
        F1_test = test_F1
    else:
        train_loss = []
        test_loss = []
        F1_train = []
        F1_test = []
    # gc.collect()
    # torch.cuda.empty_cache()
    # attr_net = attr_net.to(device)
    print('epoches started')
    if resume:
        start_epoch = stop_epoch+1
    else:
        start_epoch = 1    
    for epoch in range(start_epoch,num_epoch+1):
        
        # torch.cuda.empty_cache()
        attr_net = attr_net.to(device)        
        attr_net.train()
        loss_e = []
        loss_t = []
        ft_train = []
        ft_test = []
        
        for idx, data in enumerate(train_loader):
            
            # forward step
            optimizer.zero_grad()
            out_data = attr_net.forward(data[0].to(device))
            
            for i in range(len(data)):
                data[i] = data[i].to(device)
            # compute losses and evaluation metrics:
                
            # head 
            loss0 = criterion1(out_data[0],data[2].argmax(dim=1))        
            y = tensor_max(out_data[0])
            metrics = tensor_metrics(data[2].float(),y)
            ft_train.append(metrics[7])
            
            # body
            loss1 = criterion1(out_data[1],data[3].argmax(dim=1))
            y = tensor_max(out_data[1])
            metrics = tensor_metrics(data[3].float(),y)
            ft_train.append(metrics[7])
            
            # body type
            loss2 = criterion2(out_data[2].squeeze(),data[4].float())    
            y = tensor_thresh(out_data[2], 0.5)
            metrics = boolian_metrics(data[4].float(),y)
            ft_train.append(metrics[3])
            
            # leg
            loss3 = criterion1(out_data[3],data[5].argmax(dim=1))
            y = tensor_max(out_data[3])
            metrics = tensor_metrics(data[5].float(),y)
            ft_train.append(metrics[7])
            
            # foot 
            loss4 = criterion1(out_data[4],data[6].argmax(dim=1))      
            y = tensor_max(out_data[4])
            metrics = tensor_metrics(data[6].float(),y)  
            ft_train.append(metrics[7])
            
            # gender
            loss5 = criterion2(out_data[5].squeeze(),data[7].float())
            y = tensor_thresh(out_data[5], 0.5)
            metrics = boolian_metrics(data[7].float(),y)  
            ft_train.append(metrics[3])
            
            # bags
            loss6 = criterion1(out_data[6],data[8].argmax(dim=1))
            y = tensor_max(out_data[6])
            metrics = tensor_metrics(data[8].float(),y)
            ft_train.append(metrics[7])
            
            # body colour
            loss7 = criterion3(out_data[7],data[9])      
            y = tensor_thresh(out_data[7], 0.5)
            metrics = tensor_metrics(data[9].float(),y)
            ft_train.append(metrics[7])
            
            # leg colour
            loss8 = criterion1(out_data[8],data[10].argmax(dim=1))
            y = tensor_max(out_data[8])
            metrics = tensor_metrics(data[10].float(),y)
            ft_train.append(metrics[7])
            
            # foot colour
            loss9 = criterion3(out_data[9],data[11])
            y = tensor_max(out_data[9])
            metrics = tensor_metrics(data[11].float(),y)      
            ft_train.append(metrics[7])
            
            # # id
            # loss10 = criterion1(out_data[-1],data[1].argmax(dim=1))
            # y = tensor_max(out_data[-1])
            # metrics = tensor_metrics(data[1].float(),y)      
            # ft_train.append(metrics[7])            
            
            # total loss
            loss = loss0+loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8
            loss_e.append(loss.item())
            
            # backward step
            loss.backward()
            
            # optimization step
            optimizer.step()
            
            # print log
            if idx % 1 == 0:
                print('Train Epoch: {} [{}/{} , lr {}] \t Loss: {:.6f} , F1: {:.3f}'.format(
                    epoch, idx , len(train_loader),
                     optimizer.param_groups[0]['lr'],
                      loss.item(),np.mean(ft_train)))
       
        train_loss.append(np.mean(loss_e))
        F1_train.append(np.mean(ft_train))
        torch.save(train_loss,saving_path+'trainloss_'+version+'.pth')
        torch.save(F1_train,saving_path+'trainF1_'+version+'.pth')
        
        # evaluation:     
        attr_net.eval()
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                
                # data = data.to(device) 'list' object has no attribute 'to'
                # out_data = attr_net.predict(data[0].to(device))
                out_data = attr_net(data[0].to(device))
                for i in range(len(data)):
                    data[i] = data[i].to(device)
                # compute losses and evaluation metrics:
                    
                # head 
                loss0 = criterion1(out_data[0],data[2].argmax(dim=1))        
                y = tensor_max(out_data[0])
                metrics = tensor_metrics(data[2].float(),y)
                ft_test.append(metrics[7])
                
                # body
                loss1 = criterion1(out_data[1],data[3].argmax(dim=1))
                y = tensor_max(out_data[1])
                metrics = tensor_metrics(data[3].float(),y)
                ft_test.append(metrics[7])
                
                # body type
                loss2 = criterion2(out_data[2].squeeze(),data[4].float())    
                y = tensor_thresh(out_data[2], 0.5)
                metrics = boolian_metrics(data[4].float(),y)
                ft_test.append(metrics[3])
                
                # leg
                loss3 = criterion1(out_data[3],data[5].argmax(dim=1))
                y = tensor_max(out_data[3])
                metrics = tensor_metrics(data[5].float(),y)
                ft_test.append(metrics[7])
                
                # foot 
                loss4 = criterion1(out_data[4],data[6].argmax(dim=1))      
                y = tensor_max(out_data[4])
                metrics = tensor_metrics(data[6].float(),y)  
                ft_test.append(metrics[7])
                
                # gender
                loss5 = criterion2(out_data[5].squeeze(),data[7].float())
                y = tensor_thresh(out_data[5], 0.5)
                metrics = boolian_metrics(data[7].float(),y)  
                ft_test.append(metrics[3])
                
                # bags
                loss6 = criterion1(out_data[6],data[8].argmax(dim=1))
                y = tensor_max(out_data[6])
                metrics = tensor_metrics(data[8].float(),y)
                ft_test.append(metrics[7])
                
                # body colour
                loss7 = criterion3(out_data[7],data[9])      
                y = tensor_thresh(out_data[7], 0.5)
                metrics = tensor_metrics(data[9].float(),y)
                ft_test.append(metrics[7])
                
                # leg colour
                loss8 = criterion1(out_data[8],data[10].argmax(dim=1))
                y = tensor_max(out_data[8])
                metrics = tensor_metrics(data[10].float(),y)
                ft_test.append(metrics[7])
                
                # foot colour
                loss9 = criterion3(out_data[9],data[11])
                y = tensor_max(out_data[9])
                metrics = tensor_metrics(data[11].float(),y)      
                ft_test.append(metrics[7])

                # # id
                # loss10 = criterion1(out_data[-1],data[1].argmax(dim=1))
                # y = tensor_max(out_data[-1])
                # metrics = tensor_metrics(data[1].float(),y)      
                # ft_train.append(metrics[7])          
                
                # total loss
                loss = loss0+loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9
                loss_t.append(loss.item())
        test_loss.append(np.mean(loss_t))
        F1_test.append(np.mean(ft_test))
        print('Epoch: {}\ntrain loss: {:.6f}\ntest loss: {:.6f}\n\nF1 train: {:.4f}\nF1 test: {:.4f} '.format(
                    epoch,train_loss[-1],test_loss[-1],F1_train[-1],F1_test[-1]))
        
        torch.save(test_loss,saving_path+'testloss_'+version+'.pth')
        torch.save(F1_test,saving_path+'testF1_'+version+'.pth')
        scheduler.step()
        if len(F1_test)>2: 
            if F1_test[-1] > F1_test[-2]:
                print('our net improved')
                torch.save(attr_net, saving_path+'attrnet_'+version+'_epoch'+str(epoch)+'.pth')
                torch.save(optimizer.state_dict(), saving_path+'optimizer_'+version+'_epoch'+str(epoch)+'.pth')
#%%
def train_collection_id_weight(num_epoch,
                     attr_net,
                     train_loader,
                     test_loader,
                     optimizer,
                     scheduler,
                     criterion_head,
                     criterion_body,
                     criterion_body_type,
                     criterion_leg,
                     criterion_foot,
                     criterion_gender,
                     criterion_bags,
                     criterion_body_colour,
                     criterion_foot_colour,
                     criterion_leg_colour,
                     saving_path,                    
                     device,
                     version,
                     resume=False,
                     loss_train = None,
                     loss_test=None,
                     train_F1=None,
                     test_F1=None,
                     stop_epoch=None): 
    # model_output (tuple): (out_head,out_body,out_body_type,out_leg,out_foot,out_gender,out_bags,out_body_colour,out_leg_colour,out_foot_colour,out_id)
    # loader_outpu (tuple): (img,id,head,body,body_type,leg,foot,gender,bags,body_colour,leg_colour,foot_colour)
    print('this is start')
    f1_best = 0
    if resume:
        train_loss = loss_train
        test_loss = loss_test
        F1_train = train_F1
        F1_test = test_F1
    else:
        train_loss = []
        test_loss = []
        F1_train = []
        F1_test = []
    # gc.collect()
    # torch.cuda.empty_cache()
    # attr_net = attr_net.to(device)
    print('epoches started')
    if resume:
        start_epoch = stop_epoch+1
    else:
        start_epoch = 1    
    for epoch in range(start_epoch,num_epoch+1):
        
        # torch.cuda.empty_cache()
        attr_net = attr_net.to(device)        
        attr_net.train()
        loss_e = []
        loss_t = []
        ft_train = []
        ft_test = []
        
        for idx, data in enumerate(train_loader):
            
            # forward step
            optimizer.zero_grad()
            out_data = attr_net.forward(data[0].to(device))
            
            for i in range(len(data)):
                data[i] = data[i].to(device)
            # compute losses and evaluation metrics:
                
            # head 
            loss0 = criterion_head(out_data[0],data[2].argmax(dim=1))  
            # o = torch.softmax(out_data[0], dim=1)
            # y = tensor_max(o)
            y = tensor_max(out_data[0])
            metrics = tensor_metrics(data[2].float(),y)
            ft_train.append(metrics[7])
            
            # body
            loss1 = criterion_body(out_data[1],data[3].argmax(dim=1))
            # o = torch.softmax(out_data[1], dim=1)
            # y = tensor_max(o)
            y = tensor_max(out_data[1])
            metrics = tensor_metrics(data[3].float(),y)
            ft_train.append(metrics[7])
            
            # body type
            loss2 = criterion_body_type(out_data[2].squeeze(),data[4].float())   
            # o = torch.sigmoid(out_data[2])
            # y = tensor_thresh(o, 0.5)
            y = tensor_thresh(out_data[2], 0)
            metrics = boolian_metrics(data[4].float(),y)
            ft_train.append(metrics[3])
            
            # leg
            loss3 = criterion_leg(out_data[3],data[5].argmax(dim=1))
            # o = torch.softmax(out_data[3], dim=1)
            # y = tensor_max(o)
            y = tensor_max(out_data[3])
            metrics = tensor_metrics(data[5].float(),y)
            ft_train.append(metrics[7])
            
            # foot 
            loss4 = criterion_foot(out_data[4],data[6].argmax(dim=1))  
            # o = torch.softmax(out_data[4], dim=1)
            # y = tensor_max(o)
            y = tensor_max(out_data[4])
            metrics = tensor_metrics(data[6].float(),y)  
            ft_train.append(metrics[7])
            
            # gender
            loss5 = criterion_gender(out_data[5].squeeze(),data[7].float())
            # o = torch.sigmoid(out_data[5])
            # y = tensor_thresh(o, 0.5)
            y = tensor_thresh(out_data[5], 0)
            metrics = boolian_metrics(data[7].float(),y)  
            ft_train.append(metrics[3])
            
            # bags
            loss6 = criterion_bags(out_data[6],data[8].argmax(dim=1))
            # o = torch.softmax(out_data[6], dim=1)
            # y = tensor_max(o)
            y = tensor_max(out_data[6])
            metrics = tensor_metrics(data[8].float(),y)
            ft_train.append(metrics[7])
            
            # body colour
            loss7 = criterion_body_colour(out_data[7],data[9].float())  
            # o = torch.sigmoid(out_data[7])
            # y = tensor_thresh(o, 0.5)
            y = tensor_thresh(out_data[7], 0)
            metrics = tensor_metrics(data[9].float(),y)
            ft_train.append(metrics[7])
            
            # leg colour
            loss8 = criterion_leg_colour(out_data[8],data[10].argmax(dim=1))
            # o = torch.softmax(out_data[8], dim=1)
            # y = tensor_max(o)
            y = tensor_max(out_data[8])
            metrics = tensor_metrics(data[10].float(),y)
            ft_train.append(metrics[7])
            
            # foot colour
            loss9 = criterion_foot_colour(out_data[9],data[11].argmax(dim=1))
            # o = torch.softmax(out_data[9], dim=1)
            # y = tensor_max(o)
            y = tensor_max(out_data[9])
            metrics = tensor_metrics(data[11].float(),y)      
            ft_train.append(metrics[7])
            
            # # id
            # loss10 = criterion1(out_data[-1],data[1].argmax(dim=1))
            # y = tensor_max(out_data[-1])
            # metrics = tensor_metrics(data[1].float(),y)      
            # ft_train.append(metrics[7])            
            
            # total loss
            loss = loss0+loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8
            loss_e.append(loss.item())
            
            # backward step
            loss.backward()
            
            # optimization step
            optimizer.step()
            
            # print log
            if idx % 1 == 0:
                print('Train Epoch: {} [{}/{} , lr {}] \t Loss: {:.6f} , F1: {:.3f}'.format(
                    epoch, idx , len(train_loader),
                     optimizer.param_groups[0]['lr'],
                      loss.item(),np.mean(ft_train)))
       
        train_loss.append(np.mean(loss_e))
        F1_train.append(np.mean(ft_train))
        torch.save(train_loss,saving_path+'trainloss_'+version+'.pth')
        torch.save(F1_train,saving_path+'trainF1_'+version+'.pth')
        
        # evaluation:     
        attr_net.eval()
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                
                # data = data.to(device) 'list' object has no attribute 'to'
                # out_data = attr_net.predict(data[0].to(device))
                out_data = attr_net(data[0].to(device))
                for i in range(len(data)):
                    data[i] = data[i].to(device)
                # compute losses and evaluation metrics:
                    
                # head 
                loss0 = criterion_head(out_data[0],data[2].argmax(dim=1))        
                y = tensor_max(out_data[0])
                metrics = tensor_metrics(data[2].float(),y)
                ft_test.append(metrics[7])
                
                # body
                loss1 = criterion_body(out_data[1],data[3].argmax(dim=1))
                y = tensor_max(out_data[1])
                metrics = tensor_metrics(data[3].float(),y)
                ft_test.append(metrics[7])
                
                # body type
                loss2 = criterion_body_type(out_data[2].squeeze(),data[4].float())    
                y = tensor_thresh(out_data[2], 0)
                metrics = boolian_metrics(data[4].float(),y)
                ft_test.append(metrics[3])
                
                # leg
                loss3 = criterion_leg(out_data[3],data[5].argmax(dim=1))
                y = tensor_max(out_data[3])
                metrics = tensor_metrics(data[5].float(),y)
                ft_test.append(metrics[7])
                
                # foot 
                loss4 = criterion_foot(out_data[4],data[6].argmax(dim=1))      
                y = tensor_max(out_data[4])
                metrics = tensor_metrics(data[6].float(),y)  
                ft_test.append(metrics[7])
                
                # gender
                loss5 = criterion_gender(out_data[5].squeeze(),data[7].float())
                y = tensor_thresh(out_data[5], 0)
                metrics = boolian_metrics(data[7].float(),y)  
                ft_test.append(metrics[3])
                
                # bags
                loss6 = criterion_bags(out_data[6],data[8].argmax(dim=1))
                y = tensor_max(out_data[6])
                metrics = tensor_metrics(data[8].float(),y)
                ft_test.append(metrics[7])
                
                # body colour
                loss7 = criterion_body_colour(out_data[7],data[9].float())      
                y = tensor_thresh(out_data[7], 0)
                metrics = tensor_metrics(data[9].float(),y)
                ft_test.append(metrics[7])
                
                # leg colour
                loss8 = criterion_leg_colour(out_data[8],data[10].argmax(dim=1))
                y = tensor_max(out_data[8])
                metrics = tensor_metrics(data[10].float(),y)
                ft_test.append(metrics[7])
                
                # foot colour
                loss9 = criterion_foot_colour(out_data[9],data[11].argmax(dim=1))
                y = tensor_max(out_data[9])
                metrics = tensor_metrics(data[11].float(),y)      
                ft_test.append(metrics[7])

                # # id
                # loss10 = criterion1(out_data[-1],data[1].argmax(dim=1))
                # y = tensor_max(out_data[-1])
                # metrics = tensor_metrics(data[1].float(),y)      
                # ft_train.append(metrics[7])          
                
                # total loss
                loss = loss0+loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9
                loss_t.append(loss.item())
        test_loss.append(np.mean(loss_t))
        F1_test.append(np.mean(ft_test))
        print('Epoch: {}\ntrain loss: {:.6f}\ntest loss: {:.6f}\n\nF1 train: {:.4f}\nF1 test: {:.4f} '.format(
                    epoch,train_loss[-1],test_loss[-1],F1_train[-1],F1_test[-1]))
        
        torch.save(test_loss,saving_path+'testloss_'+version+'.pth')
        torch.save(F1_test,saving_path+'testF1_'+version+'.pth')
        scheduler.step()
        if F1_test[-1]>f1_best: 
            f1_best = F1_test[-1]
            print('our net improved')
            torch.save(attr_net, saving_path+'attrnet_'+version+'.pth')
            torch.save(optimizer.state_dict(), saving_path+'attrnet_'+version+'.pth')

#%%
def dict_training(num_epoch,
                     attr_net,
                     train_loader,
                     test_loader,
                     optimizer,
                     scheduler,
                     cce_loss,
                     bce_loss,
                     save_path,                    
                     device,
                     version,
                     need_attr = False,
                     need_id = True,
                     need_collection = True,
                     resume=False,
                     loss_train = None,
                     loss_test=None,
                     train_attr_F1=None,
                     test_attr_F1=None,
                     train_attr_acc=None,
                     test_attr_acc=None,  
                     train_id_acc=None,
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
        Acc_id_train = train_id_acc
    else:
        train_loss = []
        test_loss = []
        
        # attributes metrics lists 
        F1_train = []
        F1_test = []
        Acc_train = []
        Acc_test = []
        
        # id metrics lists
        Acc_id_train = []      

    print('epoches started')
    if resume:
        start_epoch = stoped_epoch+1
    else:
        start_epoch = 1    
    for epoch in range(start_epoch,num_epoch+1):
        
        # torch.cuda.empty_cache()
        attr_net = attr_net.to(device)        
        attr_net.train()
        loss_e = []
        loss_t = []
        
        # attributes temporary metrics lists 
        ft_train = []
        ft_test = []
        acc_train = []
        acc_test = []
        
        # ids temporary metrics lists 
        acc_id_train = []    
        
        for idx, data in enumerate(train_loader):
            
            for key, _ in data.items():
                data[key] = data[key].to(device)
            # forward step
            optimizer.zero_grad()
            out_data = attr_net.forward(data['img'])   
            
            if need_id:
                # id
                loss10 = cce_loss(out_data['id'],data['id'].argmax(dim=1))
                y_id = tensor_max(softmax(out_data['id']))  
                train_id_metrics = category_metrics(data['id'].float(), y_id)
                acc_id_train.append(train_id_metrics)     
            
            if not need_attr:
            
                # compute losses and evaluation metrics:
                # head 
                loss0 = cce_loss(out_data['head'], data['head'].argmax(dim=1))  
                y_head = tensor_max(softmax(out_data['head']))
                
                # body
                loss1 = cce_loss(out_data['body'],data['body'].argmax(dim=1))
                y_body = tensor_max(softmax(out_data['body']))
                
                # body_type
                loss2 = bce_loss(out_data['body_type'].squeeze(),data['body_type'].float())   
                y_body_type = tensor_thresh(torch.sigmoid(out_data['body_type']), 0.5)
                
                # leg
                loss3 = cce_loss(out_data['leg'],data['leg'].argmax(dim=1))
                y_leg = tensor_max(softmax(out_data['leg']))
                
                # foot 
                loss4 = cce_loss(out_data['foot'],data['foot'].argmax(dim=1))  
                y_foot = tensor_max(softmax(out_data['foot']))
                
                # gender
                loss5 = bce_loss(out_data['gender'].squeeze(),data['gender'].float())
                y_gender = tensor_thresh(torch.sigmoid(out_data['gender']), 0.5)
                
                # bags
                loss6 = cce_loss(out_data['bags'],data['bags'].argmax(dim=1))
                y_bags = tensor_max(softmax(out_data['bags']))
                
                # body_colour
                loss7 = bce_loss(out_data['body_colour'],data['body_colour'].float())  
                y_body_colour = tensor_thresh(torch.sigmoid(out_data['body_colour']), 0.5)
                
                # leg_colour
                loss8 = cce_loss(out_data['leg_colour'],data['leg_colour'].argmax(dim=1))
                y_leg_colour = tensor_max(softmax(out_data['leg_colour']))
                
                # foot_colour
                loss9 = cce_loss(out_data['foot_colour'],data['foot_colour'].argmax(dim=1))
                y_foot_colour = tensor_max(softmax(out_data['foot_colour']))
                
                y_attr = torch.cat((y_gender, y_head, y_body, 
                                    y_body_type, y_body_colour,
                                    y_bags, y_leg, y_leg_colour,
                                    y_foot, y_foot_colour), dim=1)
                    
                y_target = torch.cat((data['gender'].unsqueeze(dim=1), data['head'],
                                      data['body'], data['body_type'].unsqueeze(dim=1),
                                      data['body_colour'], data['bags'],
                                      data['leg'], data['leg_colour'],
                                      data['foot'], data['foot_colour']), dim=1)       
                if need_id:
                    # total loss
                    loss = loss0+loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9+loss10
                else:
                    loss = loss0+loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9               
            else:
                loss1 = bce_loss(out_data['attr'], data['attr'].float())
                y_attr = tensor_thresh(torch.sigmoid(out_data['attr']), 0.5)
                y_target = data['attr']
                if need_id:
                    loss = loss1 + loss10
                else:
                    loss = loss1
                
            train_attr_metrics = tensor_metrics(y_target.float(), y_attr)
            
            ft_train.append(train_attr_metrics[-2])
            acc_train.append(train_attr_metrics[-3])
            loss_e.append(loss.item())
            
            # backward step
            loss.backward()
            # optimization step
            optimizer.step()
            
            if need_id:
                # print log
                if idx % 1 == 0:
                    print('\nTrain Epoch: {} [{}/{} , lr {}] \t Loss: {:.6f} \nattr_metrics: F1_attr: {:.3f} acc_attr{:.3f} \nid_metrics: acc_id{:.3f}'.format(
                        epoch, idx , len(train_loader),
                         optimizer.param_groups[0]['lr'],
                          loss.item(),ft_train[-1], acc_train[-1],
                          acc_id_train[-1]))
            else:
                # print log
                if idx % 1 == 0:
                    print('\nTrain Epoch: {} [{}/{} , lr {}] \t Loss: {:.6f} \nattr_metrics: F1_attr: {:.3f} acc_attr{:.3f} \n'.format(
                        epoch, idx , len(train_loader),
                         optimizer.param_groups[0]['lr'],
                          loss.item(),ft_train[-1], acc_train[-1]))                
       
        train_loss.append(np.mean(loss_e))
        F1_train.append(np.mean(ft_train))
        Acc_train.append(np.mean(acc_train))
        if need_id:
            Acc_id_train.append(np.mean(acc_id_train))
        
        # evaluation:     
        attr_net.eval()
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                
                for key, _ in data.items():
                    data[key] = data[key].to(device)
                    
                # forward step
                out_data = attr_net.forward(data['img'])           
                
                if not need_attr:
                    # compute losses and evaluation metrics:
                    # head 
                    loss0 = cce_loss(out_data['head'], data['head'].argmax(dim=1))  
                    y_head = tensor_max(softmax(out_data['head']))
                    
                    # body
                    loss1 = cce_loss(out_data['body'],data['body'].argmax(dim=1))
                    y_body = tensor_max(softmax(out_data['body']))
                    
                    # body_type
                    loss2 = bce_loss(out_data['body_type'].squeeze(),data['body_type'].float())   
                    y_body_type = tensor_thresh(torch.sigmoid(out_data['body_type']), 0.5)
                    
                    # leg
                    loss3 = cce_loss(out_data['leg'],data['leg'].argmax(dim=1))
                    y_leg = tensor_max(softmax(out_data['leg']))
                    
                    # foot 
                    loss4 = cce_loss(out_data['foot'],data['foot'].argmax(dim=1))  
                    y_foot = tensor_max(softmax(out_data['foot']))
                    
                    # gender
                    loss5 = bce_loss(out_data['gender'].squeeze(),data['gender'].float())
                    y_gender = tensor_thresh(torch.sigmoid(out_data['gender']), 0.5)
                    
                    # bags
                    loss6 = cce_loss(out_data['bags'],data['bags'].argmax(dim=1))
                    y_bags = tensor_max(softmax(out_data['bags']))
                    
                    # body_colour
                    loss7 = bce_loss(out_data['body_colour'],data['body_colour'].float())  
                    y_body_colour = tensor_thresh(torch.sigmoid(out_data['body_colour']), 0.5)
                    
                    # leg_colour
                    loss8 = cce_loss(out_data['leg_colour'],data['leg_colour'].argmax(dim=1))
                    y_leg_colour = tensor_max(softmax(out_data['leg_colour']))
                    
                    # foot_colour
                    loss9 = cce_loss(out_data['foot_colour'],data['foot_colour'].argmax(dim=1))
                    y_foot_colour = tensor_max(softmax(out_data['foot_colour']))
                    
                    y_attr = torch.cat((y_gender, y_head, y_body, 
                                        y_body_type, y_body_colour,
                                        y_bags, y_leg, y_leg_colour,
                                        y_foot, y_foot_colour), dim=1)
                        
                    y_target = torch.cat((data['gender'].unsqueeze(dim=1), data['head'],
                                          data['body'], data['body_type'].unsqueeze(dim=1),
                                          data['body_colour'], data['bags'],
                                          data['leg'], data['leg_colour'],
                                          data['foot'], data['foot_colour']), dim=1)                                            

                    loss = loss0+loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9
                    
                else:
                    loss1 = bce_loss(out_data['attr'], data['attr'].float())
                    y_attr = tensor_thresh(torch.sigmoid(out_data['attr']))
                    y_target = data['attr']
                    loss = loss1      
                    
                test_attr_metrics = tensor_metrics(y_target.float(), y_attr)
                ft_test.append(test_attr_metrics[-2])
                acc_test.append(test_attr_metrics[-3]) 
                loss_t.append(loss.item())
                
        test_loss.append(np.mean(loss_t))
        F1_test.append(np.mean(ft_test))
        Acc_test.append(np.mean(acc_test))
        print('Epoch: {}\ntrain loss: {:.6f}\ntest loss: {:.6f}\n\nF1 train: {:.4f}\nF1 test: {:.4f}\n\nacc_train: {:.4f}\nacc_test: {:.4f} '.format(
                    epoch,train_loss[-1],test_loss[-1],F1_train[-1],F1_test[-1],Acc_train[-1],Acc_test[-1]))

        scheduler.step()
        
        # saving training results
        saving_path = os.path.join(save_path, version)
        if os.path.exists(saving_path):
            
            torch.save(train_loss, os.path.join(saving_path, 'train_loss.pth'))
            torch.save(F1_train, os.path.join(saving_path, 'train_attr_f1.pth'))
            torch.save(Acc_train, os.path.join(saving_path, 'train_attr_acc.pth'))
            if need_id:
                torch.save(Acc_id_train, os.path.join(saving_path, 'train_id_acc.pth'))
                       
            torch.save(test_loss, os.path.join(saving_path, 'test_attr_loss.pth'))
            torch.save(F1_test, os.path.join(saving_path, 'test_attr_f1.pth'))
            torch.save(Acc_test, os.path.join(saving_path, 'test_attr_acc.pth'))           

            torch.save(attr_net, os.path.join(saving_path, 'attr_net.pth'))
            torch.save(optimizer.state_dict(), os.path.join(saving_path, 'optimizer.pth')) 
            torch.save(scheduler.state_dict(), os.path.join(saving_path, 'scheduler.pth'))
            torch.save(epoch, os.path.join(saving_path, 'training_epoch.pth'))
        else:
            os.mkdir(saving_path)
            
            torch.save(train_loss, os.path.join(saving_path, 'train_loss.pth'))
            torch.save(F1_train, os.path.join(saving_path, 'train_f1.pth'))
            torch.save(Acc_train, os.path.join(saving_path, 'train_acc.pth'))
            if need_id:
                torch.save(Acc_id_train, os.path.join(saving_path, 'train_id_acc.pth'))
                       
            torch.save(test_loss, os.path.join(saving_path, 'test_loss.pth'))
            torch.save(F1_test, os.path.join(saving_path, 'test_f1.pth'))
            torch.save(Acc_test, os.path.join(saving_path, 'test_acc.pth'))           

            torch.save(attr_net, os.path.join(saving_path, 'attr_net.pth'))
            torch.save(optimizer.state_dict(), os.path.join(saving_path, 'optimizer.pth')) 
            torch.save(scheduler.state_dict(), os.path.join(saving_path, 'scheduler.pth'))
            torch.save(epoch, os.path.join(saving_path, 'training_epoch.pth'))
            
        
        d = 0
        if test_loss[-1] < loss_min: 
            loss_min = test_loss[-1]
            d += 1
            print('test loss improved')
            
        p = 0     
        if F1_test[-1] > f1_best: 
            f1_best = F1_test[-1]
            p += 1
            torch.save(attr_net, os.path.join(saving_path, 'best_attr_net.pth'))
            print('test f1 improved')
                
#%%
def dict_training_fastreid(num_epoch,
                     attr_net,
                     train_loader,
                     test_loader,
                     optimizer,
                     scheduler,
                     cce_loss,
                     bce_loss,
                     save_path,                    
                     device,
                     version,
                     need_attr = False,
                     need_id = True,
                     need_collection = True,
                     resume=False,
                     loss_train = None,
                     loss_test=None,
                     train_attr_F1=None,
                     test_attr_F1=None,
                     train_attr_acc=None,
                     test_attr_acc=None,  
                     train_id_acc=None,
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
        Acc_id_train = train_id_acc
    else:
        train_loss = []
        test_loss = []
        
        # attributes metrics lists 
        F1_train = []
        F1_test = []
        Acc_train = []
        Acc_test = []
        
        # id metrics lists
        Acc_id_train = []      

    print('epoches started')
    if resume:
        start_epoch = stoped_epoch+1
    else:
        start_epoch = 1    
    for epoch in range(start_epoch,num_epoch+1):
        
        # torch.cuda.empty_cache()
        attr_net = attr_net.to(device)        
        attr_net.train()
        loss_e = []
        loss_t = []
        
        # attributes temporary metrics lists 
        ft_train = []
        ft_test = []
        acc_train = []
        acc_test = []
        
        # ids temporary metrics lists 
        acc_id_train = []    
        
        for idx, data in enumerate(train_loader):
            
            for key, _ in data.items():
                data[key] = data[key].to(device)
            # forward step
            optimizer.zero_grad()
            out_data = attr_net.forward(data['img'])   
            
            # id
            loss10 = cce_loss(out_data['id'],data['id'].argmax(dim=1))
            y_id = tensor_max(softmax(out_data['id']))  
            train_id_metrics = category_metrics(data['id'].float(), y_id)
            acc_id_train.append(train_id_metrics)     
            
            if not need_attr:
            
                # compute losses and evaluation metrics:
                # head 
                loss0 = cce_loss(out_data['head'], data['head'].argmax(dim=1))  
                y_head = tensor_max(softmax(out_data['head']))
                
                # body
                loss1 = cce_loss(out_data['body'],data['body'].argmax(dim=1))
                y_body = tensor_max(softmax(out_data['body']))
                
                # body_type
                loss2 = bce_loss(out_data['body_type'].squeeze(),data['body_type'].float())   
                y_body_type = tensor_thresh(torch.sigmoid(out_data['body_type']), 0.5)
                
                # leg
                loss3 = cce_loss(out_data['leg'],data['leg'].argmax(dim=1))
                y_leg = tensor_max(softmax(out_data['leg']))
                
                # foot 
                loss4 = cce_loss(out_data['foot'],data['foot'].argmax(dim=1))  
                y_foot = tensor_max(softmax(out_data['foot']))
                
                # gender
                loss5 = bce_loss(out_data['gender'].squeeze(),data['gender'].float())
                y_gender = tensor_thresh(torch.sigmoid(out_data['gender']), 0.5)
                
                # bags
                loss6 = cce_loss(out_data['bags'],data['bags'].argmax(dim=1))
                y_bags = tensor_max(softmax(out_data['bags']))
                
                # body_colour
                loss7 = bce_loss(out_data['body_colour'],data['body_colour'].float())  
                y_body_colour = tensor_thresh(torch.sigmoid(out_data['body_colour']), 0.5)
                
                # leg_colour
                loss8 = cce_loss(out_data['leg_colour'],data['leg_colour'].argmax(dim=1))
                y_leg_colour = tensor_max(softmax(out_data['leg_colour']))
                
                # foot_colour
                loss9 = cce_loss(out_data['foot_colour'],data['foot_colour'].argmax(dim=1))
                y_foot_colour = tensor_max(softmax(out_data['foot_colour']))
                
                y_attr = torch.cat((y_gender, y_head, y_body, 
                                    y_body_type, y_body_colour,
                                    y_bags, y_leg, y_leg_colour,
                                    y_foot, y_foot_colour), dim=1)
                    
                y_target = torch.cat((data['gender'].unsqueeze(dim=1), data['head'],
                                      data['body'], data['body_type'].unsqueeze(dim=1),
                                      data['body_colour'], data['bags'],
                                      data['leg'], data['leg_colour'],
                                      data['foot'], data['foot_colour']), dim=1)                
                # total loss
                loss = loss0+loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9+loss10
                
            else:
                loss1 = bce_loss(out_data['attr'], data['attr'].float())
                y_attr = tensor_thresh(torch.sigmoid(out_data['attr']), 0.5)
                y_target = data['attr']
                loss = loss1 + loss10
                
            train_attr_metrics = tensor_metrics(y_target.float(), y_attr)
            
            ft_train.append(train_attr_metrics[7])
            acc_train.append(train_attr_metrics[6])
            loss_e.append(loss.item())
            
            # backward step
            loss.backward()
            # optimization step
            optimizer.step()
            
            # print log
            if idx % 1 == 0:
                print('\nTrain Epoch: {} [{}/{} , lr {}] \t Loss: {:.6f} \nattr_metrics: F1_attr: {:.3f} acc_attr{:.3f} \nid_metrics: acc_id{:.3f}'.format(
                    epoch, idx , len(train_loader),
                     optimizer.param_groups[0]['lr'],
                      loss.item(),ft_train[-1], acc_train[-1],
                      acc_id_train[-1]))
       
        train_loss.append(np.mean(loss_e))
        F1_train.append(np.mean(ft_train))
        Acc_train.append(np.mean(acc_train))
        Acc_id_train.append(np.mean(acc_id_train))
        
        # evaluation:     
        attr_net.eval()
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                
                for key, _ in data.items():
                    data[key] = data[key].to(device)
                    
                # forward step
                out_data = attr_net.forward(data['img'])           
                
                if not need_attr:
                    # compute losses and evaluation metrics:
                    # head 
                    loss0 = cce_loss(out_data['head'], data['head'].argmax(dim=1))  
                    y_head = tensor_max(softmax(out_data['head']))
                    
                    # body
                    loss1 = cce_loss(out_data['body'],data['body'].argmax(dim=1))
                    y_body = tensor_max(softmax(out_data['body']))
                    
                    # body_type
                    loss2 = bce_loss(out_data['body_type'].squeeze(),data['body_type'].float())   
                    y_body_type = tensor_thresh(torch.sigmoid(out_data['body_type']), 0.5)
                    
                    # leg
                    loss3 = cce_loss(out_data['leg'],data['leg'].argmax(dim=1))
                    y_leg = tensor_max(softmax(out_data['leg']))
                    
                    # foot 
                    loss4 = cce_loss(out_data['foot'],data['foot'].argmax(dim=1))  
                    y_foot = tensor_max(softmax(out_data['foot']))
                    
                    # gender
                    loss5 = bce_loss(out_data['gender'].squeeze(),data['gender'].float())
                    y_gender = tensor_thresh(torch.sigmoid(out_data['gender']), 0.5)
                    
                    # bags
                    loss6 = cce_loss(out_data['bags'],data['bags'].argmax(dim=1))
                    y_bags = tensor_max(softmax(out_data['bags']))
                    
                    # body_colour
                    loss7 = bce_loss(out_data['body_colour'],data['body_colour'].float())  
                    y_body_colour = tensor_thresh(torch.sigmoid(out_data['body_colour']), 0.5)
                    
                    # leg_colour
                    loss8 = cce_loss(out_data['leg_colour'],data['leg_colour'].argmax(dim=1))
                    y_leg_colour = tensor_max(softmax(out_data['leg_colour']))
                    
                    # foot_colour
                    loss9 = cce_loss(out_data['foot_colour'],data['foot_colour'].argmax(dim=1))
                    y_foot_colour = tensor_max(softmax(out_data['foot_colour']))
                    
                    y_attr = torch.cat((y_gender, y_head, y_body, 
                                        y_body_type, y_body_colour,
                                        y_bags, y_leg, y_leg_colour,
                                        y_foot, y_foot_colour), dim=1)
                        
                    y_target = torch.cat((data['gender'].unsqueeze(dim=1), data['head'],
                                          data['body'], data['body_type'].unsqueeze(dim=1),
                                          data['body_colour'], data['bags'],
                                          data['leg'], data['leg_colour'],
                                          data['foot'], data['foot_colour']), dim=1)                                            
                    # total loss
                    loss = loss0+loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9
                    
                else:
                    loss1 = bce_loss(out_data['attr'], data['attr'].float())
                    y_attr = tensor_thresh(torch.sigmoid(out_data['attr']))
                    y_target = data['attr']
                    loss = loss1 + loss10      
                    
                test_attr_metrics = tensor_metrics(y_target.float(), y_attr)
                ft_test.append(test_attr_metrics[7])
                acc_test.append(test_attr_metrics[6]) 
                loss_t.append(loss.item())
                
        test_loss.append(np.mean(loss_t))
        F1_test.append(np.mean(ft_test))
        Acc_test.append(np.mean(acc_test))
        print('Epoch: {}\ntrain loss: {:.6f}\ntest loss: {:.6f}\n\nF1 train: {:.4f}\nF1 test: {:.4f}\n\nacc_train: {:.4f}\nacc_test: {:.4f} '.format(
                    epoch,train_loss[-1],test_loss[-1],F1_train[-1],F1_test[-1],Acc_train[-1],Acc_test[-1]))

        scheduler.step()
        
        # saving training results
        saving_path = os.path.join(save_path, version)
        if os.path.exists(saving_path):
            
            torch.save(train_loss, os.path.join(saving_path, 'train_loss.pth'))
            torch.save(F1_train, os.path.join(saving_path, 'train_attr_f1.pth'))
            torch.save(Acc_train, os.path.join(saving_path, 'train_attr_acc.pth'))
            torch.save(Acc_id_train, os.path.join(saving_path, 'train_id_acc.pth'))
                       
            torch.save(test_loss, os.path.join(saving_path, 'test_attr_loss.pth'))
            torch.save(F1_test, os.path.join(saving_path, 'test_attr_f1.pth'))
            torch.save(Acc_test, os.path.join(saving_path, 'test_attr_acc.pth'))           

            torch.save(attr_net, os.path.join(saving_path, 'attr_net.pth'))
            torch.save(optimizer.state_dict(), os.path.join(saving_path, 'optimizer.pth')) 
            torch.save(scheduler.state_dict(), os.path.join(saving_path, 'scheduler.pth'))
            torch.save(epoch, os.path.join(saving_path, 'training_epoch.pth'))
            attr_net.save_baseline(saving_path, 'baseline')
            
        else:
            os.mkdir(saving_path)
            
            torch.save(train_loss, os.path.join(saving_path, 'train_loss.pth'))
            torch.save(F1_train, os.path.join(saving_path, 'train_attr_f1.pth'))
            torch.save(Acc_train, os.path.join(saving_path, 'train_attr_acc.pth'))
            torch.save(Acc_id_train, os.path.join(saving_path, 'train_id_acc.pth'))
                       
            torch.save(test_loss, os.path.join(saving_path, 'test_attr_loss.pth'))
            torch.save(F1_test, os.path.join(saving_path, 'test_attr_f1.pth'))
            torch.save(Acc_test, os.path.join(saving_path, 'test_attr_acc.pth'))           

            torch.save(attr_net, os.path.join(saving_path, 'attr_net.pth'))
            torch.save(optimizer.state_dict(), os.path.join(saving_path, 'optimizer.pth')) 
            torch.save(scheduler.state_dict(), os.path.join(saving_path, 'scheduler.pth'))
            torch.save(epoch, os.path.join(saving_path, 'training_epoch.pth'))
            
        
        d = 0
        if test_loss[-1] < loss_min: 
            loss_min = test_loss[-1]
            d += 1
            torch.save(attr_net, os.path.join(saving_path, 'best_attr_net.pth'))
            torch.save(epoch, os.path.join(saving_path, 'best_epoch.pth'))
            attr_net.save_baseline(saving_path, 'best_baseline')
            print('test loss improved')
            
        if F1_test[-1] > f1_best: 
            f1_best = F1_test[-1]
            if d ==0:
                print('test f1 improved')
            else:
                print('\ntest f1 improved')
                

#%%
def dict_training_multi_branch(num_epoch,
                     attr_net,
                     train_loader,
                     test_loader,
                     optimizer,
                     scheduler,
                     cce_loss,
                     bce_loss,
                     save_path,                    
                     device,
                     version,
                     resume=False,
                     loss_train = None,
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

    print('epoches started')
    if resume:
        start_epoch = stoped_epoch+1
    else:
        start_epoch = 1    
    for epoch in range(start_epoch,num_epoch+1):
        
        # torch.cuda.empty_cache()
        attr_net = attr_net.to(device)        
        attr_net.train()
        loss_e = []
        loss_t = []
        
        # attributes temporary metrics lists 
        ft_train = []
        ft_test = []
        acc_train = []
        acc_test = []
        
        for idx, data in enumerate(train_loader):
            
            for key, _ in data.items():
                data[key] = data[key].to(device)
            # forward step
            optimizer.zero_grad()
            out_data = attr_net.forward(data['img'])   
            
            # compute losses and evaluation metrics:
            # head 
            loss0 = cce_loss(out_data['head'], data['head'].argmax(dim=1))  
            y_head = tensor_max(softmax(out_data['head']))

            # head_color
            loss1 = cce_loss(out_data['head_colour'], data['head_colour'].argmax(dim=1))  
            y_head_colour = tensor_max(softmax(out_data['head_colour']))
            
            # body
            loss2 = cce_loss(out_data['body'],data['body'].argmax(dim=1))
            y_body = tensor_max(softmax(out_data['body']))
            
            # body_type
            loss3 = bce_loss(out_data['body_type'].squeeze(),data['body_type'].float())   
            y_body_type = tensor_thresh(torch.sigmoid(out_data['body_type']), 0.5)
            
            # leg
            loss4 = cce_loss(out_data['leg'],data['leg'].argmax(dim=1))
            y_leg = tensor_max(softmax(out_data['leg']))
            
            # foot 
            loss5 = cce_loss(out_data['foot'],data['foot'].argmax(dim=1))  
            y_foot = tensor_max(softmax(out_data['foot']))
            
            # gender
            loss6= bce_loss(out_data['gender'].squeeze(),data['gender'].float())
            y_gender = tensor_thresh(torch.sigmoid(out_data['gender']), 0.5)
            
            # bags
            loss7 = cce_loss(out_data['bags'],data['bags'].argmax(dim=1))
            y_bags = tensor_max(softmax(out_data['bags']))
            
            # body_colour
            loss8 = bce_loss(out_data['body_colour'],data['body_colour'].float())  
            y_body_colour = tensor_thresh(torch.sigmoid(out_data['body_colour']), 0.5)
            
            # leg_colour
            loss9 = cce_loss(out_data['leg_colour'],data['leg_colour'].argmax(dim=1))
            y_leg_colour = tensor_max(softmax(out_data['leg_colour']))
            
            # foot_colour
            loss10 = cce_loss(out_data['foot_colour'],data['foot_colour'].argmax(dim=1))
            y_foot_colour = tensor_max(softmax(out_data['foot_colour']))
            
            # age
            loss11 = cce_loss(out_data['age'],data['age'].argmax(dim=1))
            y_age = tensor_max(softmax(out_data['age']))
            
            y_attr = torch.cat((y_gender, y_head, y_head_colour, y_body, 
                                y_body_type, y_body_colour,
                                y_bags, y_leg, y_leg_colour,
                                y_foot, y_foot_colour, y_age), dim=1) 
                
            y_target = torch.cat((data['gender'].unsqueeze(dim=1), data['head'], data['head_colour'],
                                  data['body'], data['body_type'].unsqueeze(dim=1),
                                  data['body_colour'], data['bags'],
                                  data['leg'], data['leg_colour'],
                                  data['foot'], data['foot_colour'], data['age']), dim=1)                
            # total loss
            loss = loss0+loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9+loss10+loss11
            # evaluation    
            train_attr_metrics = tensor_metrics(y_target.float(), y_attr)
            # append results
            ft_train.append(train_attr_metrics[-2])
            acc_train.append(train_attr_metrics[-3])
            loss_e.append(loss.item())
            
            # backward step
            loss.backward()
            # optimization step
            optimizer.step()
            
            # print log
            if idx % 1 == 0:
                print('\nTrain Epoch: {} [{}/{} , lr {}] \t Loss: {:.6f} \nattr_metrics: F1_attr: {:.3f} acc_attr{:.3f}'.format(
                    epoch, idx , len(train_loader),
                     optimizer.param_groups[0]['lr'],
                      loss.item(),ft_train[-1], acc_train[-1]))
       
        train_loss.append(np.mean(loss_e))
        F1_train.append(np.mean(ft_train))
        Acc_train.append(np.mean(acc_train))
        
        # evaluation:     
        attr_net.eval()
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                
                for key, _ in data.items():
                    data[key] = data[key].to(device)
                    
                # forward step
                out_data = attr_net.forward(data['img'])           
                
                # compute losses and evaluation metrics:
                # head 
                loss0 = cce_loss(out_data['head'], data['head'].argmax(dim=1))  
                y_head = tensor_max(softmax(out_data['head']))

                # head_color
                loss1 = cce_loss(out_data['head_colour'], data['head_colour'].argmax(dim=1))  
                y_head_colour = tensor_max(softmax(out_data['head_colour']))  
                
                # body
                loss2 = cce_loss(out_data['body'],data['body'].argmax(dim=1))
                y_body = tensor_max(softmax(out_data['body']))
                
                # body_type
                loss3 = bce_loss(out_data['body_type'].squeeze(),data['body_type'].float())   
                y_body_type = tensor_thresh(torch.sigmoid(out_data['body_type']), 0.5)
                
                # leg
                loss4 = cce_loss(out_data['leg'],data['leg'].argmax(dim=1))
                y_leg = tensor_max(softmax(out_data['leg']))
                
                # foot 
                loss5 = cce_loss(out_data['foot'],data['foot'].argmax(dim=1))  
                y_foot = tensor_max(softmax(out_data['foot']))
                
                # gender
                loss6 = bce_loss(out_data['gender'].squeeze(),data['gender'].float())
                y_gender = tensor_thresh(torch.sigmoid(out_data['gender']), 0.5)
                
                # bags
                loss7 = cce_loss(out_data['bags'],data['bags'].argmax(dim=1))
                y_bags = tensor_max(softmax(out_data['bags']))
                
                # body_colour
                loss8 = bce_loss(out_data['body_colour'],data['body_colour'].float())  
                y_body_colour = tensor_thresh(torch.sigmoid(out_data['body_colour']), 0.5)
                
                # leg_colour
                loss9 = cce_loss(out_data['leg_colour'],data['leg_colour'].argmax(dim=1))
                y_leg_colour = tensor_max(softmax(out_data['leg_colour']))
                
                # foot_colour
                loss10 = cce_loss(out_data['foot_colour'],data['foot_colour'].argmax(dim=1))
                y_foot_colour = tensor_max(softmax(out_data['foot_colour']))

                # age
                loss11 = cce_loss(out_data['age'],data['age'].argmax(dim=1))
                y_age = tensor_max(softmax(out_data['age']))
                
                y_attr = torch.cat((y_gender, y_head, y_head_colour, y_body, 
                                    y_body_type, y_body_colour,
                                    y_bags, y_leg, y_leg_colour,
                                    y_foot, y_foot_colour, y_age), dim=1)
                    
                y_target = torch.cat((data['gender'].unsqueeze(dim=1), data['head'], data['head_colour'],
                                      data['body'], data['body_type'].unsqueeze(dim=1),
                                      data['body_colour'], data['bags'],
                                      data['leg'], data['leg_colour'],
                                      data['foot'], data['foot_colour'], data['age']), dim=1)                                            
                # total loss
                loss = loss0+loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9+loss10+loss11
                    
                test_attr_metrics = tensor_metrics(y_target.float(), y_attr)
                ft_test.append(test_attr_metrics[-2])
                acc_test.append(test_attr_metrics[-3]) 
                loss_t.append(loss.item())
                
        test_loss.append(np.mean(loss_t))
        F1_test.append(np.mean(ft_test))
        Acc_test.append(np.mean(acc_test))
        print('Epoch: {}\ntrain loss: {:.6f}\ntest loss: {:.6f}\n\nF1 train: {:.4f}\nF1 test: {:.4f}\n\nacc_train: {:.4f}\nacc_test: {:.4f} '.format(
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
                print('test f1 improved')
            else:
                print('\ntest f1 improved')
#%%    
def train_attr_id(attr_net,
                  train_loader,
                  test_loader,
                  num_epoch,
                  optimizer,
                  scheduler,
                  criterion1,
                  criterion2,
                  saving_path,
                  num_id,
                  version,
                  id_inc=True,
                  dynamic_lr=False):

    # def training(train_loader,test_loader,generator,classifier,num_epoch,optimizer,criterion1,criterion2,scheduler,device):
    train_loss = []
    test_loss = []
    F1_train = []
    F1_test = []  
    Acc_train = []
    Acc_test = []
    
    for epoch in range(1,num_epoch+1):
        
        attr_net.to(device)
        attr_net.train()
        loss_e = []
        loss_t = []
        ft_train = []
        ft_test = []
        acc_train = []
        acc_test = []
        
        for idx, data in enumerate(train_loader):

            data[0] = data[0].to(device)
            data[1] = data[1].to(device)
            data[2] = data[2].to(device)           
            # forward step
            optimizer.zero_grad()
            out_data = attr_net(data[0])
            #out_data[0] = out_data[0].to(device)
            #out_data[1] = out_data[1].to(device)
            
            # compute losses and evaluation metrics:
                
            # attributes
            loss0 = criterion2(out_data[1],data[2].float())  
            attr_out = torch.round(out_data[1])
            metrics = tensor_metrics(data[2].float(),attr_out)
            ft_train.append(metrics[7])
            
            if id_inc:
                loss1 = criterion1(out_data[0],data[1])  
                y = tensor_max(out_data[0].to('cpu'))
                oh_id = id_onehot(data[1],num_id)
                metrics = tensor_metrics(oh_id.float(),y) 
                acc_train.append(metrics[-2])  
                loss = loss0+loss1
            else:
                loss = loss0
            loss_e.append(loss.item())  
            
            # backward step
            loss.backward()
            
            # optimization step
            optimizer.step()
            if dynamic_lr:
                pass
            else:
                scheduler.step()
            # print log
            if id_inc:
                print('Train Epoch: {} [{}/{} , lr {}] \t Loss: {:.6f} \t F1: {:.3f} \t accuracy:{:.3f}'.format(
                    epoch,
                    idx , len(train_loader), 
                    optimizer.param_groups[0]['lr'], 
                    loss_e[-1],
                    np.mean(ft_train),
                    np.mean(acc_train)))
            else:
                print('Train Epoch: {} [{}/{} , lr {}] \t Loss: {:.6f} \t F1: {:.3f}'.format(
                    epoch,
                    idx , len(train_loader), 
                    optimizer.param_groups[0]['lr'], 
                    loss.item(),np.mean(ft_train)))                
        if id_inc:
            Acc_train.append(np.mean(acc_train))
            
        train_loss.append(np.mean(loss_e))
        F1_train.append(np.mean(ft_train))
        
        # evaluation:     
        attr_net.eval()
        with torch.no_grad():
            for idx, data in enumerate(test_loader):

                data[0] = data[0].to(device)
                data[1] = data[1].to(device)
                data[2] = data[2].to(device)
                out_data = attr_net(data[0])
                #out_data[0] = out_data[0].to(device)
                #out_data[1] = out_data[1].to(device)
                # compute losses and evaluation metrics:
                    
                # attributes
                loss0 = criterion2(out_data[1],data[2].float())  
                attr_out = torch.round(out_data[1])
                metrics = tensor_metrics(data[2].float(),attr_out)
                ft_test.append(metrics[7])
                
                if id_inc:
                    loss1 = criterion1(out_data[0],data[1])  
                    y = tensor_max(out_data[0])
                    oh_id = id_onehot(data[1],num_id)
                    metrics = tensor_metrics(oh_id.float(),y)
                    acc_test.append(metrics[-2])  
                    loss = loss0+loss1
                else:
                    loss = loss0
                if dynamic_lr:
                    scheduler.step(loss)
                loss_t.append(loss.item())  
        test_loss.append(np.mean(loss_t))
        F1_test.append(np.mean(ft_test))
        if id_inc:
            Acc_test.append(np.mean(acc_test))
            print('Epoch: {} \n train loss: {:.6f} \n test loss: {:.6f} \n \n F1 train: {:.4f} \n F1 test: {:.4f} \n accuracy train: {:.4f} \n accuracy test: {:.4f}'.format(
                        epoch,
                        train_loss[-1],
                        test_loss[-1],
                        F1_train[-1],
                        F1_test[-1],
                        Acc_train[-1],
                        Acc_test[-1]))
        else:
            print('Epoch: {} \n train loss: {:.6f} \n test loss: {:.6f} \n \n F1 train: {:.4f} \n F1 test: {:.4f}'.format(
                        epoch,
                        train_loss[-1],
                        test_loss[-1],
                        F1_train[-1],
                        F1_test[-1]))            
            
        
        torch.save(test_loss,saving_path+'testloss_'+version+'.pth')
        torch.save(train_loss,saving_path+'trainloss_'+version+'.pth')
        
        torch.save(F1_test,saving_path+'F1_test_'+version+'.pth')
        torch.save(F1_train,saving_path+'F1_train_'+version+'.pth')
        if id_inc:
            torch.save(Acc_test,saving_path+'testacc_'+version+'.pth')
            torch.save(Acc_train,saving_path+'trainacc_'+version+'.pth') 
        if len(F1_test)>2: 
            if F1_test[-1] > F1_test[-2]:
                print('our net improved')
                torch.save(attr_net, saving_path+'attrnet_'+version+'.pth')
                torch.save(optimizer.state_dict(), saving_path+'attrnet_'+version+'.pth')

#%%    
def train_attr(attr_net,
                  train_loader,
                  test_loader,
                  num_epoch,
                  optimizer,
                  scheduler,
                  criterion,
                  saving_path,
                  version,
                  id_inc=True,
                  dynamic_lr=False):

    # def training(train_loader,test_loader,generator,classifier,num_epoch,optimizer,criterion1,criterion2,scheduler,device):
    train_loss = []
    test_loss = []
    F1_train = []
    F1_test = []  
    Acc_train = []
    Acc_test = []
    f1_best = 0
    
    for epoch in range(1,num_epoch+1):
        
        attr_net.to(device)
        attr_net.train()
        loss_e = []
        loss_t = []
        ft_train = []
        ft_test = []
        acc_train = []
        acc_test = []
        
        for idx, data in enumerate(train_loader):

            data[0] = data[0].to(device)
            data[1] = data[1].to(device)
            data[2] = data[2].to(device)           
            # forward step
            optimizer.zero_grad()
            out_data = attr_net(data[0])
                
            # attributes
            loss = criterion(out_data,data[2].float())  
            out_data = torch.sigmoid(out_data)
            attr_out = tensor_thresh(out_data)
            metrics = tensor_metrics(data[2].float(),attr_out)
            ft_train.append(metrics[7])
            acc_train.append(metrics[6])
            loss_e.append(loss.item())  
            
            # backward step
            loss.backward()
            
            # optimization step
            optimizer.step()

            print('Train Epoch: {} [{}/{} , lr {}] \t Loss: {:.6f} \t F1: {:.3f} \t acc: {:.3f}'.format(
                epoch,
                idx , len(train_loader), 
                optimizer.param_groups[0]['lr'], 
                loss.item(),np.mean(ft_train), np.mean(acc_train)))                
            
        train_loss.append(np.mean(loss_e))
        F1_train.append(np.mean(ft_train))
        Acc_train.append(np.mean(acc_train))
        
        # evaluation:     
        attr_net.eval()
        with torch.no_grad():
            for idx, data in enumerate(test_loader):

                data[0] = data[0].to(device)
                data[1] = data[1].to(device)
                data[2] = data[2].to(device)
                out_data = attr_net(data[0])
                #out_data[0] = out_data[0].to(device)
                #out_data[1] = out_data[1].to(device)
                # compute losses and evaluation metrics:
                    
                # attributes
                loss = criterion(out_data,data[2].float())  
                out_data = torch.sigmoid(out_data)
                attr_out = tensor_thresh(out_data)
                metrics = tensor_metrics(data[2].float(),attr_out)
                ft_test.append(metrics[7])
                acc_test.append(metrics[6])
                loss_t.append(loss.item()) 
                
        test_loss.append(np.mean(loss_t))
        F1_test.append(np.mean(ft_test))
        Acc_test.append(np.mean(acc_test))
        if dynamic_lr:
            scheduler.step(np.mean(loss_t)) 
        else:
            scheduler.step()
        print('Epoch: {} \n train loss: {:.6f} \n test loss: {:.6f} \n \n F1 train: {:.4f} \n F1 test: {:.4f} \n \n acc train: {:.4f} \n acc test: {:.4f}'.format(
                    epoch,
                    train_loss[-1],
                    test_loss[-1],
                    F1_train[-1],
                    F1_test[-1],
                    Acc_train[-1],
                    Acc_test[-1]))            
        
        
        torch.save(test_loss,saving_path+'testloss_'+version+'.pth')
        torch.save(train_loss,saving_path+'trainloss_'+version+'.pth')
        
        torch.save(F1_test,saving_path+'F1_test_'+version+'.pth')
        torch.save(F1_train,saving_path+'F1_train_'+version+'.pth')

        torch.save(Acc_test,saving_path+'Acc_test'+version+'.pth')
        torch.save(Acc_train,saving_path+'Acc_train'+version+'.pth')
        
        if F1_test[-1]>f1_best: 
            f1_best = F1_test[-1]
            print('our net improved')
            torch.save(attr_net, saving_path+'attrnet_'+version+'.pth')
            torch.save(optimizer.state_dict(), saving_path+'attrnet_'+version+'.pth')

