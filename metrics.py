#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:15:59 2021

@author: hossein

here we can find different types of metrics 
that are define for person-attribute detection.
this is Hossein Bodaghies thesis
"""
import torch


'''
first defined tp tn fn fp and then manipulate them to 
calculate precision recall and accuracy and f1 

precision = tp/(tp+fp)
recall = tp/(tp+fn)
accuracy = tp+tf/(tp+tn+fp+fn)
f1 = 2*(precision+recall)/(precision+recall)
'''


def tensor_metrics(target,predict):
    '''
    
    Parameters
    ----------
    target : torch-tensor
        (N,C) the value of each element should be zero or one.
        N is batch size and C is number of classes.
        C must be more than one (not recommended for boolians).
    predict : torch-tensor
        (N,C) the value of element should be zero or one.
        N is batch size and C is number of classes. 
        C must be more than one (not recommended for boolians).
    Returns
    -------
    list
        [precision,
         recall,
         accuracy,
         f1,
         precision_total,
         recall_total,
         accuracy_total,
         f1_total].
        
    '''
    
    eps = 1e-6
    
    true_positive = torch.zeros((predict.size()[1]))
    true_negative = torch.zeros((predict.size()[1]))
    false_positive = torch.zeros((predict.size()[1]))
    false_negative = torch.zeros((predict.size()[1]))
    
    true_positive_total = 0
    true_negative_total = 0
    false_positive_total = 0
    false_negative_total = 0
    
    
    for i in range(len(predict)):
        for j in range(predict.size()[1]):
            if predict[i,j] == target[i,j] and target[i,j] == 1:
                true_positive[j] += 1
                true_positive_total += 1
            elif predict[i,j] == target[i,j] and target[i,j] == 0:
                true_negative[j] += 1
                true_negative_total += 1
            elif predict[i,j] != target[i,j] and target[i,j] == 1:
                false_negative[j] += 1
                false_negative_total += 1
            elif predict[i,j] != target[i,j] and target[i,j] == 0:
                false_positive[j] += 1
                false_positive_total += 1
    
    precision = torch.zeros((predict.size()[1]))
    recall = torch.zeros((predict.size()[1]))
    accuracy = torch.zeros((predict.size()[1]))
    f1 = torch.zeros((predict.size()[1]))
    mean_accuracy = torch.zeros((predict.size()[1]))
    
    precision_total = true_positive_total/(true_positive_total+false_positive_total+eps)
    recall_total = true_positive_total/(true_positive_total+false_negative_total+eps)
    accuracy_total = (true_positive_total+true_negative_total)/(true_positive_total+false_negative_total+true_negative_total+false_positive_total+eps)
    f1_total = 2*(precision_total*recall_total)/(precision_total+recall_total+eps)
    
    for j in range(predict.size()[1]):
        precision[j] = true_positive[j]/(true_positive[j]+false_positive[j]+eps)
        recall[j] = true_positive[j]/(true_positive[j]+false_negative[j]+eps)
        accuracy[j] = (true_positive[j]+true_negative[j])/(true_positive[j]+false_negative[j]+true_negative[j]+false_positive[j]+eps)
        f1[j] = 2*(precision[j]*recall[j])/(precision[j]+recall[j]+eps)       
        mean_accuracy[j] = true_positive[j]/(true_positive[j]+false_negative[j]+eps) + true_negative[j]/(true_negative[j]+false_positive[j])
    mean_accuracy = mean_accuracy/2
    mean_accuracy_total = torch.mean(mean_accuracy)
        
    return [precision,
            recall,
            accuracy,
            f1,
            mean_accuracy,
            precision_total,
            recall_total,
            accuracy_total,
            f1_total, 
            mean_accuracy_total]


def boolian_metrics(target,predict):
    eps = 1e-6
    true_positive_total = 0
    true_negative_total = 0
    false_positive_total = 0
    false_negative_total = 0
    
    
    for i in range(len(predict)):
        
        if predict[i] == target[i] and target[i] == 1:
            true_positive_total += 1
        elif predict[i] == target[i] and target[i] == 0:
            true_negative_total += 1
        elif predict[i] != target[i] and target[i] == 1:
            false_negative_total += 1
        elif predict[i] != target[i] and target[i] == 0:
            false_positive_total += 1
            
    precision_total = true_positive_total/(true_positive_total+false_positive_total+eps)
    recall_total = true_positive_total/(true_positive_total+false_negative_total+eps)
    accuracy_total = (true_positive_total+true_negative_total)/(true_positive_total+false_negative_total+true_negative_total+false_positive_total+eps)
    f1_total = 2*(precision_total*recall_total)/(precision_total+recall_total+eps)
    
    return [precision_total,
            recall_total,
            accuracy_total,
            f1_total]

def tensor_metrics_detailes(target,predict):
    '''
    
    Parameters
    ----------
    target : torch-tensor
        (N,C) the value of each element should be zero or one.
        N is batch size and C is number of classes.
        C must be more than one (not recommended for boolians).
    predict : torch-tensor
        (N,C) the value of element should be zero or one.
        N is batch size and C is number of classes. 
        C must be more than one (not recommended for boolians).
    Returns
    -------
    list
        [precision,
         recall,
         accuracy,
         f1,
         precision_total,
         recall_total,
         accuracy_total,
         f1_total].
        
    '''
    
    eps = 1e-6
    
    true_positive = torch.zeros((predict.size()[1]))
    true_negative = torch.zeros((predict.size()[1]))
    false_positive = torch.zeros((predict.size()[1]))
    false_negative = torch.zeros((predict.size()[1]))
    
    real_positive = torch.zeros((predict.size()[1]))
    real_negative = torch.zeros((predict.size()[1]))
    
    true_positive_total = 0
    true_negative_total = 0
    false_positive_total = 0
    false_negative_total = 0
    
    
    for i in range(len(predict)):
        for j in range(predict.size()[1]):
            if predict[i,j] == target[i,j] and target[i,j] == 1:
                real_positive[j] += 1
                true_positive[j] += 1
                true_positive_total += 1
            elif predict[i,j] == target[i,j] and target[i,j] == 0:
                real_negative[j] += 1
                true_negative[j] += 1
                true_negative_total += 1
            elif predict[i,j] != target[i,j] and target[i,j] == 1:
                real_positive[j] += 1
                false_negative[j] += 1
                false_negative_total += 1
            elif predict[i,j] != target[i,j] and target[i,j] == 0:
                real_negative[j] += 1
                false_positive[j] += 1
                false_positive_total += 1
    
    precision = torch.zeros((predict.size()[1]))
    recall = torch.zeros((predict.size()[1]))
    accuracy = torch.zeros((predict.size()[1]))
    f1 = torch.zeros((predict.size()[1]))
    
    precision_total = true_positive_total/(true_positive_total+false_positive_total+eps)
    recall_total = true_positive_total/(true_positive_total+false_negative_total+eps)
    accuracy_total = (true_positive_total+true_negative_total)/(true_positive_total+false_negative_total+true_negative_total+false_positive_total+eps)
    f1_total = 2*(precision_total*recall_total)/(precision_total+recall_total+eps)
    
    for j in range(predict.size()[1]):
        precision[j] = true_positive[j]/(true_positive[j]+false_negative[j]+eps)
        recall[j] = true_positive[j]/(true_positive[j]+false_negative[j]+eps)
        accuracy[j] = (true_positive[j]+true_negative[j])/(true_positive[j]+false_negative[j]+true_negative[j]+false_positive[j]+eps)
        f1[j] = 2*(precision[j]*recall[j])/(precision[j]+recall[j]+eps)       
        
    return [precision,
            recall,
            accuracy,
            f1,
            precision_total,
            recall_total,
            accuracy_total,
            f1_total,
            real_positive,
            true_positive,
            false_positive,
            real_negative,
            true_negative,
            false_negative]

def category_metrics(target, predict):
    t_p = 0
    t = target.argmax(dim=1)
    p = predict.argmax(dim=1)
    for i in range(len(target)):
        if t[i] == p[i]:
            t_p += 1
    acc = t_p / len(target)
    return acc
    
    

# distances: 'euclidean' , 'cosin_similarity'
# takes two dictionary that should containe ['id']

def eval_map(query, gallery, model, device, distance='euclidean', num_accept=20, thr_needed=False, thr=0.5):
    
    """

    Parameters
    ----------
    query: TYPE dict
        DESCRIPTION.
    gallery_path : TYPE dict
        DESCRIPTION. are two dictionaries that should containe 'id' and 'img_names' as key
    model : TYPE
        DESCRIPTION.  takes feature extraction part of any model 
    device : TYPE
        DESCRIPTION.
    distance : TYPE, optional
        DESCRIPTION. The default is 'euclidean'. 'euclidean' , 'cosin_similarity' 
    num_accept : TYPE, optional
        DESCRIPTION. The default is 20.
    thr_needed : TYPE, optional
        DESCRIPTION. The default is False. in some cases insted of sorting in n number 
        of best choices we put a threshold and except only greater than threshold 
    thr : TYPE, optional
        DESCRIPTION. The default is 0.5. 

    Returns
    -------


    """
    
    
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate():
            pass
            
    
    return