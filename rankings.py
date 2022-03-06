#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 15:56:42 2022

@author: hossein
"""

import torchreid
import numpy as np
import torch
from trainings import attr_concatenator

def feature_extractor(attr_net, gallery_loader, query_loader, device, activation=False):
    attr_net.eval()
    with torch.no_grad():
        predicts_q = []
        predicts = []
        test_features = []
        query_features = []
        for idx, data in enumerate(gallery_loader):
            
            for key, _ in data.items():
                data[key] = data[key].to(device)
                
            # forward step
            out_data, out_features = attr_net.get_feature(data['img'], method='baseline')
            out_data = attr_concatenator(out_data, activation=activation)
    
            test_features.append(out_features.to('cpu'))
            predicts.append(out_data.to('cpu'))
        for idx, data_query in enumerate(query_loader):
            
            for key, _ in data.items():
                data_query[key] = data_query[key].to(device)
                
            out_data_q, out_features_q = attr_net.get_feature(data_query['img'], method='baseline')
            out_data_q = attr_concatenator(out_data_q, activation = activation)
            
            query_features.append(out_features_q.to('cpu'))
            predicts_q.append(out_data_q.to('cpu'))
    
    test_features = torch.cat(test_features)
    query_features = torch.cat(query_features)  
    
    test_predics = torch.cat(predicts)
    query_predicts = torch.cat(predicts_q)  
    return (query_predicts, test_predics), (query_features, test_features)

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def k_reciprocal_neigh( initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = np.where(backward_k_neigh_index==i)[0]
    return forward_k_neigh_index[fi]

def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.
    original_dist = np.concatenate(
      [np.concatenate([q_q_dist, q_g_dist], axis=1),
       np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
      axis=0)
    original_dist = 2. - 2 * original_dist   # change the cosine similarity metric to euclidean similarity metric
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0))
    V = np.zeros_like(original_dist).astype(np.float32)
    #initial_rank = np.argsort(original_dist).astype(np.int32)
    # top K1+1
    initial_rank = np.argpartition( original_dist, range(1,k1+1) )

    query_num = q_g_dist.shape[0]
    all_num = original_dist.shape[0]

    for i in range(all_num):
        # k-reciprocal neighbors
        k_reciprocal_index = k_reciprocal_neigh( initial_rank, i, k1)
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neigh( initial_rank, candidate, int(np.around(k1/2)))
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)

    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:,i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1,all_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    return final_dist


def rank_calculator(attr_net, gallery_loader, query_loader,
                    gallery, query, device, ratio = 0.09, activation=False):
    
    print('feature extraction')
    (query_predicts, test_predics), (query_features, test_features) = feature_extractor(attr_net = attr_net,
                                                gallery_loader = gallery_loader,
                                                query_loader = query_loader,
                                                device = device)    
    print('\n','compute distances')
    attr_dist = torchreid.metrics.compute_distance_matrix(query_predicts, test_predics, metric='cosine')
    attr_dist = attr_dist.cpu().numpy()
    
    dist = torchreid.metrics.compute_distance_matrix(query_features, test_features)
    dist = dist.cpu().numpy()
    
    q_q_dist = torchreid.metrics.compute_distance_matrix(query_features, query_features)
    q_q_dist = q_q_dist.cpu().numpy()
    
    g_g_dist = torchreid.metrics.compute_distance_matrix(test_features, test_features)
    g_g_dist = g_g_dist.cpu().numpy()
    print('\n','compute re-ranking')
    new_dist = re_ranking(dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3)

    q_pids = np.asarray(query['id'])
    q_camids = np.asarray(query['cam_id'])
    q_camids -= 1
    # gallery
    
    g_pids = np.asarray(gallery['id'])
    g_camids = np.asarray(gallery['cam_id'])
    g_camids -= 1
    
    attr_feat_dist = ratio*attr_dist + (1-ratio)*new_dist    
    print('\n','compute rankings')
    cmc, mAP = eval_func(attr_feat_dist, q_pids, g_pids, q_camids, g_camids)
    return cmc, mAP



# attr_feat_cat_test = torch.cat((test_features, test_predics), dim=1)
# attr_feat_cat_query = torch.cat((query_features, query_predicts), dim=1)


# eliminates = [2,8,10,32,41,44]
# for i in eliminates:
#     test_predics[:, i] = 0
#     query_predicts[:, i] = 0



# attr_dist = torchreid.metrics.compute_distance_matrix(query_predicts, test_predics, metric='cosine')
# attr_dist = attr_dist.cpu().numpy()

# feat_dist = torchreid.metrics.compute_distance_matrix(query_features, test_features, metric='cosine')
# feat_dist = feat_dist.cpu().numpy()

# q_g_intersection = torch.matmul(query_predicts, test_predics.T)
# q_g_intersection = q_g_intersection.cpu().numpy()
# sum_q = torch.unsqueeze(torch.sum(query_predicts, dim=1), dim=1).expand(-1, test_predics.shape[0])
# sum_g = torch.unsqueeze(torch.sum(test_predics, dim=1), dim=1).expand(-1, query_predicts.shape[0]).T

# sum_q_g = sum_q + sum_g

# eps=1e-08
# iou = q_g_intersection / (sum_q_g - q_g_intersection + eps)
# iou = iou.cpu().numpy()





# dist = torchreid.metrics.compute_distance_matrix(attr_feat_cat_query, attr_feat_cat_test)
# dist = dist.cpu().numpy()
# # dist = (dist-dist.min())/(dist.max()-dist.min())

# dist = torchreid.metrics.compute_distance_matrix(query_features, test_features)
# dist = dist.cpu().numpy()

# q_q_dist = torchreid.metrics.compute_distance_matrix(query_features, query_features)
# q_q_dist = q_q_dist.cpu().numpy()
# # q_q_dist = (q_q_dist-q_q_dist.min())/(q_q_dist.max()-q_q_dist.min())

# g_g_dist = torchreid.metrics.compute_distance_matrix(test_features, test_features)
# g_g_dist = g_g_dist.cpu().numpy()
# # g_g_dist = (g_g_dist-g_g_dist.min())/(g_g_dist.max()-g_g_dist.min())

# new_dist = re_ranking(dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3)



# dist_q_q = torchreid.metrics.compute_distance_matrix(query_predicts, query_predicts)
# dist_q_q = dist_q_q.cpu().numpy()

# dist_g_g = torchreid.metrics.compute_distance_matrix(test_predics, test_predics)
# dist_g_g = dist_g_g.cpu().numpy()


# new_dist = re_ranking(dist, dist_q_q, dist_g_g, k1=20, k2=6, lambda_value=0.3)



# q_pids = np.asarray(query['id'])
# q_camids = np.asarray(query['cam_id'])
# q_camids -= 1
# # gallery

# g_pids = np.asarray(gallery['id'])
# g_camids = np.asarray(gallery['cam_id'])
# g_camids -= 1
# # query_ids = query['id']
# # gallery_ids = gallery['id']


# # g_pids = np.asarray(query_ids)
# # g_camids = np.asarray(gallery_ids)

# dist = torch.cdist(query_features, test_features)
# dist = dist.cpu().numpy()
# dist = (dist-dist.min())/(dist.max()-dist.min())

# e = 0.09
# attr_feat_dist = e*attr_dist + (1-e)*new_dist

# b = 0.05
# aa = b*(feat_dist) + (1-b)*(attr_feat_dist)

# e = 0.1
# attr_feat_dist = e*(attr_dist+feat_dist) + (1-e)*new_dist

# e = 0.2
# attr_feat_dist = e*(feat_dist) + (1-e)*new_dist

# cmc, mAP = eval_func(attr_feat_dist, q_pids, g_pids, q_camids, g_camids)
# rank = torchreid.metrics.rank.evaluate_rank(new_dist, q_pids, g_pids, q_camids,
#                                             g_camids, use_metric_cuhk03=False, use_cython=False)

# def torch_delete(arr: torch.Tensor, ind: int, dim: int) -> torch.Tensor:
#     skip = [i for i in range(arr.size(dim)) if i != ind]
#     indices = [slice(None) if i != dim else skip for i in range(arr.ndim)]
#     return arr.__getitem__(indices)




# def IOU_distance(q_attr, g_attr)