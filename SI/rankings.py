#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 15:56:42 2022
@author: hossein
"""


baseline_path = '/home/hossein/Downloads/osnet_x1_0_market.pth'
utils.load_pretrained_weights(model, baseline_path)
model = model.to(device)

model.eval()
with torch.no_grad():
    targets = []
    predicts = []
    test_features = []
    query_features = []
    for idx, data in enumerate(gallery_loader):
        
        for key, _ in data.items():
            data[key] = data[key].to(device)
            
        # forward step
        # out_data, out_features = attr_net.get_feature(data['img'], method='baseline')
        out_features = model(data['img'])

        # y_attr, y_target = CA_target_attributes_12(out_data, data, part_loss, need_tensor_max=categorical, categorical=categorical)
        # predicts.append(y_attr.to('cpu'))
        # targets.append(y_target.to('cpu'))

        test_features.append(out_features)
    
    for idx, data_query in enumerate(query_loader):
        
        for key, _ in data.items():
            data_query[key] = data_query[key].to(device)
            
        # forward step
        out_features_q = model(data_query['img'])
        # _, out_features_q = attr_net.get_feature(data_query['img'], method='baseline')
        # y_attr, y_target = CA_target_attributes_12(out_data, data, part_loss, need_tensor_max=categorical, categorical=categorical)
        # predicts.append(y_attr.to('cpu'))
        # targets.append(y_target.to('cpu'))

        query_features.append(out_features_q)


test_features = torch.cat(test_features)
query_features = torch.cat(query_features)  



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
        out_data = attr_concatenator(out_data, activation=False)

        test_features.append(out_features.to('cpu'))
        predicts.append(out_data.to('cpu'))
    for idx, data_query in enumerate(query_loader):
        
        for key, _ in data.items():
            data_query[key] = data_query[key].to(device)
            
        out_data_q, out_features_q = attr_net.get_feature(data_query['img'], method='baseline')
        out_data_q = attr_concatenator(out_data_q, activation = False)
        
        query_features.append(out_features_q.to('cpu'))
        predicts_q.append(out_data_q.to('cpu'))

# test_features = torch.cat(test_features)
# query_features = torch.cat(query_features)  

test_predics = torch.cat(predicts)
query_predicts = torch.cat(predicts_q)  



test_features = torch.load('/home/hossein/baseline_Market_test_feat.pth')
query_features = torch.load('/home/hossein/baseline_Market_query_feat.pth')  



# model_test_feat = test_features.to('cpu')
# model_query_feat = query_features.to('cpu')

# torch.save(model_test_feat, '/home/hossein/baseline_Market_test_feat.pth')
# torch.save(model_query_feat, '/home/hossein/baseline_Market_query_feat.pth')
import torchreid

attr_dist = torchreid.metrics.compute_distance_matrix(query_predicts, test_predics, metric='cosine')
attr_dist = attr_dist.cpu().numpy()

dist = torchreid.metrics.compute_distance_matrix(query_features, test_features)
dist = dist.cpu().numpy()

q_q_dist = torchreid.metrics.compute_distance_matrix(query_features, query_features)
q_q_dist = q_q_dist.cpu().numpy()

g_g_dist = torchreid.metrics.compute_distance_matrix(test_features, test_features)
g_g_dist = g_g_dist.cpu().numpy()

ratio = 0.09
attr_feat_dist = ratio*attr_dist + (1-ratio)*new_dist

new_dist = re_ranking(dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3)

q_pids = np.asarray(query['id'])
q_camids = np.asarray(query['cam_id'])
q_camids -= 1
# gallery

g_pids = np.asarray(gallery['id'])
g_camids = np.asarray(gallery['cam_id'])
g_camids -= 1
# query_ids = query['id']
# gallery_ids = gallery['id']

import numpy as np

# g_pids = np.asarray(query_ids)
# g_camids = np.asarray(gallery_ids)

dist = torch.cdist(query_features, test_features)
dist = dist.cpu().numpy()

cmc, mAP = eval_func(dist, q_pids, g_pids, q_camids, g_camids)
cmc, mAP = eval_func(new_dist, q_pids, g_pids, q_camids, g_camids)
cmc, mAP = eval_func(attr_feat_dist, q_pids, g_pids, q_camids, g_camids)
rank = torchreid.metrics.rank.evaluate_rank(dist, q_pids, g_pids, q_camids,
                                            g_camids, use_metric_cuhk03=False, use_cython=False)

# m, n = query_features.shape[0], test_features.shape[0]
# distmat = torch.pow(test_features, 2).sum(dim=1, keepdim=True).expand(m, n) + \
#           torch.pow(query_features, 2).sum(dim=1, keepdim=True).expand(n, m).t()
# distmat.addmm_(1, -2, query_features, test_features.t())
# distmat = distmat.cpu().numpy()



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