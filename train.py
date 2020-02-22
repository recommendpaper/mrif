import pickle as pkl
import os
import tensorflow as tf
import sys
from sklearn.metrics import *
import random
import time
import numpy as np

import data_loader
import util
import modelV2
import train_util


dataset = 'amazon'
print('Begin loading data.')
if dataset == 'amazon':
    with open('amazon_movie_L100_S10.pkl', 'rb') as f:
        train_set_single = pkl.load(f)
        train_set_multi = pkl.load(f)
        test_set_single = pkl.load(f)
        test_set_multi = pkl.load(f)
        dataset_infos = pkl.load(f)
print('Finished loading data.')



random.seed(1111)
np.random.seed(1111)

EMBEDDING_SIZE = 32
HIDDEN_SIZE = 32


MAX_SEQ_LEN = 100

TRAIN_BATCH_SIZE = 256
EVAL_BATCH_SIZE = 2048
MAX_EPOCH = 60
MAX_PRED_USER = 1000
PRETRAIN_EPOCH = 40
MRIF_KERNELS = [3,3]
MRIF_PARALELL = False


repeat_times = 1


model_type = 'MRIF_attn'
# lrs = [1e-3,0.8e-3,0.5e-3]
lrs = [1e-3]
reg_lambdas = [0]
keep_probs = [0.5]

eval_steps = np.ceil(1.0*len(train_set_single)/TRAIN_BATCH_SIZE) * 20
# eval_steps = 10
# MAX_PRED_USER = 50
# EVAL_BATCH_SIZE = 3

metrics = {}
metrics['loss'] = 10
metrics['auc'] = 0
metrics['gauc'] = 0
metrics['ndcg5'] = 0
metrics['ndcg10'] = 0
metrics['hit5'] = 0
metrics['hit10'] = 0
metrics['mrr'] = 0
for lr in lrs:
    for reg_lambda in reg_lambdas:
        for keep_prob in keep_probs:
            print('lr:{}, reg:{}, keep:{}'.format(lr,reg_lambda,keep_prob))
            losses = []
            aucs = []
            gaucs = []
            ndcgs5 = []
            ndcgs10 = []
            hits5 = []
            hits10 = []
            mrrs = []
            for i in range(repeat_times):
                loss,auc,gauc,ndcg,hit,mrr = train_util.train_V3(model_type, train_set_single, test_set_multi,
                                                                 test_set_single, MAX_SEQ_LEN,
                                                 lr, reg_lambda, keep_prob, eval_steps, EMBEDDING_SIZE,
                                               HIDDEN_SIZE, dataset_infos['feature_size'],TRAIN_BATCH_SIZE,EVAL_BATCH_SIZE,
                                                         dataset_infos,MAX_EPOCH,max_pred_user=MAX_PRED_USER
                                                        ,mrif_kernels=MRIF_KERNELS,mrif_parallel=MRIF_PARALELL
                                                                 ,pretrain_epoch=PRETRAIN_EPOCH)
                losses.append(loss)
                aucs.append(auc)
                gaucs.append(gauc)
                ndcgs5.append(ndcg[1])
                ndcgs10.append(ndcg[2])
                hits5.append(hit[1])
                hits10.append(hit[2])
                mrrs.append(mrr)


            if sum(losses) / repeat_times < metrics['loss']:
                metrics['loss'] = sum(losses) / repeat_times
            
            res_pair = [[aucs,'auc'],[gaucs,'gauc'],[ndcgs10,'ndcg10'],
                        [ndcgs5,'ndcg5'],[hits5,'hit5'],[hits10,'hit10'],[mrrs,'mrr']]
            for results,best_result in res_pair:
                if sum(results) / repeat_times > metrics[best_result]:
                    metrics[best_result] = sum(results) / repeat_times
            
print('final loss[{:.4f}] auc[{:.4f}] gauc[{:.4f}] ndcg@5[{:.4f}] ndcg@10[{:.4f}] hit@5[{:.4f}] hit@10[{:.4f}] mrr[{:.4f}]'.
     format(
        metrics['loss'],
        metrics['auc'],
        metrics['gauc'],
        metrics['ndcg5'],
        metrics['ndcg10'],
        metrics['hit5'],
        metrics['hit10'],
        metrics['mrr'],))    
# print("FINAL RESULT: AUC=%.4f\tLOGLOSS=%.4f" % (result_auc, result_logloss))