#!/usr/bin/env python
# coding: utf-8

# ==============================================================================
# Preprocessing code for amazon dataset.
# ==============================================================================
#
# You may obtain raw data files at
# 
#     http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/
# 
# Dataset description can be found at 
#
#     http://jmcauley.ucsd.edu/data/amazon/
#
# ==============================================================================

import sys
import pickle
import pickle as pkl
import pandas as pd
import time
import copy
import matplotlib.pyplot as plt
import random
import numpy as np
from datetime import datetime
import os
from multiprocessing.dummy import Pool as ThreadPool 
import multiprocessing
import seaborn as sns
from tqdm.notebook import tqdm

random.seed(1111)
np.random.rand(1111)

MAX_LEN = 100 # Max output user sequence length
MAX_LEN_INP = round(MAX_LEN*1.1) # Max input user sequence length. This affect number of test cases.
MIN_LEN = 10
TEST_RATIO = 0.2

subset_type = 'games'

root_data_path = '/Users/lsh/Data/amazon/'
if subset_type == 'electro':
    REVIEW_FILE = os.path.join(root_data_path,'electro/reviews_Electronics_10.json')
    META_FILE = os.path.join(root_data_path,'electro/meta_Electronics.json')
    DATASET_PKL_PATH = os.path.join(root_data_path,
                                    'electro/amazon_electro_L{}_S{}.pkl'.format(MAX_LEN,MIN_LEN))
elif subset_type == 'movie':
    REVIEW_FILE = os.path.join(root_data_path,'movie/reviews_Movies_and_TV_10.json')
    META_FILE = os.path.join(root_data_path,'movie/meta_Movies_and_TV.json')
    DATASET_PKL_PATH = os.path.join(root_data_path,
                                    'movie/amazon_movie_L{}_S{}.pkl'.format(MAX_LEN,MIN_LEN))

elif subset_type == 'games':
    REVIEW_FILE = os.path.join(root_data_path,'video_games/reviews_Video_Games_10.json')
    META_FILE = os.path.join(root_data_path,'video_games/meta_Video_Games.json')
    DATASET_PKL_PATH = os.path.join(root_data_path,
                                    'video_games/amazon_games_L{}_S{}.pkl'.format(MAX_LEN,MIN_LEN))


# In[121]:


def to_df_v0(file_path):
    """Convert raw data to dataframe. Sequential implementation.

    Args:
      file_path: string, a path

    Returns:
      A dataframe from file.
    """
    with open(file_path, 'r') as fin:
        df = {}
        i = 0
        for line in fin:
            df[i] = eval(line)
            i += 1
        df = pd.DataFrame.from_dict(df, orient='index')
        return df

def to_df_v1(file_path,core_num = 6):
    """Convert raw data to dataframe. Parallel implementation.

    Args:
      file_path: string, a path

    Returns:
      A dataframe from file.
    """
    with open(file_path, 'r') as fin:
        df = {}
        i = 0
        for line in fin:
            df[i] = line
            i += 1
    pool = multiprocessing.Pool(core_num)  # Make the Pool of workers
    res = pool.map(eval, df.values()) 
    pool.close() #close the pool and wait for the work to finish 
    pool.join()
    res = pd.DataFrame.from_records(res, index=df.keys())
    return res


# In[122]:


# convert to dataframe
review_df = to_df_v1(REVIEW_FILE)[['reviewerID', 'asin', 'unixReviewTime']]


# In[123]:


# preview
review_df


# In[124]:


# select users with at least MIN_LEN reviews
user_review_cnt = (review_df.groupby('reviewerID').
                  agg(hist_lens = pd.NamedAgg(column = 'asin', aggfunc=np.size)))
    
review_df_small = pd.merge(review_df,user_review_cnt[user_review_cnt.hist_lens>=MIN_LEN],on='reviewerID',how='inner')
print('Original dataset size: {}, dataset with at least MIN_LEN reviews size: {}'
      .format(len(review_df),len(review_df_small)))


# In[125]:


def conv2id(df):
    """Convert text to id.

    Args:
      df: Review dataframe.

    Returns:
      df: Review dataframe with text mapped to ids.
      asin_len: Number of items.
      feature_len: Number of features.
      user_len: Number of users.
    """
    asin_key = sorted(df['asin'].unique().tolist())
    asin_len = len(asin_key)
    asin_map = dict(zip(asin_key, range(1,asin_len+1)))
    df['asin'] = df['asin'].map(lambda x: asin_map[x])

    user_key = sorted(df['reviewerID'].unique().tolist())
    user_len = len(user_key)
    user_map = dict(zip(user_key, range(asin_len, asin_len + user_len)))
    df['reviewerID'] = df['reviewerID'].map(lambda x: user_map[x])
    
    feat_len = (asin_len + 1) + user_len 
    
    print('item count:{}, user count:{}, feature length:{}'
          .format(asin_len,user_len, feat_len))
    
    return df, asin_len, feat_len, user_len #remapped df and feature size


# In[126]:


# convert name to id
df, item_cnt, feature_size, user_len = conv2id(review_df_small)


# In[127]:


train_set_single = []
test_set_single = []

train_set_multi = []
test_set_multi = []

user_item_dic = {}
processed_usr_cnt = 0

user_df = df.sort_values(['reviewerID', 'unixReviewTime']).groupby('reviewerID')

for uid, hist in tqdm(user_df):
    processed_usr_cnt = processed_usr_cnt + 1
    
    item_hist = hist['asin'].tolist()
    
    user_item_dic[uid] = set(item_hist)
    
    item_hist = item_hist[-MAX_LEN_INP:]
    
    seq_len = len(item_hist)
    len_test = round(TEST_RATIO * seq_len)
    if len_test<=0:
        len_test = 1
    len_train = seq_len - len_test
    
    
    train_set_single.append([uid,item_hist[:-1]])
    test_set_single.append([uid,item_hist[:-1],item_hist[-1]])

    
    train_set_multi.append([uid,item_hist[:len_train]])
    add_all_sub_seq = False
    if add_all_sub_seq:
        for i in range(len_test):
            test_set_multi.append([uid,item_hist[:len(item_hist)-1-i],item_hist[-1-i:]])
    else:
        i = len_test-1
        test_set_multi.append([uid,item_hist[:len(item_hist)-1-i],item_hist[-1-i:]])


# In[128]:


dataset_infos = {}
dataset_infos['feature_size'] = feature_size
dataset_infos['max_seq_len'] = MAX_LEN
dataset_infos['min_seq_len'] = MIN_LEN
dataset_infos['test_ratio'] = TEST_RATIO
dataset_infos['max_item_id'] = item_cnt
dataset_infos['user_cnt'] = user_len
dataset_infos['user_item_dic'] = user_item_dic


# In[129]:


# save checkpoint
with open(DATASET_PKL_PATH, 'wb') as f:
    protocol = 2 # for legacy support
    pickle.dump(train_set_single, f, protocol)
    pickle.dump(train_set_multi, f, protocol)
    pickle.dump(test_set_single, f, protocol)
    pickle.dump(test_set_multi, f, protocol)
    pickle.dump(dataset_infos, f, protocol)


# In[2]:


# restore checkpoint
with open(DATASET_PKL_PATH, 'rb') as f:
    train_set_single = pkl.load(f)
    train_set_multi = pkl.load(f)
    test_set_single = pkl.load(f)
    test_set_multi = pkl.load(f)
    dataset_infos = pkl.load(f)
