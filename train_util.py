from sklearn.metrics import *
import sys
import tensorflow as tf
import numpy as np
import modelV2
import time
from tqdm import tqdm_notebook
import data_loader


def calculate_group_metric(labels, preds, users, calc_gauc=True, calc_ndcg=True, calc_hit=True, calc_mrr=True,
                           at_Ns=None):
    if at_Ns is None:
        at_Ns = [1, 5, 10]
    metrics = {}

    user_pred_dict = {}

    print_time_cost = False

    for i in range(len(users)):
        if users[i] in user_pred_dict:
            user_pred_dict[users[i]][0].append(preds[i])
            user_pred_dict[users[i]][1].append(labels[i])
        else:
            user_pred_dict[users[i]] = [[preds[i]], [labels[i]]]

    if calc_gauc:
        t = time.time()
        user_aucs = []
        valid_sample_num = 0
        for u in user_pred_dict:
            if 1 in user_pred_dict[u][1] and 0 in user_pred_dict[u][1]:  # contains both labels
                user_aucs.append(len(user_pred_dict[u][1]) * roc_auc_score(user_pred_dict[u][1], user_pred_dict[u][0]))
                valid_sample_num = len(user_pred_dict[u][1]) + valid_sample_num
        valid_group_num = len(user_aucs)
        total_group_num = len(user_pred_dict)
        total_sample_num = len(labels)
        metrics['gauc'] = (
            sum(user_aucs) / valid_sample_num, valid_group_num, valid_sample_num, total_group_num, total_sample_num)
        if print_time_cost:
            print("GAUC TIME: %.4fs" % (time.time() - t))

    t = time.time()
    if calc_ndcg or calc_hit or calc_mrr:
        for user, val in user_pred_dict.items():
            idx = np.argsort(val[0])[::-1]
            user_pred_dict[user][0] = np.array(val[0])[idx]
            user_pred_dict[user][1] = np.array(val[1])[idx]

    if calc_ndcg or calc_hit or calc_mrr:
        ndcg = np.zeros(len(at_Ns))
        hit = np.zeros(len(at_Ns))
        mrr = np.zeros(len(at_Ns))
        valid_user = 0
        for u in user_pred_dict:
            if 1 in user_pred_dict[u][1] and 0 in user_pred_dict[u][1]:  # contains both labels
                valid_user += 1
                pred = user_pred_dict[u][1]
                rank = np.nonzero(pred)[0]
                for idx, n in enumerate(at_Ns):
                    if rank < n:
                        ndcg[idx] += 1 / np.log2(rank + 2)
                        hit[idx] += 1
                        mrr[idx] += 1 / (rank + 1)
        ndcg = ndcg / valid_user
        hit = hit / valid_user
        mrr = mrr / valid_user
        metrics['ndcg'] = ndcg
        metrics['hit'] = hit
        metrics['mrr'] = mrr
        if print_time_cost:
            print("NDCG TIME: %.4fs" % (time.time() - t))

    return metrics


def evaluate(model_, sess, dataloader, show_preds=False):
    preds = []
    labels = []
    users = []

    t = time.time()
    show_progress = False
    # first_time = True
    for idx, batch_data in (tqdm_notebook(dataloader) if show_progress else dataloader):
        user_id, hist_seq, max_seq_len, target_id, label = batch_data
        pred = model_.eval(sess, batch_data)
        # pred,attn_score,attn_emb,attn_emb_seq = model_.eval(sess, batch_data)
        # if first_time:
        #     show_num = 3
        #     print('pred')
        #     print(pred[:show_num])
        #     print('target_id')
        #     print(target_id[:show_num])
        #     print('label')
        #     print(label[:show_num])
        #     print('batch_data')
        #     print(batch_data[1][:show_num])
        #     print('attn_score')
        #     for i in attn_score:
        #         print(i[:show_num])
        #     print('attn_emb_seq')
        #     for i in attn_emb_seq:
        #         print(i[:show_num])
        #     print('attn_emb')
        #     for i in attn_emb:
        #         print(i[:show_num])
        #     first_time = False
        users += user_id.tolist()
        preds += pred
        labels += label.tolist()

    if show_progress:
        print("EVAL TIME: %.4fs" % (time.time() - t))

    if show_preds:
        print('preds')
        print(preds[:100])
        print('labels')
        print(labels[:100])

    t = time.time()
    logloss = log_loss(labels, preds)
    auc_ = roc_auc_score(labels, preds)
    if show_progress:
        print("LOSS AND AUC TIME: %.4fs" % (time.time() - t))
    metrics = calculate_group_metric(labels, preds, users)
    gauc, valid_group_num, valid_sample_num, total_group_num, total_sample_num = metrics['gauc']
    ndcg = metrics['ndcg']
    hit = metrics['hit']
    mrr = metrics['mrr']
    # print('auc:{} gauc:{} ndcg:{} hit:{} mrr:{}'.format(auc,gauc,ndcg,hit,mrr))
    return logloss, auc_, gauc, ndcg, hit, mrr


def train_V2(model_type, train_set, test_set_multi, test_set_single, max_seq_len, lr, reg_lambda, keep_prob,
             eval_steps, emb_sz, hidden_sz, feature_size, train_batch_size, eval_batch_size, dataset_infos, max_epoch,
             max_pred_user, show_preds=False):
    sub_seq = False
    if model_type == 'POP':
        model_ = modelV2.POP(feature_size, emb_sz, hidden_sz, max_seq_len)
    elif model_type == 'SASRec':
        model_ = modelV2.SASRec(feature_size, emb_sz, hidden_sz, max_seq_len)
    elif model_type == 'GRU4Rec':
        model_ = modelV2.GRU4Rec(feature_size, emb_sz, hidden_sz, max_seq_len)
    elif model_type == 'LSTM4Rec':
        model_ = modelV2.LSTM4Rec(feature_size, emb_sz, hidden_sz, max_seq_len)
    elif model_type == 'BPR':
        model_ = modelV2.BPR(feature_size, emb_sz, hidden_sz, max_seq_len)
    elif model_type == 'DIN':
        model_ = modelV2.DIN(feature_size, emb_sz, hidden_sz, max_seq_len)
        sub_seq = True
    elif model_type == 'CASER':
        model_ = modelV2.CASER(feature_size, emb_sz, hidden_sz, max_seq_len)
        sub_seq = True
    elif model_type == 'DIEN':
        model_ = modelV2.DIEN(feature_size, emb_sz, hidden_sz, max_seq_len)
    elif model_type == 'NCF':
        model_ = modelV2.NCF(feature_size, emb_sz, hidden_sz, max_seq_len)
        sub_seq = True
    else:
        print('WRONG model_ TYPE')
        sys.exit(1)

    # gpu settings
    gpu_options = tf.GPUOptions(allow_growth=True)
    # training process
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        test_losses = []
        test_aucs = []
        test_gaucs = []
        test_ndcgs = []
        test_hits = []
        test_mrrs = []

        # before training process
        step = 0
        # begin training process
        for epoch in range(max_epoch):
            print('epoch {}'.format(epoch))
            dl_train = data_loader.DataLoaderV1(train_set, dataset_infos['max_item_id'], train_batch_size,
                                                dataset_infos['user_item_dic'], training=True, seq_len=max_seq_len,
                                                train_each_item=True,
                                                test_multi=True, epoch=epoch, train_subseq=sub_seq)

            for _, batch_data in dl_train:
                model_.train(sess, batch_data, lr, reg_lambda, keep_prob)
                step += 1
                if step % eval_steps == 0 and step >= 0:
                    print('begin_eval')
                    dl_test = data_loader.DataLoaderV1(test_set_single[:max_pred_user], dataset_infos['max_item_id'],
                                                       eval_batch_size, dataset_infos['user_item_dic'], training=False,
                                                       seq_len=max_seq_len, train_each_item=False,
                                                       test_multi=False, test_neg_num=100)

                    test_loss, test_auc, test_gauc, ndcg, hit, mrr = evaluate(model_, sess, dl_test, show_preds)

                    test_losses.append(test_loss)
                    test_aucs.append(test_auc)
                    test_gaucs.append(test_gauc)
                    test_ndcgs.append(ndcg)
                    test_hits.append(hit)
                    test_mrrs.append(mrr)

                    print(
                        ("Step{} loss[{:.4f}] auc[{:.4f}] gauc[{:.4f}] ndcg5[{:.4f}] " +
                         "ndcg10[{:.4f}] hit5[{:.4f}] hit10[{:.4f}] mrr[{:.4f}]").format(
                            step, test_loss, test_auc, test_gauc, ndcg[1], ndcg[2], hit[1], hit[2], mrr[0]))
        max_idx = np.argmax(test_gaucs)

        return (test_losses[max_idx], test_aucs[max_idx], test_gaucs[max_idx], test_ndcgs[max_idx], test_hits[max_idx],
                test_mrrs[max_idx])


def train_V3(model_type, train_set, test_set_multi, test_set_single, max_seq_len, lr, reg_lambda, keep_prob,
             eval_steps, emb_sz, hidden_sz, feature_size, train_batch_size, eval_batch_size, dataset_infos, max_epoch,
             max_pred_user, show_preds=False, mrif_parallel=False, mrif_kernels=None, mrif_attn_type='allrows',
             mrif_max_type='ind',
             pretrain_epoch=10, mrif_alpha=1.0):
    if mrif_kernels is None:
        mrif_kernels = [2, 2, 2]
    if model_type == 'MRIF_avg':
        model_ = modelV2.MRIF_avg(feature_size, emb_sz, hidden_sz, max_seq_len, mrif_parallel=mrif_parallel,
                                  mrif_kernels=mrif_kernels, pretrain_epoch=pretrain_epoch, mrif_alpha=mrif_alpha)
    elif model_type == 'MRIF_attn':
        model_ = modelV2.MRIF_attn(feature_size, emb_sz, hidden_sz, max_seq_len, mrif_parallel=mrif_parallel,
                                   mrif_kernels=mrif_kernels, mrif_attn_type=mrif_attn_type,
                                   pretrain_epoch=pretrain_epoch, mrif_alpha=mrif_alpha)
    elif model_type == 'MRIF_max':
        model_ = modelV2.MRIF_max(feature_size, emb_sz, hidden_sz, max_seq_len, mrif_parallel=mrif_parallel,
                                  mrif_kernels=mrif_kernels, mrif_max_type=mrif_max_type, pretrain_epoch=pretrain_epoch,
                                  mrif_alpha=mrif_alpha)
    else:
        print('WRONG model_ TYPE')
        sys.exit(1)

    # gpu settings
    gpu_options = tf.GPUOptions(allow_growth=True)
    # training process
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        test_losses = []
        test_aucs = []
        test_gaucs = []
        test_ndcgs = []
        test_hits = []
        test_mrrs = []

        # before training process
        step = 0
        # begin training process
        for epoch in range(max_epoch):
            print('epoch {}'.format(epoch))

            sub_seq = False if epoch < pretrain_epoch else True
            dl_train = data_loader.DataLoaderV1(train_set, dataset_infos['max_item_id'], train_batch_size,
                                                dataset_infos['user_item_dic'], training=True, seq_len=max_seq_len,
                                                train_each_item=True,
                                                test_multi=True, epoch=epoch, train_subseq=sub_seq)

            for _, batch_data in dl_train:
                model_.train(sess, batch_data, lr, reg_lambda, keep_prob, epoch)
                step += 1
                if step % eval_steps == 0 and step >= 0:
                    print('begin_eval')

                    dl_test = data_loader.DataLoaderV1(test_set_single[:max_pred_user], dataset_infos['max_item_id'],
                                                       eval_batch_size, dataset_infos['user_item_dic'], training=False,
                                                       seq_len=max_seq_len, train_each_item=False,
                                                       test_multi=False, test_neg_num=100)

                    test_loss, test_auc, test_gauc, ndcg, hit, mrr = evaluate(model_, sess, dl_test, show_preds)

                    test_losses.append(test_loss)
                    test_aucs.append(test_auc)
                    test_gaucs.append(test_gauc)
                    test_ndcgs.append(ndcg)
                    test_hits.append(hit)
                    test_mrrs.append(mrr[0])

                    print(
                        ("Step{} loss[{:.4f}] auc[{:.4f}] gauc[{:.4f}] ndcg5[{:.4f}] ndcg10[{:.4f}]" +
                         " hit5[{:.4f}] hit10[{:.4f}] mrr[{:.4f}]").format(
                            step, test_loss, test_auc, test_gauc, ndcg[1], ndcg[2], hit[1], hit[2], mrr[0]))

        max_idx = np.argmax(test_mrrs)
        max_idx2 = np.argmax(test_gaucs)

        return (test_losses[max_idx2], test_aucs[max_idx2], test_gaucs[max_idx2], test_ndcgs[max_idx], test_hits[max_idx],
                test_mrrs[max_idx])
