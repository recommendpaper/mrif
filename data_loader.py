import random
import numpy as np


class DataLoaderV1:
    def __init__(self, dataset, max_item_id, batch_size, user_item_dic, training=True, seq_len=100,
                 train_each_item=True,
                 test_multi=True, test_neg_num=100, epoch=0, train_neg_by_prob=False,
                 test_neg_by_prob=False, train_subseq=False):
        self.batch_size = batch_size
        self.max_item_id = max_item_id
        self.dataset = dataset
        self.training = training
        self.user_item_dic = user_item_dic
        self.num_of_step = len(dataset) // self.batch_size
        if self.batch_size * self.num_of_step < len(dataset):
            self.num_of_step += 1
        self.i = 0  # current position in dataset
        self.seq_len = seq_len
        self.training = training
        self.train_each_item = train_each_item
        self.test_multi = test_multi
        self.cache_data = None
        self.test_neg_num = test_neg_num

        self.train_neg_by_prob = train_neg_by_prob
        self.test_neg_by_prob = test_neg_by_prob

        self.train_subseq = train_subseq

        # get item probability
        if train_neg_by_prob or test_neg_by_prob:
            self.item_prob = np.zeros(max_item_id + 1)
            for item_set in user_item_dic.values():
                for item in item_set:
                    self.item_prob[item] += 1
            self.item_prob = self.item_prob / np.sum(self.item_prob)

        if self.training:
            np.random.seed(1111 * (epoch + 1))
            random.seed(1111 * (epoch + 1))
        else:
            np.random.seed(1212)
            random.seed(1212)

    def __iter__(self):
        return self

    def __next__(self):
        def pad_seq(m_seq, n, pad_front=False):
            if n > len(m_seq):
                if pad_front:
                    m_seq = [0] * (n - len(m_seq)) + m_seq
                else:
                    m_seq = m_seq + [0] * (n - len(m_seq))
            return m_seq

        def trunc_seq(m_seq, n, drop_front=True):
            if n < len(m_seq):
                if drop_front:
                    m_seq = m_seq[-n:]
                else:
                    m_seq = m_seq[:n]
            return m_seq

        def norm_seq(m_seq, n):
            m_seq = pad_seq(m_seq, n)
            m_seq = trunc_seq(m_seq, n)
            return m_seq

        def get_neg_ids_uniform(m_user_id, n):
            s = self.user_item_dic.get(m_user_id, set())
            if n == 1:
                t = random.randint(1, self.max_item_id)  # [] include both ends
                while t in s:
                    t = random.randint(1, self.max_item_id)  # [] include both ends
                return t
            else:
                m_res = []
                if n < 1000:
                    for _ in range(n):
                        t = random.randint(1, self.max_item_id)  # [] include both ends
                        while t in s:
                            t = random.randint(1, self.max_item_id)  # [] include both ends
                        m_res.append(t)
                else:
                    t = np.random.randint(1, self.max_item_id + 1, round(n * 1.5))  # [) exclude right
                    t = t[np.in1d(t, list(s), invert=True)]
                    t = t[:n]
                    while len(t) < n:
                        t = np.random.randint(1, self.max_item_id + 1, round(n * 1.5))  # [) exclude right
                        t = t[np.in1d(t, list(s), invert=True)]
                        t = t[:n]
                    m_res = t.tolist()
            return m_res

        def get_neg_ids_prop(m_user_id, n):
            s = self.user_item_dic.get(m_user_id, set())
            sample_num = max(n * 2, 50)
            t = np.random.choice(len(self.item_prob), size=sample_num, replace=False, p=self.item_prob)
            t = t[np.in1d(t, list(s), invert=True)]
            t = t[:n]
            while len(t) < n:
                t = np.random.choice(len(self.item_prob), size=sample_num, replace=False, p=self.item_prob)
                t = t[np.in1d(t, list(s), invert=True)]
                t = t[:n]
            m_res = t.tolist()
            if n == 1:
                return m_res[0]
            return m_res

        if (self.training and self.train_neg_by_prob) or ((not self.training) and self.test_neg_by_prob):
            get_neg_ids = get_neg_ids_prop
        else:
            get_neg_ids = get_neg_ids_uniform

        if self.training and (not self.train_subseq) and self.i == self.num_of_step:
            raise StopIteration

        if self.training and self.train_subseq and self.i == self.num_of_step and len(self.cache_data[0]) == 0:
            raise StopIteration

        if (not self.training) and self.i == self.num_of_step and len(self.cache_data[0]) == 0:
            raise StopIteration

        def select_one_batch_from_cache():
            # Cache data is in the format of feature tuple: (feat1,feat2,...,featN)
            # Each feature is a numpy array.
            m_res, self.cache_data = tuple(x[:self.batch_size] for x in self.cache_data), tuple(
                x[self.batch_size:] for x in self.cache_data)
            return m_res

        if self.training:
            if not self.train_subseq:
                if self.train_each_item:
                    # Outputs are:
                    #   user_id: [B]
                    #   item_ids: [B x T]
                    #   seq_lens: [B]
                    #   pos_ids: [B x T]
                    #   neg_ids: [B x T]
                    ts = self.dataset[self.i * self.batch_size: min(len(self.dataset), (self.i + 1) * self.batch_size)]
                    user_id = []
                    hist_seq = []
                    seq_len = []
                    pos_ids = []
                    neg_ids = []
                    for i in ts:
                        user_id.append(i[0])
                        hist_seq.append(norm_seq(i[1][:-1], self.seq_len))
                        s_len = min([len(i[1][:-1]), self.seq_len])
                        seq_len.append(s_len)
                        pos_ids.append(norm_seq(i[1][-s_len:], self.seq_len))
                        neg_ids.append(norm_seq(get_neg_ids(i[0], s_len), self.seq_len))
                    self.i += 1

                    return self.i, (np.array(user_id), np.array(hist_seq), np.array(seq_len),
                                    np.array(pos_ids), np.array(neg_ids))
                else:
                    # Outputs are:
                    #   user_id: [B]
                    #   item_ids: [B x T]
                    #   seq_lens: [B]
                    #   pos_id: [B]
                    #   neg_id: [B]
                    ts = self.dataset[self.i * self.batch_size: min(len(self.dataset), (self.i + 1) * self.batch_size)]
                    user_id = []
                    hist_seq = []
                    seq_len = []
                    pos_id = []
                    neg_id = []
                    for i in ts:
                        user_id.append(i[0])
                        hist_seq.append(norm_seq(i[1][:-1], self.seq_len))
                        seq_len.append(min([len(i[1][:-1]), self.seq_len]))
                        pos_id.append(i[1][-1])
                        neg_id.append(get_neg_ids(i[0], 1))
                    self.i += 1

                    return self.i, (np.array(user_id), np.array(hist_seq), np.array(seq_len),
                                    np.array(pos_id), np.array(neg_id))
            else:
                if self.train_each_item:
                    # Outputs are:
                    #   user_id: [B]
                    #   item_ids: [B x T]
                    #   seq_lens: [B]
                    #   pos_ids: [B x T]
                    #   neg_ids: [B x T]

                    batch_size = self.batch_size
                    if self.cache_data is not None and (
                            len(self.cache_data[0]) >= batch_size or self.i == self.num_of_step):
                        res = select_one_batch_from_cache()
                        return self.i, res
                    else:
                        ts = self.dataset[
                             self.i * self.batch_size: min(len(self.dataset), (self.i + 1) * self.batch_size)]
                        self.i += 1

                        user_id = []
                        hist_seq = []
                        seq_len = []
                        pos_ids = []
                        neg_ids = []
                        for rec in ts:
                            # loop through sub sequences
                            seq = rec[1]
                            user = rec[0]
                            for head in range(1):
                                # for head in range(len(seq)-1):
                                for tail in range(head + 1, len(seq)):
                                    i = (user, seq[head:tail + 1])
                                    user_id.append(i[0])
                                    hist_seq.append(norm_seq(i[1][:-1], self.seq_len))
                                    s_len = min([len(i[1][:-1]), self.seq_len])
                                    seq_len.append(s_len)
                                    pos_ids.append(norm_seq(i[1][-s_len:], self.seq_len))
                                    neg = get_neg_ids(i[0], s_len)
                                    if s_len == 1:
                                        neg = [neg]
                                    neg_ids.append(norm_seq(neg, self.seq_len))

                        user_id, hist_seq, seq_len, pos_ids, neg_ids = (np.array(x) for x in
                                                                        [user_id, hist_seq, seq_len, pos_ids, neg_ids])
                        idx = np.random.permutation(len(user_id))

                        user_id, hist_seq, seq_len, pos_ids, neg_ids = (x[idx] for x in
                                                                        [user_id, hist_seq, seq_len, pos_ids, neg_ids])

                        if self.cache_data is None:
                            self.cache_data = (user_id, hist_seq, seq_len, pos_ids, neg_ids)
                        else:
                            self.cache_data = (
                                np.concatenate([self.cache_data[0], user_id]),
                                np.concatenate([self.cache_data[1], hist_seq]),
                                np.concatenate([self.cache_data[2], seq_len]),
                                np.concatenate([self.cache_data[3], pos_ids]),
                                np.concatenate([self.cache_data[4], neg_ids]),
                            )
                        if self.cache_data is not None and (
                                len(self.cache_data[0]) >= batch_size or self.i == self.num_of_step):
                            res = select_one_batch_from_cache()
                            return self.i, res

        else:
            if self.test_multi:
                # Outputs are:
                #   user_id: [B]
                #   item_ids: [B x T]
                #   seq_lens: [B]
                #   target_id: [B]
                #   label: [B]

                #   Each user has multiple test entries with different target_ids. Positive target_ids take values
                #   from clicked items. Negative target_ids are randomly sampled.

                batch_size = self.batch_size
                if self.cache_data is not None and (
                        len(self.cache_data[0]) >= batch_size or self.i == self.num_of_step):
                    res = select_one_batch_from_cache()
                    return self.i, res
                else:
                    ts = self.dataset[self.i * self.batch_size: min(len(self.dataset), (self.i + 1) * self.batch_size)]
                    self.i += 1
                    user_id = []
                    hist_seq = []
                    seq_len = []
                    target_id = []
                    label = []

                    for i in ts:
                        for j in i[2]:
                            user_id.append(i[0])
                            hist_seq.append(norm_seq(i[1], self.seq_len))
                            seq_len.append(min([len(i[1]), self.seq_len]))
                            target_id.append(j)
                            label.append(1)
                        for j in get_neg_ids(i[0], self.test_neg_num):
                            user_id.append(i[0])
                            hist_seq.append(norm_seq(i[1], self.seq_len))
                            seq_len.append(min([len(i[1]), self.seq_len]))
                            target_id.append(j)
                            label.append(0)

                    user_id = np.array(user_id)
                    hist_seq = np.array(hist_seq)
                    seq_len = np.array(seq_len)
                    target_id = np.array(target_id)
                    label = np.array(label)

                    if self.cache_data is None:
                        self.cache_data = (user_id, hist_seq, seq_len, target_id, label)
                    else:
                        self.cache_data = (
                            np.concatenate([self.cache_data[0], user_id]),
                            np.concatenate([self.cache_data[1], hist_seq]),
                            np.concatenate([self.cache_data[2], seq_len]),
                            np.concatenate([self.cache_data[3], target_id]),
                            np.concatenate([self.cache_data[4], label]),
                        )
                    if self.cache_data is not None and (
                            len(self.cache_data[0]) >= batch_size or self.i == self.num_of_step):
                        res = select_one_batch_from_cache()
                        return self.i, res

            else:
                # Outputs are:
                #   user_id: [B]
                #   item_ids: [B x T]
                #   seq_lens: [B]
                #   target_id: [B]
                #   label: [B]
                #   Each user has multiple test entries with different target_ids. Positive target_id is next clicked
                #   item. Negative target_id is randomly sampled.
                batch_size = self.batch_size
                if self.cache_data is not None and (
                        len(self.cache_data[0]) >= batch_size or self.i == self.num_of_step):
                    res = select_one_batch_from_cache()
                    return self.i, res
                else:
                    ts = self.dataset[self.i * self.batch_size: min(len(self.dataset), (self.i + 1) * self.batch_size)]
                    self.i += 1
                    user_id = []
                    hist_seq = []
                    seq_len = []
                    target_id = []
                    label = []

                    for i in ts:
                        j = i[2]
                        user_id.append(i[0])
                        hist_seq.append(norm_seq(i[1], self.seq_len))
                        seq_len.append(min([len(i[1]), self.seq_len]))
                        target_id.append(j)
                        label.append(1)
                        for j in get_neg_ids(i[0], self.test_neg_num):
                            user_id.append(i[0])
                            hist_seq.append(norm_seq(i[1], self.seq_len))
                            seq_len.append(min([len(i[1]), self.seq_len]))
                            target_id.append(j)
                            label.append(0)

                    user_id = np.array(user_id)
                    hist_seq = np.array(hist_seq)
                    seq_len = np.array(seq_len)
                    target_id = np.array(target_id)
                    label = np.array(label)

                    if self.cache_data is None:
                        self.cache_data = (user_id, hist_seq, seq_len, target_id, label)
                    else:
                        self.cache_data = (
                            np.concatenate([self.cache_data[0], user_id]),
                            np.concatenate([self.cache_data[1], hist_seq]),
                            np.concatenate([self.cache_data[2], seq_len]),
                            np.concatenate([self.cache_data[3], target_id]),
                            np.concatenate([self.cache_data[4], label]),
                        )
                    if self.cache_data is not None and (
                            len(self.cache_data[0]) >= batch_size or self.i == self.num_of_step):
                        res = select_one_batch_from_cache()
                        return self.i, res

    def next(self):
        return self.__next__()
