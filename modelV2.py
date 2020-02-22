import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from util import *
from rnn import dynamic_rnn
import numpy as np


class ModelBase(object):
    def __init__(self, feature_size, emb_dim, hidden_size, max_seq_len, pairwise_loss=False):
        # reset graph
        tf.reset_default_graph()
        self.hidden_size = hidden_size
        self.emb_dim = emb_dim
        self.max_seq_len = max_seq_len
        self.pairwise_loss = pairwise_loss
        self.feature_size = feature_size

        # Input placeholders
        with tf.name_scope('Inputs'):
            self.seq_id = tf.placeholder(tf.int32, [None, max_seq_len], name='seq_id')
            self.seq_len = tf.placeholder(tf.int32, [None, ], name='seq_len')
            self.pos_id = tf.placeholder(tf.int32, [None, max_seq_len], name='pos_id')
            self.neg_id = tf.placeholder(tf.int32, [None, max_seq_len], name='neg_id')
            self.test_id = tf.placeholder(tf.int32, [None, ], name='test_id')
            self.user_id = tf.placeholder(tf.int32, [None, ], name='user_id')

            # 1-dropout
            self.keep_prob = tf.placeholder(tf.float32)
            self.drop_prob = 1 - self.keep_prob

            # learning rate
            self.lr = tf.placeholder(tf.float64, [])

            # reg lambda
            self.reg_lambda = tf.placeholder(tf.float32, [])

            # If is training
            self.training = tf.placeholder(tf.bool, [])

        # Item and User Mask
        with tf.variable_scope('Mask', reuse=tf.AUTO_REUSE):
            self.seq_mask = tf.sequence_mask(self.seq_len, tf.shape(self.seq_id)[1], dtype=tf.float32)  # [B, T]
            self.seq_mask_1 = tf.sequence_mask(self.seq_len - 1, tf.shape(self.seq_id)[1], dtype=tf.float32)  # [B, T]
            self.seq_mask_bool = tf.sequence_mask(self.seq_len, tf.shape(self.seq_id)[1], dtype=tf.bool)  # [B, T]

        # Embedding Layer
        with tf.variable_scope('Embedding', reuse=tf.AUTO_REUSE):
            # Entry 0 is zero embedding that corresponds to empty position in a sequence.
            self.emb_mtx = tf.concat(
                [tf.zeros(shape=[1, emb_dim]), tf.get_variable('emb_mtx', [feature_size - 1, emb_dim])], axis=0)
            self.raw_seq_emb = tf.nn.embedding_lookup(self.emb_mtx, self.seq_id)  # [B, T,  EMB]
            self.user_emb = tf.nn.embedding_lookup(self.emb_mtx, self.user_id)  # [B, EMB]
            self.test_emb = tf.nn.embedding_lookup(self.emb_mtx, self.test_id)  # [B, EMB]

            self.positional_emb_mtx = tf.get_variable('positional_emb_mtx', [max_seq_len, emb_dim])
            self.positional_emb = tf.nn.embedding_lookup(self.positional_emb_mtx,
                                                         tf.tile(tf.expand_dims(tf.range(self.max_seq_len), 0),
                                                                 [tf.shape(self.seq_id)[0], 1]))

            self.pos_emb = tf.nn.embedding_lookup(self.emb_mtx, self.pos_id)  # [B, T, EMB]
            self.neg_emb = tf.nn.embedding_lookup(self.emb_mtx, self.neg_id)  # [B, T, EMB]

            self.last_pos_item = tf.reduce_sum(tf.expand_dims((self.seq_mask - self.seq_mask_1), -1) * self.pos_emb,
                                               axis=1)  # [B, EMB]
            self.last_neg_item = tf.reduce_sum(tf.expand_dims((self.seq_mask - self.seq_mask_1), -1) * self.neg_emb,
                                               axis=1)  # [B, EMB]
            self.last_item = tf.reduce_sum(tf.expand_dims((self.seq_mask - self.seq_mask_1), -1) * self.raw_seq_emb,
                                           axis=1)  # [B, EMB]

        self.build_main_graph()

        with tf.variable_scope('training', reuse=tf.AUTO_REUSE):
            for v in tf.trainable_variables():
                print(v)
                if 'bias' not in v.name and 'emb_mtx' not in v.name:
                    self.loss += self.reg_lambda * tf.nn.l2_loss(v)
            # optimizer and training step
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.train_step = self.optimizer.minimize(self.loss)

    def build_main_graph(self):
        """Overide me"""
        pass

    def positive_part_loss(self, logits):
        """ -log(sigmoid(x)) = max(x, 0) - x  + log(1 + exp(-abs(x))) """
        zeros = tf.zeros_like(logits, dtype=logits.dtype)
        cond = (logits >= zeros)
        relu_logits = tf.where(cond, logits, zeros)
        neg_abs_logits = tf.where(cond, -logits, logits)
        return relu_logits - logits + tf.log1p(tf.exp(neg_abs_logits))

    def negative_part_loss(self, logits):
        """ -log(1-sigmoid(x)) = max(x, 0)   + log(1 + exp(-abs(x))) """
        zeros = tf.zeros_like(logits, dtype=logits.dtype)
        cond = (logits >= zeros)
        relu_logits = tf.where(cond, logits, zeros)
        neg_abs_logits = tf.where(cond, -logits, logits)
        return relu_logits + tf.log1p(tf.exp(neg_abs_logits))

    def build_fc_net(self, inp, output_dim=1, name='fc'):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # inp = tf.layers.batch_normalization(inputs=inp, name='bn1')
            fc1 = tf.layers.dense(inp, 200, activation=tf.nn.tanh, name='fc1')
            dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
            fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.tanh, name='fc2')
            dp2 = tf.nn.dropout(fc2, self.keep_prob, name='dp2')
            fc3 = tf.layers.dense(dp2, output_dim, activation=None, name='fc3')
        return fc3

    def train(self, sess, batch_data, lr, reg_lambda, keep_prob):
        loss, _ = sess.run([self.loss, self.train_step], feed_dict={
            self.user_id: batch_data[0],
            self.seq_id: batch_data[1],
            self.seq_len: batch_data[2],
            self.pos_id: batch_data[3],
            self.neg_id: batch_data[4],
            self.lr: lr,
            self.reg_lambda: reg_lambda,
            self.keep_prob: keep_prob,
            self.training: True
        })
        return loss

    def eval(self, sess, batch_data):
        pred = sess.run([self.y_pred], feed_dict={
            self.user_id: batch_data[0],
            self.seq_id: batch_data[1],
            self.seq_len: batch_data[2],
            self.test_id: batch_data[3],
            self.keep_prob: 1.0,
            self.training: False
        })
        return pred[0].reshape([-1, ]).tolist()

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from {}'.format(path))

    def prelu(self, _x, scope=None):
        """parametric ReLU activation"""
        with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
            _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                     dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)

    def attention(self, key, value, query, mask, name='attn'):
        """Vanilla attention.

        Attention score is softmax on output of feed forward network with `key` and `query` 
        as input.
        Output is weighted sum of `value` according to attention score.

        Args:
            key: A tensor to compute attention score with query with shape [B, T, EMB].
            value: A tensor on which weighted sum is computed. [B, T, EMB]
            query: A tensor to compute attention score with key with shape [B, EMB].
            mask: A tensor where non-empty positions in sequence is 1, otherwise is 0. [B, T]
        """
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            _, max_len, k_dim = key.get_shape().as_list()
            query = tf.layers.dense(query, k_dim, activation=None)
            query = self.prelu(query)
            queries = tf.tile(tf.expand_dims(query, 1), [1, max_len, 1])  # [B, T, Dk]
            inp = tf.concat([queries, key, queries - key, queries * key], axis=-1)
            fc1 = tf.layers.dense(inp, 80, activation=tf.nn.relu, name='fc1')
            fc2 = tf.layers.dense(fc1, 40, activation=tf.nn.relu, name='fc2')
            fc3 = tf.layers.dense(fc2, 1, activation=None, name='fc3')  # [B, T, 1]

            mask = tf.reshape(mask, [-1, max_len, 1])
            mask = tf.equal(mask, tf.ones_like(mask))  # [B, T, 1]
            paddings = tf.ones_like(fc3) * (-2 ** 32 + 1)
            score = tf.nn.softmax(tf.reshape(tf.where(mask, fc3, paddings), [-1, max_len]))  # [B, T]

            atten_output = tf.multiply(value, tf.expand_dims(score, 2))
            atten_output_sum = tf.reduce_sum(atten_output, axis=1)

        return atten_output_sum, atten_output, score

    def attention_v2(self, key, value, query, mask):
        """Vanilla attention.

        Attention score is softmax on inner product of `key` and `query`.
        Output is weighted sum of `value` according to attention score.

        Args:
            key: A tensor to compute attention score with query with shape [B, T, EMB].
            value: A tensor on which weighted sum is computed. [B, T, EMB]
            query: A tensor to compute attention score with key with shape [B, EMB].
            mask: A tensor where non-empty positions in sequence is 1, otherwise is 0. [B, T]
        """
        _, max_len, k_dim = key.get_shape().as_list()
        # query = tf.layers.dense(query, k_dim, activation=None) #[B, EMB]
        queries = tf.expand_dims(query, 1)  # [B, 1, EMB]
        product = tf.reduce_sum((queries * key), axis=-1)  # [B, T]

        mask = tf.reshape(tf.equal(mask, tf.ones_like(mask)), [-1, max_len])  # [B, T]
        paddings = tf.ones_like(product) * (-2 ** 32 + 1)
        score = tf.nn.softmax(tf.where(mask, product, paddings))  # [B, T]

        atten_output = tf.multiply(value, tf.expand_dims(score, 2))  # [B, T, Dv==Dk]
        atten_output_sum = tf.reduce_sum(atten_output, axis=1)  # [B, Dv==Dk]

        return atten_output_sum, atten_output, score


class POP(ModelBase):
    def __init__(self, feature_size, emb_dim, hidden_size, max_seq_len, pairwise_loss=False):
        super(POP, self).__init__(feature_size, emb_dim, hidden_size, max_seq_len, pairwise_loss)

    def build_main_graph(self):
        with tf.variable_scope('logits', reuse=tf.AUTO_REUSE):
            # Only consider target items without using sequence information.
            self.pos_logits = tf.reshape(self.build_fc_net(self.pos_emb), [-1, self.max_seq_len])  # [B, T]
            self.pos_logits = tf.boolean_mask(self.pos_logits, self.seq_mask_bool)  # [B,]

            self.neg_logits = tf.reshape(self.build_fc_net(self.neg_emb), [-1, self.max_seq_len])  # [B, T]
            self.neg_logits = tf.boolean_mask(self.neg_logits, self.seq_mask_bool)  # [B,]

            self.test_logits = tf.reshape(self.build_fc_net(self.test_emb), [-1, ])  # [B,]
            self.y_pred = tf.reshape(tf.nn.sigmoid(self.test_logits), [-1, ])

        with tf.variable_scope('loss'):
            if self.pairwise_loss:
                logits_diff = self.pos_logits - self.neg_logits
                self.log_loss = self.positive_part_loss(logits_diff)
            else:
                faster = True
                if faster:
                    self.log_loss = self.positive_part_loss(self.pos_logits) + self.negative_part_loss(self.neg_logits)
                else:
                    concat_label = tf.concat([tf.ones_like(self.pos_logits), tf.zeros_like(self.neg_logits)], axis=-1)
                    concat_logits = tf.concat([self.pos_logits, self.neg_logits], axis=-1)
                    self.log_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=concat_label, logits=concat_logits)
            self.loss = tf.reduce_sum(self.log_loss)


class BPR(ModelBase):
    def __init__(self, feature_size, emb_dim, hidden_size, max_seq_len, pairwise_loss=True):
        super(BPR, self).__init__(feature_size, emb_dim, hidden_size, max_seq_len, pairwise_loss)

    def build_main_graph(self):
        with tf.variable_scope('logits', reuse=tf.AUTO_REUSE):
            self.user_emb = tf.reshape(self.user_emb, [-1, 1, self.emb_dim])
            self.pos_logits = tf.reshape(tf.reduce_sum(self.user_emb * self.pos_emb, axis=-1),
                                         [-1, self.max_seq_len])  # [B, T]
            self.pos_logits = tf.boolean_mask(self.pos_logits, self.seq_mask_bool)  # [B,]

            self.neg_logits = tf.reshape(tf.reduce_sum(self.user_emb * self.neg_emb, axis=-1),
                                         [-1, self.max_seq_len])  # [B, T]
            self.neg_logits = tf.boolean_mask(self.neg_logits, self.seq_mask_bool)  # [B,]

            self.user_emb = tf.reshape(self.user_emb, [-1, self.emb_dim])
            self.test_logits = tf.reduce_sum(self.user_emb * self.test_emb, axis=-1)  # [B,]
            self.y_pred = tf.reshape(tf.nn.sigmoid(self.test_logits), [-1, ])

        with tf.variable_scope('loss'):
            if self.pairwise_loss:
                logits_diff = self.pos_logits - self.neg_logits
                self.log_loss = self.positive_part_loss(logits_diff)
            else:
                faster = True
                if faster:
                    self.log_loss = self.positive_part_loss(self.pos_logits) + self.negative_part_loss(self.neg_logits)
                else:
                    concat_label = tf.concat([tf.ones_like(self.pos_logits), tf.zeros_like(self.neg_logits)], axis=-1)
                    concat_logits = tf.concat([self.pos_logits, self.neg_logits], axis=-1)
                    self.log_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=concat_label, logits=concat_logits)
            self.loss = tf.reduce_sum(self.log_loss)


class GRU4Rec(ModelBase):
    def __init__(self, feature_size, emb_dim, hidden_size, max_seq_len, pairwise_loss=False):
        super(GRU4Rec, self).__init__(feature_size, emb_dim, hidden_size, max_seq_len, pairwise_loss)

    def build_main_graph(self):
        # RNN layer
        with tf.variable_scope('model'):
            rnn_out, final_state = tf.nn.dynamic_rnn(GRUCell(self.hidden_size), inputs=self.raw_seq_emb,
                                                     sequence_length=self.seq_len, dtype=tf.float32, scope='gru1')
            # project state into item embedding space
            self.seq_emb = tf.layers.dense(rnn_out, self.emb_dim, activation=None)  # [B, T, EMB]
            self.last_step_emb = extract_last_step(self.seq_emb, self.seq_len - 1)  # [B, EMB]

        with tf.variable_scope('logits'):
            self.pos_logits = tf.reduce_sum(self.seq_emb * self.pos_emb, axis=-1)  # [B, T]
            self.pos_logits = tf.boolean_mask(self.pos_logits, self.seq_mask_bool)
            self.neg_logits = tf.reduce_sum(self.seq_emb * self.neg_emb, axis=-1)  # [B, T]
            self.neg_logits = tf.boolean_mask(self.neg_logits, self.seq_mask_bool)
            self.test_logits = tf.reduce_sum(self.test_emb * self.last_step_emb, axis=-1)  # [B,]
            self.y_pred = tf.reshape(tf.nn.sigmoid(self.test_logits), [-1, ])

        with tf.variable_scope('loss'):
            if self.pairwise_loss:
                logits_diff = self.pos_logits - self.neg_logits
                self.log_loss = self.positive_part_loss(logits_diff)
            else:
                faster = True
                if faster:
                    self.log_loss = self.positive_part_loss(self.pos_logits) + self.negative_part_loss(self.neg_logits)
                else:
                    concat_label = tf.concat([tf.ones_like(self.pos_logits), tf.zeros_like(self.neg_logits)], axis=-1)
                    concat_logits = tf.concat([self.pos_logits, self.neg_logits], axis=-1)
                    self.log_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=concat_label, logits=concat_logits)

            self.loss = tf.reduce_sum(self.log_loss)


class LSTM4Rec(ModelBase):
    def __init__(self, feature_size, emb_dim, hidden_size, max_seq_len, pairwise_loss=False):
        super(LSTM4Rec, self).__init__(feature_size, emb_dim, hidden_size, max_seq_len, pairwise_loss)

    def build_main_graph(self):
        # RNN layer
        with tf.variable_scope('model'):
            rnn_out, final_state = tf.nn.dynamic_rnn(LSTMCell(self.hidden_size), inputs=self.raw_seq_emb,
                                                     sequence_length=self.seq_len, dtype=tf.float32, scope='lstm')
            # project state into item embedding space
            self.seq_emb = tf.layers.dense(rnn_out, self.emb_dim, activation=None)  # [B, T, EMB]
            self.last_step_emb = extract_last_step(self.seq_emb, self.seq_len - 1)  # [B, EMB]

        with tf.variable_scope('logits'):
            self.pos_logits = tf.reduce_sum(self.seq_emb * self.pos_emb, axis=-1)  # [B, T]
            self.pos_logits = tf.boolean_mask(self.pos_logits, self.seq_mask_bool)
            self.neg_logits = tf.reduce_sum(self.seq_emb * self.neg_emb, axis=-1)  # [B, T]
            self.neg_logits = tf.boolean_mask(self.neg_logits, self.seq_mask_bool)
            self.test_logits = tf.reduce_sum(self.test_emb * self.last_step_emb, axis=-1)  # [B,]
            self.y_pred = tf.reshape(tf.nn.sigmoid(self.test_logits), [-1, ])

        with tf.variable_scope('loss'):
            if self.pairwise_loss:
                logits_diff = self.pos_logits - self.neg_logits
                self.log_loss = self.positive_part_loss(logits_diff)
            else:
                faster = True
                if faster:
                    self.log_loss = self.positive_part_loss(self.pos_logits) + self.negative_part_loss(self.neg_logits)
                else:
                    concat_label = tf.concat([tf.ones_like(self.pos_logits), tf.zeros_like(self.neg_logits)], axis=-1)
                    concat_logits = tf.concat([self.pos_logits, self.neg_logits], axis=-1)
                    self.log_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=concat_label, logits=concat_logits)

            self.loss = tf.reduce_sum(self.log_loss)


class CASER(ModelBase):
    def __init__(self, feature_size, emb_dim, hidden_size, max_seq_len, pairwise_loss=False):
        super(CASER, self).__init__(feature_size, emb_dim, hidden_size, max_seq_len, pairwise_loss)

    def build_main_graph(self):
        with tf.variable_scope('SeqModel', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('item_cnn', reuse=tf.AUTO_REUSE):
                vertical_filter_num = 4
                horizontal_filter_num = 16
                v_kernel_size_item = [self.max_seq_len, 1]

                hist_item_emb = self.raw_seq_emb * tf.reshape(self.seq_mask, [-1, self.max_seq_len, 1])
                hist_item_emb = tf.expand_dims(hist_item_emb, 3)  # [B, T, EMB, 1]

                # horizontal filters
                item_hori_out = []
                for h in range(1, 9):
                    h_kernel_size_item = [h, self.emb_dim]
                    conv1 = tf.layers.conv2d(hist_item_emb, horizontal_filter_num, h_kernel_size_item,
                                             activation=tf.nn.relu,
                                             name='horizontal_conv_{}'.format(h))  # [B, T, 1, FILTER_NUM]
                    max1 = tf.layers.max_pooling2d(conv1, [conv1.get_shape().as_list()[1], 1],
                                                   1)  # [B, 1, 1, FILTER_NUM]
                    item_hori_out.append(tf.reshape(max1, [-1, horizontal_filter_num]))  # [B, FILTER_NUM]
                item_hori_out = tf.concat(item_hori_out, axis=-1)

                # vertical
                conv2 = tf.layers.conv2d(hist_item_emb, vertical_filter_num, v_kernel_size_item, activation=tf.nn.relu,
                                         name='vertical_conv')  # [B, 1, EMB, FILTER_NUM]
                item_vert_out = tf.reshape(conv2, [-1, self.emb_dim * vertical_filter_num])  # [B, EMB, FILTER_NUM]

                item_part = tf.concat([item_hori_out, item_vert_out, self.user_emb], axis=1)
            with tf.variable_scope('fc_output', reuse=tf.AUTO_REUSE):
                # project final state into item embedding space
                out_state = tf.layers.dense(item_part, self.emb_dim, activation=None, name='fc0')  # [B, EMB]
                out_state = tf.nn.dropout(out_state, self.keep_prob)
                self.seq_emb = out_state

            self.last_step_pos_emb = extract_last_step(self.pos_emb, self.seq_len - 1)  # [B, EMB]
            self.last_step_neg_emb = extract_last_step(self.neg_emb, self.seq_len - 1)  # [B, EMB]

        with tf.variable_scope('logits'):
            self.pos_logits = tf.reduce_sum(self.seq_emb * self.last_step_pos_emb, axis=-1)  # [B,]
            self.neg_logits = tf.reduce_sum(self.seq_emb * self.last_step_neg_emb, axis=-1)  # [B, T]
            self.test_logits = tf.reduce_sum(self.test_emb * self.seq_emb, axis=-1)  # [B,]
            self.y_pred = tf.reshape(tf.nn.sigmoid(self.test_logits), [-1, ])

        with tf.variable_scope('loss'):
            if self.pairwise_loss:
                logits_diff = self.pos_logits - self.neg_logits
                self.log_loss = self.positive_part_loss(logits_diff)
            else:
                faster = True
                if faster:
                    self.log_loss = self.positive_part_loss(self.pos_logits) + self.negative_part_loss(self.neg_logits)
                else:
                    concat_label = tf.concat([tf.ones_like(self.pos_logits), tf.zeros_like(self.neg_logits)], axis=-1)
                    concat_logits = tf.concat([self.pos_logits, self.neg_logits], axis=-1)
                    self.log_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=concat_label, logits=concat_logits)

            self.loss = tf.reduce_sum(self.log_loss)


class DIN(ModelBase):
    def __init__(self, feature_size, emb_dim, hidden_size, max_seq_len, pairwise_loss=False):
        super(DIN, self).__init__(feature_size, emb_dim, hidden_size, max_seq_len, pairwise_loss)

    def build_main_graph(self):
        with tf.variable_scope('logits', reuse=tf.AUTO_REUSE):
            pos_seq_emb, _, _ = self.attention(self.raw_seq_emb, self.raw_seq_emb, self.last_pos_item, self.seq_mask)
            neg_seq_emb, _, _ = self.attention(self.raw_seq_emb, self.raw_seq_emb, self.last_neg_item, self.seq_mask)
            test_seq_emb, _, _ = self.attention(self.raw_seq_emb, self.raw_seq_emb, self.test_emb, self.seq_mask)

            self.pos_logits = tf.reduce_sum(pos_seq_emb * self.last_pos_item, axis=-1)  # [B,]
            self.neg_logits = tf.reduce_sum(neg_seq_emb * self.last_neg_item, axis=-1)  # [B, T]
            self.test_logits = tf.reduce_sum(self.test_emb * test_seq_emb, axis=-1)  # [B,]
            self.y_pred = tf.reshape(tf.nn.sigmoid(self.test_logits), [-1, ])

        with tf.variable_scope('loss'):
            if self.pairwise_loss:
                logits_diff = self.pos_logits - self.neg_logits
                self.log_loss = self.positive_part_loss(logits_diff)
            else:
                faster = True
                if faster:
                    self.log_loss = self.positive_part_loss(self.pos_logits) + self.negative_part_loss(self.neg_logits)
                else:
                    concat_label = tf.concat([tf.ones_like(self.pos_logits), tf.zeros_like(self.neg_logits)], axis=-1)
                    concat_logits = tf.concat([self.pos_logits, self.neg_logits], axis=-1)
                    self.log_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=concat_label, logits=concat_logits)

            self.loss = tf.reduce_sum(self.log_loss)


class DIEN(ModelBase):
    def __init__(self, feature_size, emb_dim, hidden_size, max_seq_len, pairwise_loss=False):
        super(DIEN, self).__init__(feature_size, emb_dim, hidden_size, max_seq_len, pairwise_loss)

    def pointwise_pred(self, target_emb):
        with tf.variable_scope('dien_extraction', reuse=tf.AUTO_REUSE):
            item_part_output, _ = tf.nn.dynamic_rnn(GRUCell(self.hidden_size), inputs=self.raw_seq_emb,
                                                    sequence_length=self.seq_len, dtype=tf.float32, scope='gru1')
            _, _, item_score = self.attention(item_part_output, item_part_output, target_emb, self.seq_mask)
        with tf.variable_scope('dien_evolve', reuse=tf.AUTO_REUSE):
            _, item_part_final_state = dynamic_rnn(VecAttGRUCell(self.hidden_size), inputs=item_part_output,
                                                   att_scores=tf.expand_dims(item_score, -1),
                                                   sequence_length=self.seq_len, dtype=tf.float32, scope="argru1")
        # project final state into item embedding space
        with tf.variable_scope('fc', reuse=tf.AUTO_REUSE):
            out_state = tf.layers.dense(item_part_final_state, self.emb_dim, activation=None, name='fc')  # [B, Dk]
            res = tf.reduce_sum(out_state * target_emb, axis=-1)

        gru_out = item_part_output
        return res, gru_out

    def build_main_graph(self):
        with tf.variable_scope('logits', reuse=tf.AUTO_REUSE):
            self.pos_logits, gru_out = self.pointwise_pred(self.last_pos_item)
            self.neg_logits, _ = self.pointwise_pred(self.last_neg_item)
            self.test_logits, _ = self.pointwise_pred(self.test_emb)

            self.pos_logits_aux = tf.reduce_sum(gru_out * self.pos_emb, axis=-1)
            self.pos_logits_aux = tf.boolean_mask(self.pos_logits_aux, self.seq_mask_bool)

            self.neg_logits_aux = tf.reduce_sum(gru_out * self.neg_emb, axis=-1)
            self.neg_logits_aux = tf.boolean_mask(self.neg_logits_aux, self.seq_mask_bool)

            self.y_pred = tf.reshape(tf.nn.sigmoid(self.test_logits), [-1, ])

        with tf.variable_scope('loss'):
            if not self.pairwise_loss:
                self.loss = (2.0 * tf.reduce_sum(
                    self.positive_part_loss(self.pos_logits) + self.negative_part_loss(self.neg_logits))
                             + 1.0 * tf.reduce_sum(
                            self.positive_part_loss(self.pos_logits_aux) + self.negative_part_loss(
                                self.neg_logits_aux)))


class NCF(ModelBase):
    def __init__(self, feature_size, emb_dim, hidden_size, max_seq_len, pairwise_loss=False):
        super(NCF, self).__init__(feature_size, emb_dim, hidden_size, max_seq_len, pairwise_loss)

    def pointwise_pred(self, target_emb, target_emb2, name='ncf'):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            mf_target_item = tf.layers.dense(target_emb, self.hidden_size, activation=None,
                                             name='mf_item_fc')  # [B, EMB]
            mf_target_item = tf.nn.dropout(mf_target_item, self.keep_prob, name='dp1')
            mf_target_user = tf.layers.dense(self.user_emb, self.hidden_size, activation=None,
                                             name='mf_user_fc')  # [B, EMB]
            mf_target_user = tf.nn.dropout(mf_target_user, self.keep_prob, name='dp2')

            mlp_target_item = tf.layers.dense(target_emb2, self.hidden_size, activation=None,
                                              name='mlp_item_fc')  # [B, EMB]
            mlp_target_item = tf.nn.dropout(mlp_target_item, self.keep_prob, name='dp3')
            mlp_target_user = tf.layers.dense(self.user_emb2, self.hidden_size, activation=None,
                                              name='mlp_user_fc')  # [B, EMB]
            mlp_target_user = tf.nn.dropout(mlp_target_user, self.keep_prob, name='dp4')

            gmf = mf_target_item * mf_target_user
            mlp = tf.concat([mlp_target_item, mlp_target_user], axis=-1)

            units = [256, 128, 64]
            for idx, unit in enumerate(units):
                mlp = tf.layers.dense(mlp, unit, activation=tf.nn.relu, name='mlp_fc_{}'.format(idx))  # [B, EMB]
                mlp = tf.nn.dropout(mlp, self.keep_prob, name='dp_fc_{}'.format(idx))

            fusion = tf.concat([gmf, mlp], axis=-1)
            fc_out = tf.layers.dense(fusion, 1, activation=None, name='fc_out')
            fc_out = tf.nn.dropout(fc_out, self.keep_prob, name='dp6')

            ret = tf.reshape(fc_out, [-1, ])
        return ret

    def build_main_graph(self):
        # Embedding Layer
        with tf.variable_scope('Embedding2', reuse=tf.AUTO_REUSE):
            # Entry 0 is zero embedding that corresponds to empty position in a sequence.
            self.emb_mtx2 = tf.concat(
                [tf.zeros(shape=[1, self.emb_dim]), tf.get_variable('emb_mtx2', [self.feature_size - 1, self.emb_dim])],
                axis=0)
            self.raw_seq_emb2 = tf.nn.embedding_lookup(self.emb_mtx2, self.seq_id)  # [B, T,  EMB]
            self.user_emb2 = tf.nn.embedding_lookup(self.emb_mtx2, self.user_id)  # [B, EMB]
            self.test_emb2 = tf.nn.embedding_lookup(self.emb_mtx2, self.test_id)  # [B, EMB]

            self.pos_emb2 = tf.nn.embedding_lookup(self.emb_mtx2, self.pos_id)  # [B, T, EMB]
            self.neg_emb2 = tf.nn.embedding_lookup(self.emb_mtx2, self.neg_id)  # [B, T, EMB]

            self.last_pos_item2 = tf.reduce_sum(tf.expand_dims((self.seq_mask - self.seq_mask_1), -1) * self.pos_emb2,
                                                axis=1)  # [B, EMB]
            self.last_neg_item2 = tf.reduce_sum(tf.expand_dims((self.seq_mask - self.seq_mask_1), -1) * self.neg_emb2,
                                                axis=1)  # [B, EMB]
            self.last_item2 = tf.reduce_sum(tf.expand_dims((self.seq_mask - self.seq_mask_1), -1) * self.raw_seq_emb2,
                                            axis=1)  # [B, EMB]

        with tf.variable_scope('logits', reuse=tf.AUTO_REUSE):
            self.pos_logits = self.pointwise_pred(self.last_pos_item, self.last_pos_item2)
            self.neg_logits = self.pointwise_pred(self.last_neg_item, self.last_neg_item2)
            self.test_logits = self.pointwise_pred(self.test_emb, self.test_emb2)

            self.y_pred = tf.reshape(tf.nn.sigmoid(self.test_logits), [-1, ])

        with tf.variable_scope('loss'):
            if self.pairwise_loss:
                logits_diff = self.pos_logits - self.neg_logits
                self.log_loss = self.positive_part_loss(logits_diff)
            else:
                faster = True
                if faster:
                    self.log_loss = self.positive_part_loss(self.pos_logits) + self.negative_part_loss(self.neg_logits)
                else:
                    concat_label = tf.concat([tf.ones_like(self.pos_logits), tf.zeros_like(self.neg_logits)], axis=-1)
                    concat_logits = tf.concat([self.pos_logits, self.neg_logits], axis=-1)
                    self.log_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=concat_label, logits=concat_logits)

            self.loss = tf.reduce_sum(self.log_loss)


class SASRec(ModelBase):
    def __init__(self, feature_size, emb_dim, hidden_size, max_seq_len, pairwise_loss=False, num_blocks=2, num_heads=1):
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        super(SASRec, self).__init__(feature_size, emb_dim, hidden_size, max_seq_len, pairwise_loss)

    def build_main_graph(self):
        # RNN layer
        with tf.variable_scope('model'):
            self.seq_emb = self.raw_seq_emb + self.positional_emb
            self.seq_emb = tf.nn.dropout(self.seq_emb, self.keep_prob, name='seq_emb_inp_dropout')
            self.seq_emb *= tf.expand_dims(self.seq_mask, -1)
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):
                    # Self-attention
                    self.seq_emb = multihead_attention(queries=normalize(self.seq_emb, scope='inp_emb_norm'),
                                                       keys=self.seq_emb,
                                                       num_units=self.hidden_size,
                                                       num_heads=self.num_heads,
                                                       dropout_rate=self.drop_prob,
                                                       is_training=self.training,
                                                       causality=True,
                                                       scope="self_attention")

                    # Feed forward
                    self.seq_emb = feedforward(normalize(self.seq_emb, scope='feedfw_inp_norm'),
                                               num_units=[self.hidden_size, self.hidden_size],
                                               dropout_rate=self.drop_prob, is_training=self.training)
                    self.seq_emb *= tf.expand_dims(self.seq_mask, -1)

            self.seq_emb = normalize(self.seq_emb)
            self.last_step_emb = extract_last_step(self.seq_emb, self.seq_len - 1)  # [B, EMB]

        with tf.variable_scope('logits'):
            self.pos_logits = tf.reduce_sum(self.seq_emb * self.pos_emb, axis=-1)  # [B, T]
            self.pos_logits = tf.boolean_mask(self.pos_logits, self.seq_mask_bool)
            self.neg_logits = tf.reduce_sum(self.seq_emb * self.neg_emb, axis=-1)  # [B, T]
            self.neg_logits = tf.boolean_mask(self.neg_logits, self.seq_mask_bool)
            self.test_logits = tf.reduce_sum(self.test_emb * self.last_step_emb, axis=-1)  # [B,]
            self.y_pred = tf.reshape(tf.nn.sigmoid(self.test_logits), [-1, ])

        with tf.variable_scope('loss'):
            if self.pairwise_loss:
                logits_diff = self.pos_logits - self.neg_logits
                self.log_loss = self.positive_part_loss(logits_diff)
            else:
                faster = True
                if faster:
                    self.log_loss = self.positive_part_loss(self.pos_logits) + self.negative_part_loss(self.neg_logits)
                else:
                    concat_label = tf.concat([tf.ones_like(self.pos_logits), tf.zeros_like(self.neg_logits)], axis=-1)
                    concat_logits = tf.concat([self.pos_logits, self.neg_logits], axis=-1)
                    self.log_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=concat_label, logits=concat_logits)

            self.loss = tf.reduce_sum(self.log_loss)


class ModelMRIF(object):
    def __init__(self, feature_size, emb_dim, hidden_size, max_seq_len, pairwise_loss=False,
                 mrif_parallel=False, mrif_kernels=None, num_blocks=2, num_heads=1, pretrain_epoch=10,
                 mrif_alpha=1.0):
        # reset graph
        tf.reset_default_graph()
        self.hidden_size = hidden_size
        self.emb_dim = emb_dim
        self.max_seq_len = max_seq_len
        self.feature_size = feature_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.mrif_parallel = mrif_parallel
        self.mrif_kernels = mrif_kernels
        self.pretrain_epoch = pretrain_epoch
        self.mrif_alpha = mrif_alpha
        if mrif_kernels is None:
            mrif_kernels = [2, 2]

        # Input placeholders
        with tf.name_scope('Inputs'):
            self.seq_id = tf.placeholder(tf.int32, [None, max_seq_len], name='seq_id')
            self.seq_len = tf.placeholder(tf.int32, [None, ], name='seq_len')
            self.pos_id = tf.placeholder(tf.int32, [None, max_seq_len], name='pos_id')
            self.neg_id = tf.placeholder(tf.int32, [None, max_seq_len], name='neg_id')
            self.test_id = tf.placeholder(tf.int32, [None, ], name='test_id')
            self.user_id = tf.placeholder(tf.int32, [None, ], name='user_id')

            # 1-dropout
            self.keep_prob = tf.placeholder(tf.float32)
            self.drop_prob = 1 - self.keep_prob

            # learning rate
            self.lr = tf.placeholder(tf.float64, [])

            # reg lambda
            self.reg_lambda = tf.placeholder(tf.float32, [])

            # If is training
            self.training = tf.placeholder(tf.bool, [])

            # current epoch
            self.epoch = tf.placeholder(tf.int32, [])

        with tf.variable_scope('Mask', reuse=tf.AUTO_REUSE):
            # Item and User Mask
            self.seq_mask = tf.sequence_mask(self.seq_len, tf.shape(self.seq_id)[1], dtype=tf.float32)  # [B, T]
            self.seq_mask_1 = tf.sequence_mask(self.seq_len - 1, tf.shape(self.seq_id)[1], dtype=tf.float32)  # [B, T]
            self.seq_mask_bool = tf.sequence_mask(self.seq_len, tf.shape(self.seq_id)[1], dtype=tf.bool)  # [B, T]

        # Embedding Layer
        with tf.variable_scope('Embedding', reuse=tf.AUTO_REUSE):
            # Entry 0 is zero embedding that corresponds to empty position in a sequence.
            self.emb_mtx = tf.concat(
                [tf.zeros(shape=[1, emb_dim]), tf.get_variable('emb_mtx', [feature_size - 1, emb_dim])], axis=0)
            self.raw_seq_emb = tf.nn.embedding_lookup(self.emb_mtx, self.seq_id)  # [B, T,  EMB]
            self.user_emb = tf.nn.embedding_lookup(self.emb_mtx, self.user_id)  # [B, EMB]
            self.test_emb = tf.nn.embedding_lookup(self.emb_mtx, self.test_id)  # [B, EMB]

            self.positional_emb_mtx = tf.get_variable('positional_emb_mtx', [max_seq_len, emb_dim])
            self.positional_emb = tf.nn.embedding_lookup(self.positional_emb_mtx,
                                                         tf.tile(tf.expand_dims(tf.range(self.max_seq_len), 0),
                                                                 [tf.shape(self.raw_seq_emb)[0], 1]))

            self.pos_emb = tf.nn.embedding_lookup(self.emb_mtx, self.pos_id)  # [B, T, EMB]
            self.neg_emb = tf.nn.embedding_lookup(self.emb_mtx, self.neg_id)  # [B, T, EMB]

            self.last_item = tf.reduce_sum(tf.expand_dims((self.seq_mask - self.seq_mask_1), -1) * self.raw_seq_emb,
                                           axis=1)  # [B, EMB]
            self.last_pos_item = tf.reduce_sum(tf.expand_dims((self.seq_mask - self.seq_mask_1), -1) * self.pos_emb,
                                               axis=1)  # [B, EMB]
            self.last_neg_item = tf.reduce_sum(tf.expand_dims((self.seq_mask - self.seq_mask_1), -1) * self.neg_emb,
                                               axis=1)  # [B, EMB]

            self.seq_emb = self.pointwise_pred()
            self.last_step_emb = extract_last_step(self.seq_emb, self.seq_len - 1)  # [B, EMB]

        with tf.variable_scope('logits', reuse=tf.AUTO_REUSE):
            self.pos_logits = tf.reduce_sum(self.seq_emb * self.pos_emb, axis=-1)  # [B, T]
            self.pos_logits = tf.boolean_mask(self.pos_logits, self.seq_mask_bool)
            self.neg_logits = tf.reduce_sum(self.seq_emb * self.neg_emb, axis=-1)  # [B, T]
            self.neg_logits = tf.boolean_mask(self.neg_logits, self.seq_mask_bool)

            self.final_step_pos_logits = self.pointwise_pred_last(self.last_pos_item)
            self.final_step_neg_logits = self.pointwise_pred_last(self.last_neg_item)

            self.test_logits = self.pointwise_pred_last(self.test_emb)
            self.y_pred = tf.reshape(tf.nn.sigmoid(self.test_logits), [-1, ])

            self.log_loss = tf.reduce_sum(
                self.positive_part_loss(self.pos_logits) + self.negative_part_loss(self.neg_logits))
            self.log_loss_final_step = tf.reduce_sum(
                self.positive_part_loss(self.final_step_pos_logits) + self.negative_part_loss(
                    self.final_step_neg_logits))

        alpha = tf.cond(
            self.epoch < self.pretrain_epoch,
            true_fn=lambda: tf.constant(0.0),
            false_fn=lambda: tf.constant(self.mrif_alpha),
            name='alpha'
        )
        self.loss = self.log_loss * (1 - alpha) + self.log_loss_final_step * alpha

        for v in tf.trainable_variables():
            print(v)
            if 'bias' not in v.name and 'emb_mtx' not in v.name:
                self.loss += self.reg_lambda * tf.nn.l2_loss(v)
        # optimizer and training step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

    def positive_part_loss(self, logits):
        """ -log(sigmoid(x)) = max(x, 0) - x  + log(1 + exp(-abs(x))) """
        zeros = tf.zeros_like(logits, dtype=logits.dtype)
        cond = (logits >= zeros)
        relu_logits = tf.where(cond, logits, zeros)
        neg_abs_logits = tf.where(cond, -logits, logits)
        return relu_logits - logits + tf.log1p(tf.exp(neg_abs_logits))

    def negative_part_loss(self, logits):
        """ -log(1-sigmoid(x)) = max(x, 0)   + log(1 + exp(-abs(x))) """
        zeros = tf.zeros_like(logits, dtype=logits.dtype)
        cond = (logits >= zeros)
        relu_logits = tf.where(cond, logits, zeros)
        neg_abs_logits = tf.where(cond, -logits, logits)
        return relu_logits + tf.log1p(tf.exp(neg_abs_logits))

    def build_fc_net(self, inp, output_dim=1):
        # inp = tf.layers.batch_normalization(inputs=inp, name='bn1')
        fc1 = tf.layers.dense(inp, 200, activation=tf.nn.tanh, name='fc1')
        dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
        fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.tanh, name='fc2')
        dp2 = tf.nn.dropout(fc2, self.keep_prob, name='dp2')
        fc3 = tf.layers.dense(dp2, output_dim, activation=None, name='fc3')
        # output
        if output_dim == 1:
            return tf.reshape(fc3, [-1, ])
        else:
            return fc3

    def train(self, sess, batch_data, lr, reg_lambda, keep_prob, epoch):
        loss, _ = sess.run([self.loss, self.train_step], feed_dict={
            self.user_id: batch_data[0],
            self.seq_id: batch_data[1],
            self.seq_len: batch_data[2],
            self.pos_id: batch_data[3],
            self.neg_id: batch_data[4],
            self.lr: lr,
            self.reg_lambda: reg_lambda,
            self.keep_prob: keep_prob,
            self.training: True,
            self.epoch: epoch,
        })
        return loss

    def eval(self, sess, batch_data):
        # pred, attn_score, attn_emb, attn_emb_seq = sess.run(
        #     [self.y_pred, self.attn_scores, self.attn_user_embs, self.attn_emb_seq], feed_dict={
        #         self.user_id: batch_data[0],
        #         self.seq_id: batch_data[1],
        #         self.seq_len: batch_data[2],
        #         self.test_id: batch_data[3],
        #         self.keep_prob: 1.0,
        #         self.training: False,
        #         self.epoch: 1000,
        #     })
        # return pred.reshape([-1, ]).tolist(), attn_score, attn_emb, attn_emb_seq
        pred = sess.run([self.y_pred], feed_dict={
            self.user_id: batch_data[0],
            self.seq_id: batch_data[1],
            self.seq_len: batch_data[2],
            self.test_id: batch_data[3],
            self.keep_prob: 1.0,
            self.training: False,
            self.epoch: 1000,
        })
        return pred[0].reshape([-1, ]).tolist()

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from {}'.format(path))

    def prelu(self, _x, scope=None):
        """parametric ReLU activation"""
        with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
            _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                     dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)

    def attention(self, key, value, query, mask):
        """Vanilla attention.

        Attention score is softmax on output of feed forward network with `key` and `query` 
        as input.
        Output is weighted sum of `value` according to attention score.

        Args:
            key: A tensor to compute attention score with query with shape [B, T, EMB].
            value: A tensor on which weighted sum is computed. [B, T, EMB]
            query: A tensor to compute attention score with key with shape [B, EMB].
            mask: A tensor where non-empty positions in sequence is 1, otherwise is 0. [B, T]
        """
        _, max_len, k_dim = key.get_shape().as_list()
        query = tf.layers.dense(query, k_dim, activation=None)
        query = self.prelu(query)
        queries = tf.tile(tf.expand_dims(query, 1), [1, max_len, 1])  # [B, T, Dk]
        inp = tf.concat([queries, key, queries - key, queries * key], axis=-1)
        fc1 = tf.layers.dense(inp, 80, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 40, activation=tf.nn.relu)
        fc3 = tf.layers.dense(fc2, 1, activation=None)  # [B, T, 1]

        mask = tf.reshape(mask, [-1, max_len, 1])
        mask = tf.equal(mask, tf.ones_like(mask))  # [B, T, 1]
        paddings = tf.ones_like(fc3) * (-2 ** 32 + 1)
        score = tf.nn.softmax(tf.reshape(tf.where(mask, fc3, paddings), [-1, max_len]))  # [B, T]

        atten_output = tf.multiply(value, tf.expand_dims(score, 2))
        atten_output_sum = tf.reduce_sum(atten_output, axis=1)

        return atten_output_sum, atten_output, score

    def attention_v2(self, key, value, query, mask):
        """Vanilla attention.

        Attention score is softmax on inner product of `key` and `query`.
        Output is weighted sum of `value` according to attention score.

        Args:
            key: A tensor to compute attention score with query with shape [B, T, EMB].
            value: A tensor on which weighted sum is computed. [B, T, EMB]
            query: A tensor to compute attention score with key with shape [B, EMB].
            mask: A tensor where non-empty positions in sequence is 1, otherwise is 0. [B, T]
        """
        _, max_len, k_dim = key.get_shape().as_list()
        # query = tf.layers.dense(query, k_dim, activation=None) #[B, EMB]
        queries = tf.expand_dims(query, 1)  # [B, 1, EMB]
        product = tf.reduce_sum((queries * key), axis=-1)  # [B, T]

        mask = tf.reshape(tf.equal(mask, tf.ones_like(mask)), [-1, max_len])  # [B, T]
        paddings = tf.ones_like(product) * (-2 ** 32 + 1)
        score = tf.nn.softmax(tf.where(mask, product, paddings))  # [B, T]

        atten_output = tf.multiply(value, tf.expand_dims(score, 2))  # [B, T, Dv==Dk]
        atten_output_sum = tf.reduce_sum(atten_output, axis=1)  # [B, Dv==Dk]

        return atten_output_sum, atten_output, score

    def pointwise_pred(self):
        self.seq_emb = self.raw_seq_emb + self.positional_emb
        self.seq_emb = tf.nn.dropout(self.seq_emb, self.keep_prob, name='seq_emb_inp_dropout')
        self.seq_emb *= tf.expand_dims(self.seq_mask, -1)
        for i in range(self.num_blocks):
            with tf.variable_scope("num_blocks_%d" % i):
                # Self-attention
                self.seq_emb = multihead_attention(queries=normalize(self.seq_emb, scope='inp_emb_norm'),
                                                   keys=self.seq_emb,
                                                   num_units=self.hidden_size,
                                                   num_heads=self.num_heads,
                                                   dropout_rate=self.drop_prob,
                                                   is_training=self.training,
                                                   causality=True,
                                                   scope="self_attention")

                # Feed forward
                self.seq_emb = feedforward(normalize(self.seq_emb, scope='feedfw_inp_norm'),
                                           num_units=[self.hidden_size, self.hidden_size],
                                           dropout_rate=self.drop_prob, is_training=self.training)
                self.seq_emb *= tf.expand_dims(self.seq_mask, -1)

        self.seq_emb = normalize(self.seq_emb)
        return self.seq_emb

    def pointwise_pred_last(self, target_emb):
        # res = tf.reduce_sum(self.last_step_emb*target_emb,axis=-1)
        with tf.variable_scope('agg', reuse=tf.AUTO_REUSE):
            exts = [self.seq_emb]
            masks = [self.seq_mask]
            if self.mrif_parallel:
                kernels = self.mrif_kernels
                self.seq_emb = self.seq_emb * tf.expand_dims(self.seq_mask, -1)
                for idx, kernel_sz in enumerate(kernels):
                    seq, mask = self.aggregator(self.seq_emb, self.seq_mask, kernel_sz, name='agg_' + str(idx))
                    exts.append(seq)
            else:
                kernels = self.mrif_kernels
                for idx, kernel_sz in enumerate(kernels):
                    seq, mask = self.aggregator(exts[-1] * tf.expand_dims(masks[-1], -1), masks[-1], kernel_sz,
                                                name='agg_' + str(idx))
                    exts.append(seq)
                    masks.append(mask)

        self.attn_scores = []
        self.attn_user_embs = [self.last_step_emb]
        self.attn_emb_seq = exts
        with tf.variable_scope('attention', reuse=tf.AUTO_REUSE):
            user_embs = [self.last_step_emb]
            user_embs_sum = self.last_step_emb
            for seq in exts:
                item_part, _, attn_score = self.attention_v2(seq, seq, target_emb, self.seq_mask)
                self.attn_scores.append(attn_score)
                user_embs.append(item_part)
                self.attn_user_embs.append(item_part)
                user_embs_sum = user_embs_sum + item_part
            res = tf.reduce_sum(user_embs_sum * target_emb, axis=-1)
        return res


def mean_aggregator(inp, mask, kernel_sz=3, name='mean_aggregator'):
    """Mean aggregator for behavior sequence
    
    Args:
        inp: Input sequence with shape [batch, seq_len, embedding_size]
    
    Returns:
        A Tensor with shape [batch, seq_len, embedding_size]
    """
    with tf.name_scope(name):
        _, seq_len, emb_sz = inp.get_shape().as_list()
        inp = tf.expand_dims(inp, -1)
        kernel = tf.ones(shape=[kernel_sz, 1, 1, 1]) / kernel_sz
        res = tf.nn.conv2d(inp, kernel, strides=[1, 1, 1, 1], padding='SAME')  # [B, T, EMB, 1]
        res = tf.reshape(res, [-1, seq_len, emb_sz])

        # normalization
        norm = tf.reshape(mask, [-1, seq_len, 1, 1, ])  # [B, T, 1, 1]
        norm = tf.nn.conv2d(norm, kernel, strides=[1, 1, 1, 1], padding='SAME')  # [B, T, 1, 1]
        norm = tf.reshape(norm, [-1, seq_len, 1])

        new_mask = tf.reshape(norm, [-1, seq_len])  # [B, T]

        norm = norm + 1e-9
        res = res / norm

        # new mask

        new_mask = tf.stop_gradient(tf.cast(tf.not_equal(new_mask, tf.zeros_like(new_mask)), dtype=tf.float32))

        return res, new_mask


class MRIF_avg(ModelMRIF):
    def __init__(self, feature_size, emb_dim, hidden_size, max_seq_len, pairwise_loss=False,
                 mrif_parallel=False, mrif_kernels=None, num_blocks=2, num_heads=1, pretrain_epoch=10,
                 mrif_alpha=1.0):
        self.aggregator = mean_aggregator
        super(MRIF_avg, self).__init__(feature_size, emb_dim, hidden_size, max_seq_len, pairwise_loss,
                                       mrif_parallel, mrif_kernels, num_blocks, num_heads, pretrain_epoch, mrif_alpha)


def max_aggregator_byindex(inp, mask, kernel_sz=3, name='max_aggregator'):
    """Max aggregator for behavior sequence
    
    Args:
        inp: Input sequence with shape [batch, seq_len, embedding_size]
    
    Returns:
        A Tensor with shape [batch, seq_len, embedding_size]
    """
    with tf.name_scope(name):
        _, seq_len, emb_sz = inp.get_shape().as_list()
        inp = tf.expand_dims(inp, -1)

        norm = tf.norm(inp, axis=2, keepdims=True)

        _, argmax = tf.nn.max_pool_with_argmax(norm, [1, kernel_sz, 1, 1], strides=[1, 1, 1, 1], padding='SAME')

        argmax = tf.reshape(argmax, [-1, seq_len])  # each element is the index after flattening

        inp = tf.reshape(inp, [-1, emb_sz])
        res = tf.gather(inp, argmax)
        res = tf.reshape(res, [-1, seq_len, emb_sz])

        new_mask = mask
        return res, new_mask


def max_aggregator_byval(inp, mask, kernel_sz=3, name='max_aggregator'):
    """Max aggregator for behavior sequence
    
    Args:
        inp: Input sequence with shape [batch, seq_len, embedding_size]
    
    Returns:
        A Tensor with shape [batch, seq_len, embedding_size]
    """
    with tf.name_scope(name):
        _, seq_len, emb_sz = inp.get_shape().as_list()
        inp = tf.expand_dims(inp, -1)

        res = tf.layers.max_pooling2d(inp, pool_size=[kernel_sz, 1], strides=[1, 1], padding='SAME')
        res = tf.reshape(res, [-1, seq_len, emb_sz])
        new_mask = mask
        return res, new_mask


class MRIF_max(ModelMRIF):
    def __init__(self, feature_size, emb_dim, hidden_size, max_seq_len, pairwise_loss=False,
                 mrif_parallel=False, mrif_kernels=None, num_blocks=2, num_heads=1, pretrain_epoch=10,
                 mrif_max_type='ind', mrif_alpha=1.0):
        self.mrif_max_type = mrif_max_type
        if self.mrif_max_type == 'ind':
            self.aggregator = max_aggregator_byindex
        else:
            self.aggregator = max_aggregator_byval
        super(MRIF_max, self).__init__(feature_size, emb_dim, hidden_size, max_seq_len, pairwise_loss,
                                       mrif_parallel, mrif_kernels, num_blocks, num_heads, pretrain_epoch, mrif_alpha)


def attn_aggregator_allrows(inp, mask, kernel_sz=3, name='attn_aggregator'):
    """Attentional aggregator for behavior sequence
    
    Args:
        inp: Input sequence with shape [batch, seq_len, embedding_size]
    
    Returns:
        A Tensor with shape [batch, seq_len, embedding_size]
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        _, seq_len, emb_sz = inp.get_shape().as_list()
        inp = tf.expand_dims(inp, -1)
        kernel = tf.get_variable(name='kernel', shape=[kernel_sz, 1, 1, 1])
        res = tf.nn.conv2d(inp, kernel, strides=[1, 1, 1, 1], padding='SAME')
        res = tf.reshape(res, [-1, seq_len, emb_sz])

        # normalization
        norm = tf.reshape(mask, [-1, seq_len, 1, 1, ])  # [B, T, 1, 1]
        kernel_ones = tf.ones(shape=[kernel_sz, 1, 1, 1]) / kernel_sz
        norm = tf.nn.conv2d(norm, kernel_ones, strides=[1, 1, 1, 1], padding='SAME')  # [B, T, 1, 1]
        norm = tf.reshape(norm, [-1, seq_len, 1])

        new_mask = tf.reshape(norm, [-1, seq_len])  # [B, T]

        norm = norm + 1e-9
        res = res / norm

        # new mask

        new_mask = tf.stop_gradient(tf.cast(tf.not_equal(new_mask, tf.zeros_like(new_mask)), dtype=tf.float32))

        return res, new_mask


def attn_aggregator_rowbyrow(inp, mask, kernel_sz=3, name='attn_aggregator'):
    """Attentional aggregator for behavior sequence
    
    Args:
        inp: Input sequence with shape [batch, seq_len, embedding_size]
    
    Returns:
        A Tensor with shape [batch, seq_len, embedding_size]
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        _, seq_len, emb_sz = inp.get_shape().as_list()
        inp = tf.expand_dims(inp, -1)

        slices = []
        for i in range(emb_sz):
            kernel = tf.get_variable(name='kernel_{}'.format(i), shape=[kernel_sz, 1, 1, 1])
            res = tf.nn.conv2d(inp[:, :, i:i + 1, :], kernel, strides=[1, 1, 1, 1], padding='SAME')
            res = tf.reshape(res, [-1, seq_len, 1])
            slices.append(res)
        res = tf.concat(slices, axis=-1)

        # normalization
        norm = tf.reshape(mask, [-1, seq_len, 1, 1, ])  # [B, T, 1, 1]
        kernel_ones = tf.ones(shape=[kernel_sz, 1, 1, 1]) / kernel_sz
        norm = tf.nn.conv2d(norm, kernel_ones, strides=[1, 1, 1, 1], padding='SAME')  # [B, T, 1, 1]
        norm = tf.reshape(norm, [-1, seq_len, 1])

        new_mask = tf.reshape(norm, [-1, seq_len])  # [B, T]

        norm = norm + 1e-9
        res = res / norm

        # new mask

        new_mask = tf.stop_gradient(tf.cast(tf.not_equal(new_mask, tf.zeros_like(new_mask)), dtype=tf.float32))

        return res, new_mask


def attn_aggregator_multifilter(inp, mask, kernel_sz=3, name='attn_aggregator'):
    """Attentional aggregator for behavior sequence
    
    Args:
        inp: Input sequence with shape [batch, seq_len, embedding_size]
    
    Returns:
        A Tensor with shape [batch, seq_len, embedding_size]
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        _, seq_len, emb_sz = inp.get_shape().as_list()
        inp = tf.expand_dims(inp, -1)
        kernel = tf.get_variable(name='kernel', shape=[kernel_sz, emb_sz, 1, emb_sz])
        res = tf.nn.conv2d(inp, kernel, strides=[1, 1, 1, 1],
                           padding='SAME')  # since padding is same, the output has shape [B,T,EMB,EMB]
        valid_index = int(np.floor((emb_sz + 1.0) / 2) - 1)
        res = res[:, :, valid_index, :]
        res = tf.reshape(res, [-1, seq_len, emb_sz])

        # normalization
        norm = tf.reshape(mask, [-1, seq_len, 1, 1, ])  # [B, T, 1, 1]
        kernel_ones = tf.ones(shape=[kernel_sz, 1, 1, 1]) / kernel_sz
        norm = tf.nn.conv2d(norm, kernel_ones, strides=[1, 1, 1, 1], padding='SAME')  # [B, T, 1, 1]
        norm = tf.reshape(norm, [-1, seq_len, 1])

        new_mask = tf.reshape(norm, [-1, seq_len])  # [B, T]

        norm = norm + 1e-9
        res = res / norm

        # new mask
        new_mask = tf.stop_gradient(tf.cast(tf.not_equal(new_mask, tf.zeros_like(new_mask)), dtype=tf.float32))

        return res, new_mask


class MRIF_attn(ModelMRIF):
    def __init__(self, feature_size, emb_dim, hidden_size, max_seq_len, pairwise_loss=False,
                 mrif_parallel=False, mrif_kernels=None, num_blocks=2, num_heads=1, pretrain_epoch=10,
                 mrif_attn_type='allrows', mrif_alpha=1.0):
        self.mrif_attn_type = mrif_attn_type
        if self.mrif_attn_type == 'allrows':
            self.aggregator = attn_aggregator_allrows
        elif self.mrif_attn_type == 'rowbyrow':
            self.aggregator = attn_aggregator_rowbyrow
        else:
            self.aggregator = attn_aggregator_multifilter
        super(MRIF_attn, self).__init__(feature_size, emb_dim, hidden_size, max_seq_len, pairwise_loss,
                                        mrif_parallel, mrif_kernels, num_blocks, num_heads, pretrain_epoch, mrif_alpha)
