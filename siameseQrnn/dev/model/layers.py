# -- coding:utf-8 --
'''
这个文件定义和tensorflow相关的一些全局变量，比如特殊的初始化方式，正则化项，GraphKeys的命名等
'''
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers.utils import smart_cond

# global DEFINE
tf.GraphKeys.NO_CLIP = 'tensorflow_noclip_params'

# basic settings
regular_function = tf.contrib.layers.sum_regularizer(
                   [tf.contrib.layers.l2_regularizer(1.0),
                   tf.contrib.layers.l1_regularizer(0.5)])
standard_initializer = tf.random_normal_initializer(0.0, 1e-2)

def word_lookup(inp, E, keep_prob=0.8, is_training=True, scope='WordLookup'):
    '''
    :param: tf.int64 tensor [batch_size, sent_length], word embedding tensor [vocab_size, dims]
    '''
    with tf.variable_scope(scope, reuse=None):
        sents_ebd = tf.nn.embedding_lookup(E, inp, partition_strategy='div')  # for partion
        sents_ebd = tf.contrib.layers.dropout(sents_ebd,
                    keep_prob=keep_prob, is_training=is_training, scope='Dropout')
    return sents_ebd

def DenseLayer(inp, size, fn, keep_prob=0.8, is_training=False, reuse=False, scope='DenseLayer'):
    out = tf.contrib.layers.fully_connected(inp, size, fn, reuse=reuse, scope=scope)
    return tf.contrib.layers.dropout(out, keep_prob=keep_prob, is_training=is_training)

def cos_distance(x, y, normalize=True):
    '''
    :param x, y: (bs, n_dims)
    '''
    dot = tf.reduce_sum(tf.multiply(x, y), axis=1)  # (bs)
    norm_x = tf.norm(x, ord=2, axis=1)  # (bs)
    norm_y = tf.norm(y, ord=2, axis=1)
    cos_dist = tf.divide(dot, tf.multiply(norm_x, norm_y) + 1e-8)
    if normalize:
        cos_dist = (cos_dist + 1.0) / 2.0
    return cos_dist

def logistic_regression(inconming, scope='regression', reuse=None):
    '''
    logistic regression layer with normalization.
    :param inconming: (bs, n) matrix, ndims==2
    '''
    with tf.variable_scope(scope, reuse=reuse,
         initializer=tf.random_normal_initializer(0.0, 1e-2),
         regularizer=tf.contrib.layers.sum_regularizer(
         [tf.contrib.layers.l2_regularizer(1.0), tf.contrib.layers.l1_regularizer(0.5)])):
        return tf.contrib.layers.fully_connected(inconming, 1, tf.nn.sigmoid, scope='fc')

def BiLinearFeat(q_vec, t_vec, dm, reuse=None, scope='LogLinear'):
    '''
    ref: NASM
    q_vec, t_vec have shape(bs, ns, vec_dim)
    '''
    feat_concat = tf.concat([q_vec, t_vec], 2)

    q_vec_dim = q_vec.get_shape()[-1].value
    t_vec_dim = t_vec.get_shape()[-1].value  # static value, if None, raise error.
    batch_size = tf.shape(t_vec)[0]  # batch size is dynamic.

    with tf.variable_scope(scope, reuse=reuse,
         initializer=tf.random_normal_initializer(0.0, 1e-2),
         regularizer=tf.contrib.layers.sum_regularizer(
         [tf.contrib.layers.l2_regularizer(1.0), tf.contrib.layers.l1_regularizer(0.5)])):

        U = tf.get_variable('U', shape=[q_vec_dim, t_vec_dim])
        V = tf.get_variable('V', shape=[q_vec_dim + t_vec_dim])
        b = tf.get_variable('b', shape=[])

        r = b + tf.reshape(
                tf.matmul(tf.reshape(feat_concat, [-1, q_vec_dim + t_vec_dim]), V[:, None]),
                [batch_size, -1])
        # (bs*ns, 2rnn)(2rnn, 1) --> (bs, ns, 1) --> (bs, ns)
        mid = tf.reshape(tf.matmul(tf.reshape(q_vec, [-1, q_vec_dim]), U), [batch_size, -1, t_vec_dim])
        r += tf.reduce_sum(mid * t_vec, 2)  # mul+sum better than batch_matmul
        # (bs, ns, rnn) dot (rnn, rnn) --> (bs, ns, rnn); (bs, None, rnn) * (bs, ns, rnn) --> (bs. ns)
        max_r = tf.reduce_max(r, 1)
        mean_r = tf.reduce_mean(r, 1)
        # max_r = tf.reduce_max(r[:, :dm, :], 1)
        # mean_r = tf.reduce_mean(r[:, :dm, :], 1)
        # r = tf.reduce_max(tf.reduce_max(q_vec, 2),1)
    return tf.nn.sigmoid(max_r+mean_r, name='Score')  # out of the variable scope
    #




def MatchingMethodsCombine(q_vec, t_vec, is_training, dm, extra_score, reuse=None, scope='MatchingMethodsCombine'):
    '''
    ref: Supervised Learning of Universal Sentence Representations from Natural Language Inference Data
    q_vec, t_vec have shape(bs, ns, vec_dim)
    '''

    q_vec_dim = q_vec.get_shape()[-1].value
    t_vec_dim = t_vec.get_shape()[-1].value  # static value, if None, raise error.
    batch_size = tf.shape(t_vec)[0]  # batch size is dynamic.
    extra_score_dim = extra_score.get_shape()[-1].value
    n_samples = 1


    q_vec = tf.contrib.layers.fully_connected(q_vec, q_vec_dim, tf.nn.tanh, scope='query_to_att')
    q_vec = tf.contrib.layers.dropout(q_vec, keep_prob=0.6, is_training=is_training, scope='dqa')


    feat_concat = tf.concat([q_vec, t_vec], 2)
    diff = q_vec - t_vec
    feat_concat = tf.concat([feat_concat, diff], 2)
    dot = q_vec * t_vec
    feat_concat = tf.concat([feat_concat, dot], 2)
    feat_concat = tf.concat([feat_concat, extra_score], 2)


    with tf.variable_scope(scope, reuse=reuse,
         initializer=tf.random_normal_initializer(0.0, 1e-2),
         regularizer=tf.contrib.layers.sum_regularizer(
         [tf.contrib.layers.l2_regularizer(1.0), tf.contrib.layers.l1_regularizer(0.5)])):

        # U = tf.get_variable('U', shape=[q_vec_dim, t_vec_dim])
        V = tf.get_variable('V', shape=[2*(q_vec_dim + t_vec_dim)+extra_score_dim])
        b = tf.get_variable('b', shape=[])
        # W_score = tf.get_variable('W_score', shape=[3])
        # b_score = tf.get_variable('b_score', shape=[])

        r = b + tf.reshape(
                tf.matmul(tf.reshape(feat_concat, [-1, 2*(q_vec_dim + t_vec_dim)+extra_score_dim]), V[:, None]),
                [batch_size, -1])
        # (bs*ns, 2rnn)(2rnn, 1) --> (bs, ns, 1) --> (bs, ns)
        # mid = tf.reshape(tf.matmul(tf.reshape(q_vec, [-1, q_vec_dim]), U), [batch_size, -1, t_vec_dim])
        # r += tf.reduce_sum(mid * t_vec, 2)  # mul+sum better than batch_matmul
        # (bs, ns, rnn) dot (rnn, rnn) --> (bs, ns, rnn); (bs, None, rnn) * (bs, ns, rnn) --> (bs. ns)
        # r = tf.reshape(r, [-1, tf.shape(r)[-1], 1])
        # r = tf.nn.sigmoid(r)
        # r = tf.concat([r, extra_score], 2)
        # r = b_score + tf.matmul(tf.reshape(r, [-1, 3*n_samples]), W_score[:, None])
        # r = tf.reshape(r, [batch_size, -1])
        max_r = tf.reduce_max(r, 1)
        mean_r = tf.reduce_mean(r, 1)
    return tf.nn.sigmoid(max_r+mean_r, name='Score') # out of the variable scope
    #

def sent_encoder_multi(incomings, settings, keep_prob=0.75,
                       reuse=None, scope='Sent_Encoder'):
    '''
    Multi-layer LSTM sentence encoder. 16.11.1 Haoze Sun
    :param incomings=[sents, sent_length], sents is (batch_size, max_sent_length, embedding_dims) tensor
                                           sent_length is (batch_size,) tensor scalar in [0,1,2,3,...]
           use dynamic_rnn for variable length
    :param num_units: LSTM unit numbers, int
    :param keep_prob: placeholder for dropout, control training or test
    :param scope: variable_scope
    :return: states of each words (batch_size, sent_length, rnn); final states of last layer (bs, rnn)
    '''
    sents, sent_length = incomings
    rnn_units, num_layers = settings
    with tf.variable_scope(scope, reuse=reuse):  # no regularization here
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_units, state_is_tuple=True)
        # dropout in every step is slow, default=1.0 || norm_gain smaller? || unstable in dist training?
        #lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(rnn_units, norm_gain=0.01,
                                                          # dropout_keep_prob=keep_prob)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                                                  output_keep_prob=keep_prob)  # using keep_prob as a placeholder
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True)
        states, tuple_final = tf.nn.dynamic_rnn(lstm_cells, inputs=sents, sequence_length=sent_length,
                                                dtype=tf.float32, parallel_iterations=256, scope='LSTM')
    # states = tf.reshape(tf.concat(1, states), [-1, max_sent_length, rnn_units])
    # states: tensor, (bs, sl, last rnn layer output size)
    return states, tuple_final[num_layers-1][1]  # tuple_final [(c0,h0),(c1,h1),....]



def sent_encoder_biLstm(incomings, settings, keep_prob=0.75,
                       reuse=None, scope='Sent_Encoder_BiLstm'):
    '''
    Multi-layer Bi-LSTM sentence encoder . 18.01.26 wangyuan
    :param incomings=[sents, sent_length], sents is (batch_size, max_sent_length, embedding_dims) tensor
                                           sent_length is (batch_size,) tensor scalar in [0,1,2,3,...]
           use dynamic_rnn for variable length
    :param num_units: LSTM unit numbers, int
    :param keep_prob: placeholder for dropout, control training or test
    :param scope: variable_scope
    :return: states of each words (batch_size, sent_length, rnn); final states of last layer (bs, rnn)
    '''
    sents, sent_length = incomings
    rnn_units, num_layers = settings
    with tf.variable_scope(scope, reuse=reuse):  # no regularization here
        fw_cell =  tf.nn.rnn_cell.BasicLSTMCell(rnn_units, state_is_tuple=True)
        fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell,
                                                  output_keep_prob=keep_prob)  # using keep_prob as a placeholder
        fw_cell = tf.nn.rnn_cell.MultiRNNCell([fw_cell] * num_layers, state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_units, state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell,
                                                output_keep_prob=keep_prob)  # using keep_prob as a placeholder
        bw_cell = tf.nn.rnn_cell.MultiRNNCell([bw_cell] * num_layers, state_is_tuple=True)


        states, tuple_final = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                                               inputs=sents,
                                               dtype=tf.float32, sequence_length=sent_length,
                                               parallel_iterations=256, scope='BILSTM')
        fw_cell_final = tuple_final[0][num_layers - 1][1]
        bw_cell_final = tuple_final[1][num_layers - 1][1]
    return states, tf.concat([fw_cell_final, bw_cell_final], 1)  # tuple_final [(c0,h0),(c1,h1),....]



def sent_encoder_biLstm_maxpooling(incomings, settings, keep_prob=0.75,
                       reuse=None, scope='Sent_Encoder_BiLstm'):
    '''
    Multi-layer Bi-LSTM sentence encoder . 18.01.26 wangyuan
    :param incomings=[sents, sent_length], sents is (batch_size, max_sent_length, embedding_dims) tensor
                                           sent_length is (batch_size,) tensor scalar in [0,1,2,3,...]
           use dynamic_rnn for variable length
    :param num_units: LSTM unit numbers, int
    :param keep_prob: placeholder for dropout, control training or test
    :param scope: variable_scope
    :return: states of each words (batch_size, sent_length, rnn); final states of last layer (bs, rnn)
    '''
    sents, sent_length = incomings
    rnn_units, num_layers = settings
    with tf.variable_scope(scope, reuse=reuse):  # no regularization here
        fw_cell =  tf.nn.rnn_cell.BasicLSTMCell(rnn_units, state_is_tuple=True)
        fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell,
                                                  output_keep_prob=keep_prob)  # using keep_prob as a placeholder
        fw_cell = tf.nn.rnn_cell.MultiRNNCell([fw_cell] * num_layers, state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_units, state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell,
                                                output_keep_prob=keep_prob)  # using keep_prob as a placeholder
        bw_cell = tf.nn.rnn_cell.MultiRNNCell([bw_cell] * num_layers, state_is_tuple=True)


        states, tuple_final = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                                               inputs=sents, time_major=False,
                                               dtype=tf.float32, sequence_length=sent_length,
                                               parallel_iterations=256, scope='BILSTM')
        outputs = tf.concat(states, 2)
        outputs = tf.reduce_max(outputs, 2)
        fw_cell_final = tuple_final[0][num_layers - 1][1]
        bw_cell_final = tuple_final[1][num_layers - 1][1]
    return outputs , tf.concat([fw_cell_final, bw_cell_final], 1)  # tuple_final [(c0,h0),(c1,h1),....]





def DAMFeat(incomings, vec_dim, num_layers, reuse=None, scope='DAMFeat'):
    '''
    ref: A Decomposable Attention Model for Natural Language Inference
    '''
    q_vec, t_vec = incomings
    feat = tf.concat(axis=2, values=[q_vec, t_vec, q_vec - t_vec, q_vec * t_vec])  # (bs, ns, 4d)
    with tf.variable_scope(scope, reuse=reuse,
        initializer=tf.random_normal_initializer(0.0, 1e-2),
        regularizer=tf.contrib.layers.sum_regularizer(
        [tf.contrib.layers.l2_regularizer(1.0), tf.contrib.layers.l1_regularizer(0.5)])):
        for i in range(num_layers):
            units = 1 if i == num_layers - 1 else vec_dim
            act_fn = tf.nn.sigmoid if i == num_layers - 1 else tf.nn.tanh
            feat = tf.contrib.layers.fully_connected(feat, units, act_fn, scope="H" + str(i))

    return tf.squeeze(feat, name='score')


def NLIFeat(incomings, vec_dim, reuse=None, scope='NLIFeat'):
    '''
    ref: arxiv/abs/1603.06021.
    :param incomings = [q_vec, t_vec] with shape (batch_size, vec_dim) and (batch_size, n_samples, vec_dim)
    :param settings: [batch_size, n_samples, vec_dim]
    '''
    q_vec, t_vec = incomings
    feat_vec = tf.concat(axis=2, values=[q_vec, t_vec, q_vec - t_vec, q_vec * t_vec])  # (bs, ns, 4d)
    with tf.variable_scope(scope, reuse=reuse,
        initializer=tf.random_normal_initializer(0.0, 1e-2),
        regularizer=tf.contrib.layers.sum_regularizer(
         [tf.contrib.layers.l2_regularizer(1.0), tf.contrib.layers.l1_regularizer(0.5)])):
        R_W = tf.get_variable('U', shape=[vec_dim * 4])
        R_b = tf.get_variable('b', shape=[])
        r = tf.reduce_sum(feat_vec * R_W[None, None, :], 2) + R_b  # (bs,ns,4d)--> (bs,ns)
    return tf.nn.sigmoid(r, name='Score')  # out of the variable scope
    #

def CosDis(q_vec, t_vec, reuse=None, scope='CosDis'):
    with tf.variable_scope(scope, reuse=reuse):
        l2_sum = tf.reduce_sum(tf.square(q_vec), 2) * tf.reduce_sum(tf.square(t_vec), 2)
        l2_sum = tf.sqrt(l2_sum) + tf.to_float(1e-6)  # (bs,)
        dot = tf.reduce_sum(q_vec * t_vec, 2)  #(bs,)
        dot = (dot / l2_sum + 1.0) / 2.0
        dot = tf.reduce_max(dot, 1)
        return dot
