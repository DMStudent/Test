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

def BiLinearFeat(q_vec, t_vec, reuse=None, scope='LogLinear'):
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
    return tf.nn.sigmoid(r, name='Score')  # out of the variable scope
    #
