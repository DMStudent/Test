# --coding: utf-8 --
from dev.model.layers.base import *

def DotMatchAttention(source_states, source_mask, target_states, target_mask,
                     mode='concat', out_dim=None,
                     l2=1.0, smooth=1e-6,
                     keep_prob=0.9, is_training=True, reuse=None, scope='DotMatchAtt'):
    assert (mode in ['linear', 'concat'])
    batch_size = tf.shape(target_states)[0]
    n_samples = tf.shape(target_states)[1]
    cur_source_length = tf.shape(source_mask)[-1]
    cur_target_length = tf.shape(target_mask)[-1]
    source_ndims = source_states.get_shape()[-1].value
    target_ndims = target_states.get_shape()[-1].value

    # expand the dim of source sequence
    if source_mask.get_shape().ndims == 2:
        source_states *= source_mask[:, :, None]  # 前置蒙版
        source_states = tf.tile(tf.reshape(source_states, [batch_size, 1, cur_source_length, source_ndims]),
                            [1, n_samples, 1, 1])  # (bs, ns, sq, nq)
        source_mask = tf.tile(tf.reshape(source_mask, [batch_size, 1, cur_source_length]),
                          [1, n_samples, 1])  # (bs, ns, sq)
    elif source_mask.get_shape().ndims == 3:
        source_states *= source_mask[:, :, :, None]
    else:
        raise ValueError('Source must have rank 2 or 3')

    with tf.variable_scope(scope, reuse=reuse,
                           initializer=tf.random_normal_initializer(0.0, 5e-3),
                           regularizer=tf.contrib.layers.sum_regularizer(
                           [tf.contrib.layers.l2_regularizer(l2), tf.contrib.layers.l1_regularizer(0.5 * l2)])):

        trans_source = tf.transpose(source_states, [0, 1, 3, 2], name='transpose_alpha')  # (bs, ns, nq, sq)
        # bilinear alpha = t'M q;  (bs, ns, st, sq)
        M = tf.get_variable('M_att', [target_ndims, source_ndims])
        alpha = tf.matmul(tf.reshape(target_states, [-1, target_ndims]), M, name='matmul_alpha')  # (bs*ns*st, nq)
        alpha = tf.contrib.layers.dropout(alpha, keep_prob=keep_prob, is_training=is_training)
        alpha = tf.reshape(alpha, [batch_size, n_samples, cur_target_length, source_ndims], name='reshape_alpha')
        alpha = tf.matmul(alpha, trans_source, name='batch_matmul_alpha')

        mask_a = tf.matmul(tf.reshape(target_mask, [batch_size, n_samples, cur_target_length, 1]),
                             tf.reshape(source_mask, [batch_size, n_samples, 1, cur_source_length]),
                             name='dot_mask')
        alpha *= mask_a  
        alpha_l1 = tf.reduce_sum(tf.abs(alpha)) / (1e-3 + tf.reduce_sum(mask_a))

        valid_offset = tf.reduce_max(alpha - 10.0 * (1.0 - mask_a), 3, True, name='valid_offset')
        # (bs, ns, st, 1)
        exp_a = tf.exp(alpha - valid_offset, name='exp_alpha') + smooth  # (bs, ns, st, sq)
        exp_a_masked = exp_a * mask_a  # (bs, ns, st, sq)
        e = exp_a_masked / (1e-8 + tf.reduce_sum(exp_a_masked, 3, True))  # (bs,ns,st,sq) / (bs,ns,st,1)
        trans_e = tf.transpose(e, [0, 1, 3, 2])  # (bs, ns, sq, st)

        c = tf.matmul(e, source_states, name='batch_matmul_c')

    with tf.variable_scope(scope, reuse=reuse,
                           initializer=tf.random_normal_initializer(0.0, 5e-3),
                           regularizer=tf.contrib.layers.sum_regularizer(
                               [tf.contrib.layers.l2_regularizer(1.0), tf.contrib.layers.l1_regularizer(0.5)])):

        if mode == 'linear':
	    final_enc = tf.maximum(c, target_states)
        elif mode == 'concat':
            final_enc = tf.concat([c, target_states], 3)  # concat results, not masked!!
        else:
            raise ValueError('>>>>???<<<<<')

    return final_enc * target_mask[:, :, :, None], alpha_l1  # (bs,st,nq+nt) or (bs,st,nt); (bs,ns,sq,st)