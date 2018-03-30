# --coding:utf-8 --
from dev.model.layers.base import *
from tensorflow.python.framework import ops
cuda_conv_fw = tf.load_op_library('/search/odin/hzsun/new_webrank_relevance/dev/model/libs/kernel_fo_conv.so')
cuda_conv_bk = tf.load_op_library('/search/odin/hzsun/new_webrank_relevance/dev/model/libs/kernel_fo_conv_grad.so')
cuda_pooling_fw = tf.load_op_library('/search/odin/hzsun/new_webrank_relevance/dev/model/libs/kernel_fo_pooling.so')
cuda_pooling_bk = tf.load_op_library('/search/odin/hzsun/new_webrank_relevance/dev/model/libs/kernel_fo_pooling_grad.so')

@ops.RegisterGradient("KernelFoPooling")
def _kernel_fo_pooling_grad(op, g_hid, g_hid_final, g_cell):
    Z, F, O, seq_length = op.inputs
    cell = op.outputs[2]
    #g_hid, g_hid_final = grad
    g_Z, g_F, g_O = cuda_pooling_bk.kernel_fo_pooling_grad(Z, F, O, seq_length, cell, g_hid, g_hid_final)
    return [g_Z, g_F, g_O, None]

@ops.RegisterGradient("KernelFoConv")
def _kernel_fo_conv_grad(op, g_concat_seq, g_concat_zfo, g_Z, g_F, g_O):
    _, conv_w, seq_length, _, _ = op.inputs
    concat_seq, Z, F, O = op.outputs[0], op.outputs[2], op.outputs[3], op.outputs[4]
    #g_hid, g_hid_final = grad
    g_seq, _, _, g_w, g_b, g_proj = \
        cuda_conv_bk.kernel_fo_conv_grad(concat_seq, conv_w, seq_length, Z, F, O, g_Z, g_F, g_O)
    return [g_seq, g_w, None, g_b, g_proj]


def CuQRNNLayer(seq_enc, seq_mask, num_units, initial_state=None,
              keep_prob=0.9, zoneout_keep_prob=0.9, is_training=True, reuse=None, scope='CuQRNN'):

    valid_seq_length = tf.to_int32(tf.reduce_sum(seq_mask, 1))
    batch_size =tf.shape(seq_enc)[0]
    inp_size = seq_enc.get_shape()[-1].value

    with tf.variable_scope(scope, reuse=reuse):
        with tf.variable_scope('ConvZFO', reuse=reuse,
             initializer=tf.random_normal_initializer(0.0, 5e-2),
             regularizer=tf.contrib.layers.l2_regularizer(1.0)):  # keep the scope and shape
            conv_w = tf.get_variable('conv_w', [2, inp_size, 3 * num_units])
            conv_w_re = tf.reshape(conv_w, [2 * inp_size, 3 * num_units])
            conv_b = tf.get_variable('conv_b', [3 * num_units])
            if initial_state is not None:
                assert (initial_state.get_shape().ndims == 2) 
                init_size = initial_state.get_shape()[-1].value
                init_v = tf.get_variable('init_v', [init_size, 3 * num_units])
                init_proj = tf.matmul(initial_state, init_v, name='initial_state_projection')
            else:
                init_proj = tf.zeros([batch_size, 3 * num_units], dtype=tf.float32)

        _, _, Z, F, O = \
            cuda_conv_fw.kernel_fo_conv(seq_enc, conv_w_re, valid_seq_length, conv_b, init_proj)
        F = tf.contrib.layers.dropout(1.0 - F,
                                      keep_prob=zoneout_keep_prob, is_training=is_training)
        F = smart_cond(is_training, lambda: F * zoneout_keep_prob, lambda: F)
        F = 1.0 - F

        hid_states, hid_final, cell_states = cuda_pooling_fw.kernel_fo_pooling(Z, F, O, valid_seq_length)
        hid_final = tf.contrib.layers.dropout(hid_final, keep_prob=keep_prob, is_training=is_training)
        #cell_states = tf.contrib.layers.dropout(cell_states, keep_prob=keep_prob, is_training=is_training)
        hid_states = tf.contrib.layers.dropout(hid_states, keep_prob=keep_prob, is_training=is_training)
        return hid_states, hid_final, O, cell_states

