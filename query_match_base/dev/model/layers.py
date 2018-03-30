import tensorflow as tf
import numpy as np

def word_lookup(incomings, keep_prob=0.75, is_training=True, scope='Word_Lookup'):
    '''
    :param incomings: tf.int64 tensor [batch_size, sent_length], word embedding tensor [vocab_size, dims]
    '''
    inp, E = incomings
    with tf.variable_scope(scope, reuse=None):
        sents_ebd = tf.contrib.layers.dropout(tf.nn.embedding_lookup(E, inp),
                                          keep_prob=keep_prob, is_training=is_training, scope='Dropout')
    return sents_ebd


def sent_encoder(incomings, num_units, keep_prob=0.75, is_training=True,
                 reuse=None, scope='Sent_Encoder'):
    '''
    :param incomings=[sents, sent_length], sents is (batch_size, max_sent_length, embedding_dims) tensor
                                           sent_length is (batch_size,) tensor scalar in [0,1,2,3,...]
           use dynamic_rnn for variable length
    :param num_units: LSTM unit numbers, int
    :param is_training: placeholder indicating the phase
    :param keep_prob: placeholder for dropout
    :param scope: variable_scope
    :return: encodered sentences (batch_size, num_units)
    '''
    sents, sent_length = incomings
    with tf.variable_scope(scope, reuse=reuse, regularizer=tf.contrib.layers.l2_regularizer(1.0)):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units, state_is_tuple=True)
        # lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units, norm_gain=0.01)
        states, tuple_final = tf.nn.dynamic_rnn(lstm_cell, inputs=sents, sequence_length=sent_length,
                                                dtype=tf.float32, parallel_iterations=256, scope='LSTM')
        cell, hid = tuple_final
        hid = tf.contrib.layers.dropout(hid, keep_prob=keep_prob, is_training=is_training, scope='Dropout')
    return hid


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

def sent_decoder(incomings, num_units, word_dict_size, keep_prob=0.75, is_training=True,
                     reuse=None, scope='Sent_Decoder'):
    '''
    :param incomings=[shifted_sents, sent_length, init_state], sents is (batch_size, max_sent_length, embedding_dims) tensor
                                           shifted,
                     sent_length is (batch_size,) tensor
                     init_state is a tuple = (cell(0s), hid), important for generator
    :param num_units: LSTM unit numbers, int
    :param word_dict_size: numbers of words used, for softmax layer.
    :param is_training: placeholder indicating the phase
    :param keep_prob: placeholder for dropout
    :param scope: variable_scope
    :return: word_prob. (batch_size*max_sent_length, word_dict_size)
             tupled final states (for generator) (cell, hid)
    '''
    shifted_sents, sent_length, init_cell, init_hid = incomings
    init_state = tf.nn.rnn_cell.LSTMStateTuple(init_cell, init_hid)
    with tf.variable_scope(scope, reuse=reuse, initializer=tf.random_uniform_initializer(-1e-2, 1e-2)):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units, state_is_tuple=True)
        # lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units, dropout_keep_prob=keep_prob)
        states, tuple_final = tf.nn.dynamic_rnn(lstm_cell, inputs=shifted_sents, initial_state=init_state,
                                                sequence_length=sent_length,  parallel_iterations=32, scope='LM')
        states = tf.contrib.layers.dropout(states, keep_prob=keep_prob, is_training=is_training, scope='Dropout')
        logit = tf.contrib.layers.fully_connected(states, word_dict_size, None, scope='Softmax')
        logit = tf.reshape(logit, [-1, word_dict_size])
        word_prob = tf.nn.softmax(logit)
        # use nn.sampled_softmax_loss when train if with large dictionary
        # or tf.nn.nce_loss
    return word_prob, logit, tuple_final


def TextConvPool(inp, settings, keep_prob=0.75, is_training=True,
                  reuse=None, scope='TextCNN'):
    '''
    Convolution layers for text classification. Haoze Sun 16.10.15
    Use SAME Padding.
    :param inp: 4-d tensor (batch_size, max_sent_length, embd_dims, 1) 4th dimention for channel.
    :param settings: [batch_size, max_sent_length, embd_dims, window_size, filter_num]
    :return:
    '''
    batch_size, max_sent_length, embd_dims, window_size, filter_num = settings
    with tf.variable_scope(scope, reuse=reuse):
        text_conv = tf.contrib.layers.convolution2d(inp, filter_num, [window_size, embd_dims],
                                                    padding='VALID')
        # default (bs, sl, 1, filter_num), VALID padding: (bs, sl-ws+1, 1, filter_num)
        text_pool = tf.contrib.layers.max_pool2d(text_conv, [max_sent_length - window_size + 1, 1])
        text_enc = tf.contrib.layers.flatten(text_pool)
        return tf.contrib.layers.dropout(text_enc, keep_prob=keep_prob, is_training=is_training)
    #

def NASMAttention(incomings, settings, keep_prob=0.6, is_training=True, reuse=None, scope='NASMAtt'):
    '''
    :param query_enc (bs, rnn)
    :param title_states (bs*ns, sl, rnn) actually the 'ns' is unknow value (None)
    :param title_final (bs*ns, rnn)
    :param title_mask (bs, ns, sl)
    :return:
    '''
    # for special collection, use tf.add_to_collection(key, var)
    query_enc, title_states, title_final, title_mask = incomings
    max_sent_length, num_layers, rnn_units = settings
    batch_size = tf.shape(query_enc)[0]  # dynamic shape
    # first we reshape all the incoming titles
    title_states = tf.reshape(title_states, [batch_size, -1, max_sent_length, rnn_units])
    with tf.variable_scope(scope, reuse=reuse,
        initializer=tf.random_normal_initializer(0.0, 1e-2),
        regularizer=tf.contrib.layers.l2_regularizer(1.0)):
        # encode the query_enc again, batch norm no useful.
        # 2016-12-28 multi layers.
        for i in range(num_layers):
            suffix = '' if i == 0 else str(i)  # layer names compatible for earlier versions
            query_enc = tf.contrib.layers.fully_connected(query_enc, rnn_units, tf.nn.tanh,
                                                          scope='query_to_att' + suffix)
            query_enc = tf.contrib.layers.dropout(query_enc, keep_prob=keep_prob,
                                              is_training=is_training, scope='dqa' + suffix)

        # attention with mask
        U_title_att = tf.get_variable('U_title_att', shape=[rnn_units, rnn_units])
        U_query_att = tf.get_variable('U_query_att', shape=[rnn_units, rnn_units])
        W_title_c = tf.get_variable('W_title_c', shape=[rnn_units, rnn_units])
        W_title_f = tf.get_variable('W_title_f', shape=[rnn_units, rnn_units])
        V_att = tf.get_variable('V_att', shape=[rnn_units])

        u_t = tf.matmul(tf.reshape(title_states, [-1, rnn_units]), U_title_att)  # (bs*ns*sl, rnn)
        u_q = tf.matmul(query_enc, U_query_att)  # (bs, rnn)
        u_sum = tf.reshape(u_t, [batch_size, -1, max_sent_length, rnn_units]) + u_q[:, None, None, :]  # (bs, ns, sl, rnn)
        #a = tf.reduce_sum(tf.nn.tanh(u_sum) * V_att[None, None, None, :], 3)  # (bs, ns, sl)
        u_sum = tf.nn.tanh(u_sum)
        a = tf.reshape(
            tf.matmul(tf.reshape(u_sum, (-1, rnn_units)), V_att[:, None]),  # (...,rnn) (rnn, 1)->(...,1)
            (batch_size, -1, max_sent_length))
        exp_a = tf.exp(a - tf.reduce_max(a, 2, True)) + 1e-6  # (bs, ns ,sl)
        #exp_a = tf.exp(a - tf.reduce_max(a-1000.0*(1.0-title_mask), 2, True))  # (bs, ns ,sl)

        exp_a_masked = exp_a * title_mask
        e = exp_a_masked / (1e-6 + tf.reduce_sum(exp_a_masked, 2, True))  # (bs, ns , sl) softmax
        #title_att_enc = tf.reduce_sum(title_states * e[:, :, :, None], 2)  # (bs, ns, rnn)
        title_att_enc = tf.squeeze(tf.batch_matmul(e[:, :, None, :], title_states))  # (bs, ns, 1, rnn)
        w_tc = tf.matmul(tf.reshape(title_att_enc, [-1, rnn_units]), W_title_c)  # (bs*ns, rnn)
        w_tf = tf.matmul(title_final, W_title_f)
        att_enc = tf.reshape(tf.nn.tanh(w_tc + w_tf), [batch_size, -1, rnn_units])
    return att_enc, e


def DAMAttention(incomings,
                 units, num_layers=2, use_intra=True,
                 keep_prob=0.75, is_training=True, scope='DAMAttention'):
    '''
    ref: A Decomposable Attention Model for Natural Language Inference
    symmetri hypothesis:
    asymmetric structure is better for QA tasks like query-title?
    symmetric structure is better for NLI llike tasks?
    '''

    def dense(inp, reuse=None, scope='inter'):
        with tf.variable_scope(scope, reuse=reuse,
            regularizer=tf.contrib.layers.l2_regularizer(1.0)):
            for i in range(num_layers):
                inp = tf.contrib.layers.fully_connected(inp, units, tf.nn.tanh, scope='Dense' + str(i))
                inp = tf.contrib.layers.dropout(inp, keep_prob=keep_prob, is_training=is_training,
                                                scope='dDense' + str(i))
            return inp

    def decompos(q, t):
        '''
        q (bs, ns, sq, n) --> (bs, ns, sq, n ,1)
        t (bs, ns, st, n) --> (bs, ns, n, st) --> (bs, ns, 1, n, st)
        '''
        et = tf.batch_matmul(q, tf.transpose(t, [0, 1, 3, 2]), name='deco_mul')
        '''
        e = tf.mul(q[:, :, :, :, None],
                   tf.transpose(t, [0, 1, 3, 2])[:, :, None, :, :],
                   name='decomposable_mul')
        # e (bs, ns, sq, units, st)
        et = tf.reduce_sum(e, 3)  # (bs, ns, sq, st), attention on title
        '''
        eq = tf.transpose(et, [0, 1, 3, 2])  # (bs, ns, st ,sq), attention on query
        return et, eq

    def att(e, enc, m):
        '''
        e (bs, ns, sq, st), softmax on last dim (3)
        enc (bs, ns, st, n),
        m (bs, ns, st)
        '''
        gamma = (tf.exp(e - tf.reduce_max(e, 3, True)) + 1e-8) * m[:, :, None, :]  # (bs, ns, sq ,st)
        gamma /= (1e-8 + tf.reduce_sum(gamma, 3, True))
        gamma = tf.batch_matmul(gamma, enc, name='att_mul')
        # (bs, ns, sq, st) (bs, ns, st, n) --> (bs, ns, sq, n)
        return gamma  # (bs, ns, sq, n)

    def aggregate(v, m, scope='BothAgg', reuse=None):
        v = dense(v, scope=scope, reuse=reuse)
        # v (bs, ns, sl, n) --> (bs, ns, n), use mean with mask
        # v = tf.reduce_sum(tf.mul(v, m[:, :, :, None], name='agg_mul'), 2)  # (bs, ns, sl, n) -->  (bs, ns, n)
        v = tf.reshape(tf.batch_matmul(m[:, :, None, :], v, name='agg_mul'),
                       tf.concat(0, [tf.shape(m)[:2], [units]]))
        v /= (tf.reduce_sum(m, 2, True) + 1e-6)
        return v

    def intra_attention(ra, ia, am, dis_diff, reuse=None):
        '''
        ia (bs, ns, sq, units)
        ra  raw word embedding representation of sentence (bs, ns, sq, embd_dims)
        am, its mask (bs, ns, sq)
        '''
        with tf.variable_scope(scope, reuse=reuse,
            regularizer=tf.contrib.layers.l2_regularizer(1.0)):
            dis_w = tf.get_variable('dis_w', [], dtype=tf.float32)
            dis_b = tf.get_variable('dis_b', [], dtype=tf.float32)
            d = tf.nn.softplus(dis_diff * tf.nn.softplus(dis_w) + dis_b)
            #d = dis_diff * tf.nn.softplus(dis_w)

        e, _ = decompos(ia, ia)
        e += d[None, None, :, :]  # (bs, ns, sq ,sq)
        a_bar = tf.concat(3, [att(e, ra, am), ra], name='abar')
        return a_bar  # (bs, ns, sq, ebd_dims + units)

    rq, qm, rt, tm, dis_diff = incomings
    # rq (bs, sq, n), qm (bs, sq), rt (bs, ns, st, n), tm (bs, ns, st), dis_diff (sq, st)
    input_shape = rq.get_shape()[-1].value
    cur_max_query_length = tf.shape(qm)[-1]
    cur_max_title_length = tf.shape(tm)[-1]
    n_samples = tf.shape(tm)[1]

    # tile the queries
    rq = tf.tile(tf.reshape(rq, [-1, 1, cur_max_query_length, input_shape]), [1, n_samples, 1, 1])
    qm = tf.tile(tf.reshape(qm, [-1, 1, cur_max_query_length]), [1, n_samples, 1])


    if use_intra:
        iq = dense(rq, scope='Intra')
        it = dense(rt, scope='Intra', reuse=True)
        rq = intra_attention(rq, iq, qm, dis_diff[:cur_max_query_length, :cur_max_query_length])
        rt = intra_attention(rt, it, tm, dis_diff[:cur_max_title_length, :cur_max_title_length],
                             reuse=True)

    q = dense(rq, scope='Inter')  # (bs, ns, sq, units or input_shape + units)
    t = dense(rt, scope='Inter', reuse=True)  # (bs, ns, st, ...)

    et, eq = decompos(q, t)  # et (bs, ns, sq, st), eq (bs, ns, st, sq)
    beta = att(et, rt, tm)  # (bs, ns, sq, n)
    alpha = att(eq, rq, qm)  # (bs, ns, st, n)
    vq = tf.concat(3, [rq, beta])  # (bs, ns, sq, 3n)
    vt = tf.concat(3, [rt, alpha])  # (bs, ns, st, 3n)

    vgq = aggregate(vq, qm, scope='Agg')
    vgt = aggregate(vt, tm, scope='Agg', reuse=True)

    return vgq, vgt


def LogLinearFeat(incomings, reuse=None, scope='LogLinear'):
    '''
    ref: NASM
    q_vec, t_vec have shape(bs, ns, vec_dim)
    '''
    q_vec, t_vec = incomings  # each with 3 dims, q_vec = q_vec[:,None,:] + tf.zeros_like(t_vec)
    feat_concat = tf.concat(2, [q_vec, t_vec])

    vec_dim = t_vec.get_shape()[-1].value  # static value, if None, raise error.
    print("vec_dim:%d"%vec_dim)

    batch_size = tf.shape(t_vec)[0]  # batch size is dynamic.

    with tf.variable_scope(scope, reuse=reuse,
        initializer=tf.random_normal_initializer(0.0, 1e-2),
        regularizer=tf.contrib.layers.sum_regularizer(
         [tf.contrib.layers.l2_regularizer(1.0), tf.contrib.layers.l1_regularizer(0.5)])):

        U = tf.get_variable('U', shape=[vec_dim, vec_dim])
        V = tf.get_variable('V', shape=[2 * vec_dim])
        b = tf.get_variable('b', shape=[])

        r = b + tf.reshape(
                tf.matmul(tf.reshape(feat_concat, [-1, 2 * vec_dim]), V[:, None]),
                [batch_size, -1])
        # (bs*ns, 2rnn)(2rnn, 1) --> (bs, ns, 1) --> (bs, ns)
        mid = tf.reshape(tf.matmul(tf.reshape(q_vec, [-1, vec_dim]), U), [batch_size, -1, vec_dim])
        r += tf.reduce_sum(mid * t_vec, 2)  # mul+sum better than batch_matmul
        # (bs, ns, rnn) dot (rnn, rnn) --> (bs, ns, rnn); (bs, None, rnn) * (bs, ns, rnn) --> (bs. ns)
    return tf.nn.sigmoid(r, name='Score')  # out of the variable scope
    #


def DAMFeat(incomings, vec_dim, num_layers, reuse=None, scope='DAMFeat'):
    '''
    ref: A Decomposable Attention Model for Natural Language Inference
    '''
    q_vec, t_vec = incomings
    feat = tf.concat(2, [q_vec, t_vec, q_vec - t_vec, q_vec * t_vec])  # (bs, ns, 4d)
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
    feat_vec = tf.concat(2, [q_vec, t_vec, q_vec - t_vec, q_vec * t_vec])  # (bs, ns, 4d)
    with tf.variable_scope(scope, reuse=reuse,
        initializer=tf.random_normal_initializer(0.0, 1e-2),
        regularizer=tf.contrib.layers.sum_regularizer(
         [tf.contrib.layers.l2_regularizer(1.0), tf.contrib.layers.l1_regularizer(0.5)])):
        R_W = tf.get_variable('U', shape=[vec_dim * 4])
        R_b = tf.get_variable('b', shape=[])
        r = tf.reduce_sum(feat_vec * R_W[None, None, :], 2) + R_b  # (bs,ns,4d)--> (bs,ns)
    return tf.nn.sigmoid(r, name='Score')  # out of the variable scope
    #

def CosDis(incomings, reuse=None, scope='CosDis'):
    q_vec, t_vec = incomings
    with tf.variable_scope(scope, reuse=reuse):
        l2_sum = tf.reduce_sum(tf.square(q_vec), 2) * tf.reduce_sum(tf.square(t_vec), 2)
        l2_sum = tf.sqrt(l2_sum) + tf.to_float(1e-6)  # (bs,)
        dot = tf.reduce_sum(q_vec * t_vec, 2)  #(bs,)
        return (dot / l2_sum + 1.0)/2.0
