from dev.model.model_base import *

class Model(ModelBase):
    def model_setup(self):
        # shape variables for each batched input data
        # cur_max_query_length = tf.shape(self.query_sent_mask)[-1]
        cur_max_title_length = tf.shape(self.title_sent_mask)[-1]
        n_samples = tf.shape(self.title_sent_mask)[1]

        query_sent_length = tf.to_int32(tf.reduce_sum(self.query_sent_mask, 1))  # (bs, sq) --> (bs,)
        title_sent_length = tf.reshape(tf.to_int32(tf.reduce_sum(self.title_sent_mask, 2)), [-1])  # (bs*ns,)

        # --------------------------- forward networks ----------------------------------
        # put word embedding here,
        # !!MAKE SURE self.E == tf.trainable_variables()[0]
        with tf.device('/cpu:0'):
            self.E = tf.get_variable('word_embedding', shape=[args.vocab_size, args.embd_dims],
                                     trainable=args.train_embd,
                                     initializer=tf.constant_initializer(self.embedding_init))

        query_ebd = word_lookup([self.query_sent_in, self.E],
                                self.keep_prob, self.is_training)  # (bs, sl, embd_dims)
        title_ebd = word_lookup([self.title_sent_in, self.E],
                                self.keep_prob, self.is_training)  # (bs, ns, sl, embd_dims)

        # encoded by lstm structure, reshape first
        title_ebd = tf.reshape(title_ebd, [-1, cur_max_title_length, args.embd_dims])

        query_ebd, _ = sent_encoder_multi([query_ebd, query_sent_length], [args.units, 1],
                                       self.keep_prob, scope='EncoderRNN')
        title_ebd, _ = sent_encoder_multi([title_ebd, title_sent_length], [args.units, 1],
                                       self.keep_prob, scope='EncoderRNN', reuse=True)  # 2 rnn better?
        title_ebd = tf.reshape(title_ebd, [-1, n_samples, cur_max_title_length, args.units])

        # distance for DAM.
        dis = np.tile(np.arange(args.max_sent_length), (args.max_sent_length, 1))
        dis = tf.constant(dis, dtype=tf.float32, name='dis')
        dis_diff = tf.abs(dis - tf.transpose(dis, [1, 0]), name='dis_diff')
        dis_diff = tf.minimum(dis_diff, tf.to_float(20.0), name='dis_diff')

        # Decomposable Attention layer
        query_enc, title_enc = DAMAttention([query_ebd, self.query_sent_mask,
                                             title_ebd, self.title_sent_mask, dis_diff],
                                            args.units, args.num_layers, args.use_intra,
                                            self.keep_prob, self.is_training)
        # score layer
        #self.score = DAMFeat([query_enc, title_enc], args.units, 1)
        #self.score = CosDis([query_enc, title_enc])
        self.score = LogLinearFeat([query_enc, title_enc])
        # --------------------------------  Loss and Training---------------------------------------------
        self.pair_loss, self.acc, self.acc01 = self.pair_loss_op(self.score, self.pair_margin, self.gap_index)
        self.regu_loss = args.l2 * tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        self.tol_cost = self.pair_loss + self.regu_loss
        self.train_op = self.async_training_op(self.tol_cost, tf.trainable_variables())

        # -------------- initialization and restore ---------------
        # For distributed version, assign/initialization/saver is not allowed on each GPU
        self.saver = tf.train.Saver(max_to_keep=args.max_to_keep, write_version=saver_pb2.SaverDef.V1)  # save all, including word embeddings
        '''
        # C++ API: tf.train.write_graph(sess.graph_def, '/tmp/my-model', 'train.pbtxt').
        '''
        return tf.trainable_variables(), tf.all_variables(), [self.score]  # vt, vs, vo; vo for C++ api name.

