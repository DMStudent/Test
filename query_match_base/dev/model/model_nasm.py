from dev.model.model_base import *

class Model(ModelBase):
    def model_setup(self):
        # --------------------------- forward networks -----------------------------------
        # self.n_samples = self.title_sent_in.get_shape()[1].value  # dynamically get n_samples
        query_sent_length = tf.to_int32(tf.reduce_sum(self.query_sent_mask, 1))  # (bs, sl) --> (bs,)
        title_sent_length = tf.reshape(tf.to_int32(tf.reduce_sum(self.title_sent_mask, 2)),
                                       [-1])  # (bs, ns, sl) --> (bs, ns,) --> (bs*ns,)

        # put word embedding here,
        # !!MAKE SURE self.E == tf.trainable_variables()[0]
        with tf.device('/cpu:0'):
            self.E = tf.get_variable('word_embedding', shape=[args.vocab_size, args.embd_dims],
                                     trainable=args.train_embd,
                                     initializer=tf.constant_initializer(self.embedding_init))
        query_sent = word_lookup([self.query_sent_in, self.E],
                                 self.keep_prob, self.is_training, scope='query_lookup')  # (bs, sl, embd_dims)
        title_sent = word_lookup([tf.reshape(self.title_sent_in, [-1, args.max_sent_length]), self.E],
                                 self.keep_prob, self.is_training, scope='title_lookup')  # (bs*ns, sl, embd_dims)

        _, query_enc = sent_encoder_multi([query_sent, query_sent_length], [args.units, args.num_layers],
                                          self.keep_prob, scope='query_encoder')
        title_states, title_final = sent_encoder_multi([title_sent, title_sent_length], [args.units, args.num_layers],
                                                       self.keep_prob, scope='title_encoder')
        title_enc, self.att = NASMAttention([query_enc, title_states, title_final, self.title_sent_mask],
                                       [args.max_sent_length, args.num_layers, args.units],
                                       keep_prob=self.keep_prob, is_training=self.is_training)
        self.score = LogLinearFeat([query_enc[:, None, :] + tf.zeros_like(title_enc), title_enc])

        # --------------------------------  Loss and Training---------------------------------------------
        self.pair_loss, self.acc, self.acc01 = self.pair_loss_op(self.score, self.pair_margin, self.gap_index)
        self.regu_loss = args.l2 * tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.tol_cost = self.pair_loss + self.regu_loss
        self.train_op = self.async_training_op(self.tol_cost, tf.trainable_variables())

        # modified by sam
        #self.train_op = self.async_training_op(self.pair_loss, tf.trainable_variables())

        # -------------- initialization and restore ---------------
        # For distributed version, assign/initialization/saver is not allowed on each GPU
        self.saver = tf.train.Saver(max_to_keep=args.max_to_keep, write_version=saver_pb2.SaverDef.V1)  # save all, including word embeddings
        '''
        # C++ API: tf.train.write_graph(sess.graph_def, '/tmp/my-model', 'train.pbtxt').
        '''
        return tf.trainable_variables(), tf.all_variables(), [self.score]  # vt, vs, vo; vo for C++ api name.
