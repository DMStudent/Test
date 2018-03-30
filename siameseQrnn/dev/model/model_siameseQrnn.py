# --coding:utf-8--
from dev.model.model_base import *
from dev.model.quasi_rnn import *
from dev.model.layers import *
from fuzzywuzzy import fuzz

class Model(ModelBase):

    def extra_embedding_init(self):
        self.docid_map = {}
        docid_stream = open(args.docid_path, 'r')
        while True:
            line = docid_stream.readline()
            if line == '':
                break
            line = line.strip()
            tokens = line.split('\t')
            self.docid_map[tokens[0]] = int(tokens[1])
        np.random.seed(1234567890)
        self.url_init = 5e-2 * \
                        np.random.randn(110000, 100).astype(np.float32)
        np.random.seed()
    
    def data_proc(self, qt_text_list, title_steps=2, n_samples=args.n_samples):
        '''
        :param qt_text: ['w w w\tw w w w\tw w w ...',...,.]  = [query, title1, title2.....]
        '''
        q = np.zeros((args.batchsize, args.max_sent_length), dtype=np.int32)
        extra_score = np.zeros((args.batchsize, n_samples, 2), dtype=np.float32)
        qm = np.zeros_like(q, dtype=np.float32)
        t = np.zeros((args.batchsize, n_samples, args.max_sent_length), dtype=np.int32)
        tm = np.zeros_like(t, dtype=np.float32)
        dm = np.zeros((args.batchsize, n_samples), dtype=np.int32)
        label = np.zeros((args.batchsize), dtype=np.int32)
        cur_max_q_len = 0
        cur_max_t_len = 0

        for bid, qt_text_line in enumerate(qt_text_list):
            qt_text = qt_text_line.strip().split('\t')  # ['w w w', 'w w w w', ....]
            query = qt_text[1]
            q_id_list, q_id_len = self.word2id(qt_text[2])
            label[bid] = float(qt_text[0])
            q[bid, :q_id_len] = q_id_list
            qm[bid, :q_id_len] = 1.0
            cur_max_q_len = max(cur_max_q_len, q_id_len)

            dlen = min((len(qt_text)-3)/2, n_samples)
            # print "--------------------------"
            # print "dlen:" + str(dlen)
            dm[bid, :dlen] = 1.0
            for ns in range(dlen):
                title = qt_text[3 + title_steps * ns]
                score1 = float(fuzz.ratio(query, title))/100
                score2 = float(fuzz.partial_ratio(query, title)) / 100
                extra_score[bid, ns, :] = [score1, score2]
                t_id_list, t_id_len = self.word2id(qt_text[4 + title_steps * ns])
                t[bid, ns, :t_id_len] = t_id_list
                tm[bid, ns, :t_id_len] = 1.0
                cur_max_t_len = max(cur_max_t_len, t_id_len)
            # print [score1, score2]
            # print [score1/dlen, score2/dlen]

            # print extra_score[bid, :]
        return label, q, qm, t, tm, dm, extra_score

    def run_epoch(self, sess, inp, istrn, run_options=None, run_metadata=None):
        label, q, qm, t, tm, dm, extra_score = inp
        if istrn:
            feed_dict = {self.query_sent_in: q, self.query_sent_mask: qm,
                         self.title_sent_in: t, self.title_sent_mask: tm,
                         self.title_doc_mask: dm, self.label: label, self.extra_score: extra_score, self.is_training: True}
            fetch = [self.global_step, self.tol_cost, self.loss,
                     self.acc, self.score, self.train_op]
        else:
            feed_dict = {self.query_sent_in: q, self.query_sent_mask: qm,
                         self.title_sent_in: t, self.title_sent_mask: tm,
                         self.title_doc_mask: dm, self.label: label, self.extra_score: extra_score, self.is_training: False}
            fetch = [self.acc, self.score]

        if run_options is not None and run_metadata is not None:
            return sess.run(fetch, feed_dict, options=run_options, run_metadata=run_metadata)
        else:
            return sess.run(fetch, feed_dict)

    def run_score(self, sess, inp):
        q, qm, t, tm = inp
        feed_dict = {self.query_sent_in: q, self.query_sent_mask: qm,
                     self.title_sent_in: t, self.title_sent_mask: tm,
                     self.is_training: False}
        fetch = [self.score]
        return sess.run(fetch, feed_dict)

    def model_graph(self):
        # shape variables for each batched input data
        batch_size = tf.shape(self.title_sent_mask)[0]  # (batch_size, n_samples,) dynamic shape
        cur_max_query_length = tf.shape(self.query_sent_mask)[-1]
        cur_max_title_length = tf.shape(self.title_sent_mask)[-1]
        n_samples = tf.shape(self.title_sent_mask)[1]
        query_sent_length = tf.to_int32(tf.reduce_sum(self.query_sent_mask, 1))  # (bs, sl) --> (bs,)
        title_sent_length = tf.reshape(tf.to_int32(tf.reduce_sum(self.title_sent_mask, 2)),[-1])  # (bs, ns, sl) --> (bs, ns,) --> (bs*ns,)
        # --------------------------- forward networks -----------------------------------
        #with tf.device('/cpu:0'):
        self.E = tf.create_partitioned_variables([args.vocab_size, args.embd_dims], [16, 1],
                     initializer=tf.constant(self.embedding_init, dtype=tf.float32),
                     dtype=tf.float32, trainable=args.train_embd,
                     collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.NO_CLIP],
                     name='word_embedding')

        query_ebd = word_lookup(self.query_sent_in, self.E,
                                0.8, self.is_training, scope='query_lookup')  # (bs, sq, embd_dims)
        title_ebd = word_lookup(self.title_sent_in, self.E,
                                0.8, self.is_training, scope='title_lookup')  # (bs, ns, st, embd_dims)
        title_ebd = tf.reshape(title_ebd, (-1, cur_max_title_length, args.embd_dims))  # (bs*ns, st, ebd_dims)
        title_sent_mask_re = tf.reshape(self.title_sent_mask, [-1, cur_max_title_length])


        # -----------------  encode query and then shift, tile.  ----------------------
        # query_seq, query_final, _, _ = CuQRNNLayer(query_ebd, query_sent_length, 284,
        #                       keep_prob=0.8, zoneout_keep_prob=0.9, is_training=self.is_training,
        #                       scope='QuerySeq')
        #
        # # ----------------  encode titles  ---------------------
        # title_seq, title_final, _, _ = CuQRNNLayer(title_ebd, title_sent_length, 284,
        #                      keep_prob=0.8, zoneout_keep_prob=0.9, is_training=self.is_training,
        #                      scope='TitleSeq')
        # sent_encoder_multi
        query_state, query_final = sent_encoder_biLstm([query_ebd, query_sent_length], [284, 1],
                                          0.8, scope='query_encoder')

        title_state, title_final = sent_encoder_biLstm([title_ebd, title_sent_length], [284, 1],
                                            0.8, scope='title_encoder')

        # query_state, query_final = sent_encoder_biLstm_maxpooling([query_ebd, query_sent_length], [284, 1],
        #                                                0.8, scope='query_encoder')
        #
        # title_state, title_final = sent_encoder_biLstm_maxpooling([title_ebd, title_sent_length], [284, 1],
        #                                                0.8, scope='title_encoder')


        # ----------- calc score use regression or bi-linear  -------------
        self.score = MatchingMethodsCombine(tf.tile(tf.reshape(query_final, [-1, 1, query_final.get_shape()[-1].value]),
                                           [1, n_samples, 1]),
                                           tf.reshape(title_final,
                                           [batch_size, n_samples, title_final.get_shape()[-1].value]), self.is_training, self.title_doc_mask, self.extra_score)


        # --------------------------------  Loss and Training---------------------------------------------

        self.loss, self.acc = self.mse_loss_op(self.score, self.label)
        self.regu_loss = args.l2 * tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.regu_loss = 0
        # 加正则防止embedding过快增长
        # self.regu_loss += args.l2 * tf.reduce_sum(url_enc ** 2) / 2.0

        self.tol_cost = self.loss + self.regu_loss
        self.tol_cost = self.loss
        self.train_op = self.async_training_op(self.tol_cost, tf.trainable_variables())


