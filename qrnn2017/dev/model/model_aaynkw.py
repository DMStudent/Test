# --coding:utf-8--
from dev.model.model_base import *
from dev.model.libs.quasi_rnn import *

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

    def search_doc_id(self, docid_str):
        arr = docid_str.split("-")
        id_leve2 = ""
        if len(arr) < 1:
            return -1
        if len(arr) > 1:
            id_leve2 = arr[0] + "-" + arr[1]
            if id_leve2 in self.docid_map:
                return self.docid_map[id_leve2]
        if arr[0] in self.docid_map:
            return self.docid_map[arr[0]]
        return -1
    
    def data_proc(self, qt_text_list, title_steps=4, n_samples=args.n_samples):
        '''
        :param qt_text: ['w w w\tw w w w\tw w w ...',...,.]  = [query, title1, title2.....]
        '''
        def uniq_from_title(kw_id_list, t_id_list):
            t_id_set = set(t_id_list)
            t_id_set.add(0)
            kw_id_set = set(kw_id_list)
            id_list = list(kw_id_set - t_id_set)
            return id_list, len(id_list)

        q = np.zeros((args.batchsize, args.max_sent_length), dtype=np.int32)
        qm = np.zeros_like(q, dtype=np.float32)
        t = np.zeros((args.batchsize, n_samples, args.max_sent_length), dtype=np.int32)
        tm = np.zeros_like(t, dtype=np.float32)
        # t_url = np.zeros((args.batchsize, n_samples), dtype=np.int32)
        kw = np.zeros((args.batchsize, n_samples, args.n_keyword), dtype=np.int32)
        km = np.zeros((args.batchsize, n_samples, args.n_keyword), dtype=np.float32)
	g = np.ones((args.batchsize), dtype=np.float32)

        cur_max_q_len = 0
        cur_max_t_len = 0

        for bid, qt_text_line in enumerate(qt_text_list):
            qt_text = qt_text_line.strip().split('\t')  # ['w w w', 'w w w w', ....]
            # assert (len(qt_text) >= 2 * args.n_samples + 1)
	    if args.use_g and len(qt_text) == 2 + title_steps * n_samples:
		g[bid] = np.float32(qt_text[-1])

            q_id_list, q_id_len = self.word2id(qt_text[0])
            q[bid, :q_id_len] = q_id_list
            qm[bid, :q_id_len] = 1.0
            cur_max_q_len = max(cur_max_q_len, q_id_len)

            for ns in range(n_samples):
                t_id_list, t_id_len = self.word2id(qt_text[1 + title_steps * ns])
                t[bid, ns, :t_id_len] = t_id_list
                tm[bid, ns, :t_id_len] = 1.0
                cur_max_t_len = max(cur_max_t_len, t_id_len)

                # url_id = 109999
                # url_id = self.search_doc_id(qt_text[2 + title_steps * ns])
                # if url_id == -1:
                #     url_id = 109999
                # t_url[bid, ns] = np.int32(url_id)

            has_kw_flag = 1.0
            for ns in range(n_samples):
                kw1 = qt_text[3 + title_steps * ns].strip().split(' ')
                kw2 = qt_text[4 + title_steps * ns].strip().split(' ')
		kw1 = [x.lstrip('0') for x in kw1]
		kw2 = [x.lstrip('0') for x in kw2]
                if (len(kw1) + len(kw2) != args.n_keyword) or has_kw_flag != 1.0:
                    has_kw_flag = 0.0
                    break
                #
		kw1 = [w.split('|')[0] for w in kw1]
                kw2 = [w.split('|')[0] for w in kw2]
                kw_1_2 = ' '.join(kw1 + kw2)
                kw_id_list, _ = self.word2id(kw_1_2)

                # kw_uniq_id_list, kw_id_len = uniq_from_title(kw_id_list, t[bid, ns, :])
                # kw[bid, ns, :kw_id_len] = kw_uniq_id_list
                # km[bid, ns, :kw_id_len] = 1.0
                kw[bid, ns, :] = kw_id_list
                km[bid, ns, :] = 1.0

            km[bid, :, :] *= has_kw_flag

        q = q[:, :cur_max_q_len]
        qm = qm[:, :cur_max_q_len]
        t = t[:, :, :cur_max_t_len]
        tm = tm[:, :, :cur_max_t_len]

        return q, qm, t, tm, kw, km, g

    def extra_feed(self):
        self.keyword_in = tf.placeholder(tf.int32, [None, None, args.n_keyword])
        self.keyword_mask = tf.placeholder(tf.float32, [None, None, args.n_keyword])
	self.g = tf.placeholder(tf.float32, [None])

    def run_epoch(self, sess, inp, istrn, run_options=None, run_metadata=None):
        q, qm, t, tm, kw, km, g = inp
        if istrn:
            feed_dict = {self.query_sent_in: q, self.query_sent_mask: qm,
                         self.title_sent_in: t, self.title_sent_mask: tm,
                        self.keyword_in: kw, self.keyword_mask: km,
			self.g: g,
                        self.is_training: True, self.keep_prob: args.keep_prob,
                        self.pair_margin: args.pair_margin}
            fetch = [self.global_step, self.tol_cost, self.pair_loss, self.regu_loss,
                     self.acc, self.acc01, self.score, self.train_op]
        else:
            feed_dict = {self.query_sent_in: q, self.query_sent_mask: qm,
                         self.title_sent_in: t, self.title_sent_mask: tm,
                         self.keyword_in: kw, self.keyword_mask: km,
			 self.g: g,
                         self.is_training: False, self.keep_prob: 1.0,
                         self.pair_margin: args.pair_margin}
            fetch = [self.pair_loss, self.acc, self.acc01, self.score]

        if run_options is not None and run_metadata is not None:
            return sess.run(fetch, feed_dict, options=run_options, run_metadata=run_metadata)
        else:
            return sess.run(fetch, feed_dict)

    def run_score(self, sess, inp):
        q, qm, t, tm, kw, km, g = inp
        feed_dict = {self.query_sent_in: q, self.query_sent_mask: qm,
                     self.title_sent_in: t, self.title_sent_mask: tm,
                     self.keyword_in: kw, self.keyword_mask: km,
		     self.g: g,
                     self.is_training: False, self.keep_prob: 1.0,
                     self.pair_margin: args.pair_margin}
        fetch = [self.score]
        return sess.run(fetch, feed_dict)

    def model_graph(self):
        # shape variables for each batched input data
        batch_size = tf.shape(self.title_sent_mask)[0]  # (batch_size, n_samples,) dynamic shape
        cur_max_query_length = tf.shape(self.query_sent_mask)[-1]
        cur_max_title_length = tf.shape(self.title_sent_mask)[-1]
        n_samples = tf.shape(self.title_sent_mask)[1]

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
        query_seq1, _, _, _ = CuQRNNLayer(query_ebd, self.query_sent_mask, 284,
                              keep_prob=0.8, zoneout_keep_prob=0.9, is_training=self.is_training,
                              scope='QuerySeq1')
        query_seq1 = tf.concat([query_seq1, query_ebd], 2)  # 384
        query_seq2, _, _, _ = CuQRNNLayer(query_seq1, self.query_sent_mask, 384,
                              keep_prob=0.8, zoneout_keep_prob=0.9, is_training=self.is_training,
                              scope='QuerySeq2')
        _, query_final, _, _ = CuQRNNLayer(query_seq2, self.query_sent_mask, 384,
                               keep_prob=0.8, zoneout_keep_prob=0.9, is_training=self.is_training,
                               scope='QueryFinal')
        # shift 1 position to add a 'Null' in the query.
        query_mask_shifted = tf.concat([tf.ones((batch_size, 1)), self.query_sent_mask], 1)
        # "Null" word is fixed as a zero vector in original paper. 可以改成一个参数学习
        query_seq1 = tf.pad(query_seq1, [[0, 0], [1, 0], [0, 0]])
        query_seq2 = tf.pad(query_seq2, [[0, 0], [1, 0], [0, 0]])

        # ----------------  encode titles  ---------------------
        title_seq1, _, _, _ = CuQRNNLayer(title_ebd, title_sent_mask_re, 284,
                             keep_prob=0.8, zoneout_keep_prob=0.9, is_training=self.is_training,
                             scope='TitleSeq1')
        title_seq1 = tf.concat([title_seq1, title_ebd], 2)  # 384
        title_seq1, e1 = DotMatchAttention(query_seq1, query_mask_shifted,
                         tf.reshape(title_seq1,
                                   [batch_size, n_samples, cur_max_title_length, title_seq1.get_shape()[-1].value]),
                         self.title_sent_mask,
                         mode='linear', l2=1.0, smooth=1e-3,
                         keep_prob=1.0, is_training=self.is_training, scope='InterAtt1')
        title_seq1 = tf.reshape(title_seq1, [-1, cur_max_title_length, title_seq1.get_shape()[-1].value])

        title_seq2, _, _, _ = CuQRNNLayer(title_seq1, title_sent_mask_re, 384,
                                        initial_state=None,
                                        keep_prob=0.8, zoneout_keep_prob=0.9, is_training=self.is_training,
                                        scope='TitleSeq2')
        title_enc, e2 = DotMatchAttention(query_seq2, query_mask_shifted,
                        tf.reshape(title_seq2,
                                   [batch_size, n_samples, cur_max_title_length, title_seq2.get_shape()[-1].value]),
                        self.title_sent_mask,
                        mode='linear', l2=1.0, smooth=1e-3,
                        keep_prob=1.0, is_training=self.is_training, scope='InterAtt2')

        title_enc = tf.reshape(title_enc, [-1, cur_max_title_length, title_enc.get_shape()[-1].value])
        _, title_final, _, _ = CuQRNNLayer(title_enc, title_sent_mask_re, 384,
                               keep_prob=0.8, zoneout_keep_prob=0.9, is_training=self.is_training,
                               scope='TitleFinal')


        # ----------- calc score use regression or bi-linear  -------------
        self.score = BiLinearFeat(tf.tile(tf.reshape(query_final, [-1, 1, query_final.get_shape()[-1].value]),
                                           [1, n_samples, 1]),
                                           tf.reshape(title_final,
                                           [batch_size, n_samples, title_final.get_shape()[-1].value]))

        # --------------------------------  Loss and Training---------------------------------------------

        self.pair_loss, self.acc, self.acc01 = self.pair_loss_op(self.score, self.g, self.pair_margin)
        self.regu_loss = args.l2 * tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        # 加正则防止embedding过快增长
        # self.regu_loss += args.l2 * tf.reduce_sum(url_enc ** 2) / 2.0

        self.tol_cost = self.pair_loss + self.regu_loss
        self.train_op = self.async_training_op(self.tol_cost, tf.trainable_variables())
        self.tol_cost, self.pair_loss = e2, e1  # for log


