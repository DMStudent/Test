import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import numpy as np
import pickle
from dev.config.final_config import *
from dev.model.layers import *
from dev.model.AsyncOptimizer import *

class ModelBase(object):
    def __init__(self):
        # word dict initialization.
        args.message('Loading word dictionary and word vectors.')
        if args.wdict_path is not None:
            self.wdict = pickle.load(open(args.wdict_path, 'rb'))
            self.word_set = set(self.wdict.keys())

        # word embedding initialization.
        np.random.seed(1234567890)
        if args.embd_path is None:
            self.embedding_init = 5e-2 * \
                                  np.random.randn(args.vocab_size, args.embd_dims).astype(np.float32)
        else:
            embedding_np = pickle.load(open(args.embd_path, 'rb'))
            assert(embedding_np.shape[0] == args.vocab_size
                    and embedding_np.shape[1] == args.embd_dims)
            self.embedding_init = embedding_np
        np.random.seed()

    def data_proc(self, qt_text_list):
        '''
        :param qt_text: ['w w w\tw w w w\tw w w ...',...,.]  = [query, title1, title2.....]
        '''
        def word2id(aline):  # input is string like: 'w w w'
            id_list = []
            for n, w in enumerate(aline.strip().split()):  # ['w', 'w', 'w',...]
                if n >= args.max_sent_length:
                    break
                if w not in self.word_set:
                    id_list.append(0)
                else:
                    id_list.append(self.wdict[w])
            return id_list, len(id_list)

        q = np.zeros((args.batchsize, args.max_sent_length), dtype=np.int32)
        qm = np.zeros_like(q, dtype=np.float32)
        t = np.zeros((args.batchsize, args.n_samples, args.max_sent_length), dtype=np.int32)
        tm = np.zeros_like(t, dtype=np.float32)
        g = np.ones((args.batchsize,), dtype=np.float32)

        for bid, qt_text_line in enumerate(qt_text_list):
            qt_text = qt_text_line.strip().split('\t')  #['w w w', 'w w w w', ....]
            #assert (len(qt_text) >= args.n_samples + 1)

            q_id_list, q_id_len = word2id(qt_text[0])
            q[bid, :q_id_len] = q_id_list
            qm[bid, :q_id_len] = 1.0

            for ns in range(args.n_samples):
                #t_id_list, t_id_len = word2id(qt_text[1+ns])
                t_id_list, t_id_len = word2id(qt_text[1+4*ns])
                #t_id_list, t_id_len = word2id(qt_text[1])
                t[bid, ns, :t_id_len] = t_id_list
                tm[bid, ns, :t_id_len] = 1.0
            if len(qt_text) == args.n_samples + 2 and args.use_g:
                g[bid] *= np.float32(qt_text[-1].strip())

        return q, qm, t, tm, g


    def init_global_step(self):
        # Global steps for asynchronous distributed training.
        with tf.device('/cpu:0'):
            self.global_step = tf.get_variable('global_step', [],
                                               initializer=tf.constant_initializer(0), trainable=False)

        # -------------------------- Inputs (tensorflow placeholder) --------------------
        self.keep_prob = tf.placeholder(tf.float32, [], 'keep_prob')
        self.is_training = tf.placeholder(tf.bool, [], 'train_flag')
        self.pair_margin = tf.placeholder(tf.float32, [], 'pair_margin')
        self.gap_index = tf.placeholder(tf.float32, [None], 'gap_index')

        self.query_sent_in = tf.placeholder(tf.int32, [None, args.max_sent_length],
                                            name='query_sent')
        # titles: (bs, n_samples, sl), n_samples is dynamic for train or test.
        self.title_sent_in = tf.placeholder(tf.int32, [None, None, args.max_sent_length],
                                            name='title_sent')
        self.query_sent_mask = tf.placeholder(tf.float32, [None, args.max_sent_length],
                                              name='query_mask')
        self.title_sent_mask = tf.placeholder(tf.float32, [None, None, args.max_sent_length],
                                              name='title_mask')

        # -------------------------- Outpus (pass) --------------------------------------
        self.tol_cost, self.pair_loss, self.regu_loss,\
        self.acc, self.acc01, self.att, self.score = 0, 0, 0, 0, 0, 0, 0
        self.train_op = False

    def async_training_op(self, cost, var_list, grads=None):
        '''
        2016-11-15, Haoze Sun
        0. gradient for word embeddings is a tf.sparse_tensor
        1. clip_norm and clip by value operations do not support sparse tensor
        2. When using Adam, it seems word embedding is not trained on GPU.
           Actually, CPU is not capable to execute 8-worker word embedding Adam updating, which
            cause the GPU usage-->0% and the train is very slow.
        3. We employ AdaGrad instead, if args.train_embd == True.
           Gradient clip is barely used in AdaGrad.
           Other optimizator like RMSProp, Momentum have not tested.

        ref: http://stackoverflow.com/questions/40621240/gpu-cpu-tensorflow-training
        ref: http://stackoverflow.com/questions/36498127/
                  how-to-effectively-apply-gradient-clipping-in-tensor-flow
        ref: http://stackoverflow.com/questions/35828037/
                  training-a-cnn-with-pre-trained-word-embeddings-is-very-slow-tensorflow
        '''

        # ------------- calc gradients --------------------------
        if grads is None:
            grads = tf.gradients(cost, var_list)

        # ------------- Optimization -----------------------
        # 0. global step used for asynchronous distributed training.
        # 1. Adam (default lr 0.0004 for 8 GPUs, 300 batchsize) if args.train_embd == False,
        #    apply gradient clip operations (default 10, 100)
        # 2. Adagrad (default 0.01~0.03? 1e-8? for 8 GPUs, 300 batchsize) if train embedding,
        #    no gradient clip.
        #    However, Adagrad is not suitable for large datasets.
        # 3. Momentum (default 0.001)
        # 4. RMSProp/Adadelta (default 0.001) is also OK......

        if not args.train_embd:
            # -------------------- clip all gradients--------------
            grads = [tf.clip_by_value(g, -args.grad_clip, args.grad_clip) for g in grads]
            grads, _ = tf.clip_by_global_norm(grads, args.max_norm)

            optimizer = tf.train.AdamOptimizer(args.lr)
        else:
            # !!! We assume the word embedding has index 0 in var_list
            assert (var_list[0].get_shape()[0].value == args.vocab_size and
                    var_list[0].get_shape()[1].value == args.embd_dims)

            word_grad, other_grads = [grads[0]], grads[1:]
            # -------------------- clip all gradients except word embedding gradient --------------
            other_grads = [tf.clip_by_value(g, -args.grad_clip, args.grad_clip) for g in other_grads]
            other_grads, _ = tf.clip_by_global_norm(other_grads, args.max_norm)
            grads = word_grad + other_grads  # keep the order

            optimizer = tf.train.RMSPropOptimizer(args.lr)
            #optimizer = AsyncAdamOptimizer(args.lr)

        return optimizer.apply_gradients(zip(grads, var_list), global_step=self.global_step)  # a tf.bool

    def pair_loss_op(self, score, margin=0.2, gap_index=None):
        '''
        calc losses for pair matching
        :param score (bs, ns) score for each pair. We assume that score[:,0] > score[:,1]
        marginal hinge loss.
        2016-12-3
        log_gap: useless. Can't find a suitable priority, use linear function for simply.
        2016-12-26
        Gap index: a weight for different pars, ref:
        http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7317747&tag=1
        Weakly Semi-Supervised Deep Learning for Multi-Label Image Annotation
        '''
        gap = score[:, 1] - score[:, 0]  # (bs,)
        if gap_index is None:
            gap_index = tf.ones_like(gap, dtype=tf.float32)
        #modified by sam
        l = tf.reduce_mean(tf.nn.relu(gap + margin) * gap_index)
        #l = tf.reduce_mean(tf.nn.softplus(4.0*(gap + margin)) * gap_index)
        #l = tf.reduce_mean(tf.log(1.0 + tf.exp(10.0 * gap)))
        acc = tf.reduce_mean(tf.nn.relu(tf.sign(-gap)))
        acc01 = tf.reduce_mean(tf.nn.relu(tf.sign(-gap-0.1)))
        return l, acc, acc01

    def model_setup(self):
        pass

    def init_single(self):
        if args.distributed == True:
            return

    def run_epoch(self, sess, inp, run_options=None, run_metadata=None):
        if len(inp) == 5:
            q, qm, t, tm, istrn = inp  # or q, qm, t, tm, g, istrn
            g = np.ones([args.batchsize], dtype=np.float32)
        elif len(inp) == 6:
            q, qm, t, tm, g, istrn = inp  # or q, qm, t, tm, g, istrn
        else:
            raise ValueError('Model must have 5 or 6 inputs')

        if istrn:
            feed_dict = {self.query_sent_in: q, self.query_sent_mask: qm,
                         self.title_sent_in: t, self.title_sent_mask: tm,
                         self.gap_index: g,
                         self.is_training: True, self.keep_prob: args.keep_prob,
                         self.pair_margin: args.pair_margin}
            fetch = [self.global_step, self.tol_cost, self.pair_loss, self.regu_loss,
                     self.acc, self.acc01, self.score, self.train_op]
        else:
            feed_dict = {self.query_sent_in: q, self.query_sent_mask: qm,
                         self.title_sent_in: t, self.title_sent_mask: tm,
                         self.gap_index: g,
                         self.is_training: False, self.keep_prob: 1.0,
                         self.pair_margin: args.pair_margin}
            fetch = [self.pair_loss, self.acc, self.acc01, self.score]

        if run_options is not None and run_metadata is not None:
            return sess.run(fetch, feed_dict, options=run_options, run_metadata=run_metadata)
        else:
            return sess.run(fetch, feed_dict)

    def run_epoch_att(self, sess, inp, run_options=None, run_metadata=None):
        if len(inp) == 5:
            q, qm, t, tm, istrn = inp  # or q, qm, t, tm, g, istrn
            g = np.ones([args.batchsize], dtype=np.float32)
        elif len(inp) == 6:
            q, qm, t, tm, g, istrn = inp  # or q, qm, t, tm, g, istrn
        else:
            raise ValueError('Model must have 5 or 6 inputs')

        if istrn:
            feed_dict = {self.query_sent_in: q, self.query_sent_mask: qm,
                         self.title_sent_in: t, self.title_sent_mask: tm,
                         self.gap_index: g,
                         self.is_training: True, self.keep_prob: args.keep_prob,
                         self.pair_margin: args.pair_margin}
            fetch = [self.global_step, self.tol_cost, self.pair_loss, self.regu_loss,
                     self.acc, self.acc01, self.score, self.train_op]
        else:
            feed_dict = {self.query_sent_in: q, self.query_sent_mask: qm,
                         self.title_sent_in: t, self.title_sent_mask: tm,
                         self.gap_index: g,
                         self.is_training: False, self.keep_prob: 1.0,
                         self.pair_margin: args.pair_margin}
            fetch = [self.pair_loss, self.acc, self.acc01, self.att, self.score]

        if run_options is not None and run_metadata is not None:
            return sess.run(fetch, feed_dict, options=run_options, run_metadata=run_metadata)
        else:
            return sess.run(fetch, feed_dict)
