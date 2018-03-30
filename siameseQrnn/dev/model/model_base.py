# -- coding:utf-8 --
import re
import sys
from tensorflow.core.protobuf import saver_pb2
import pickle as pkl
from dev.model.layers import *
from dev.config.final_config import *

class ModelBase(object):

    def extra_embedding_init(self):
        pass

    def __init__(self):
        # load word dict and word embeddings
        args.message('Loading word dictionary and word vectors.')
        if args.wdict_path is not None:
            self.wdict = pkl.load(open(args.wdict_path, 'rb'))
            self.word_set = set(self.wdict.keys())
        np.random.seed(1234567890)
        if args.embd_path is None:
            args.message('Build word vectors.')
            self.embedding_init = 5e-2 * \
                                  np.random.randn(args.vocab_size, args.embd_dims).astype(np.float32)
        else:
            embedding_np = pkl.load(open(args.embd_path, 'rb'))
            assert (embedding_np.shape[0] == args.vocab_size
                    and embedding_np.shape[1] == args.embd_dims)
            self.embedding_init = embedding_np
        np.random.seed()

        #self.extra_embedding_init()

    def word2id(self, aline):  # input is string like: 'w w w'
        id_list = []
        for n, w in enumerate(aline.strip().split()):  # ['w', 'w', 'w',...]
            if n >= args.max_sent_length:
                break
            if w not in self.word_set:
                id_list.append(0)
            else:
                id_list.append(self.wdict[w])
        return id_list, len(id_list)

    def data_proc(self, qt_text_list, title_steps=4, n_samples=args.n_samples):
        '''
        qt_text: = [query, title1, title2.....] 用\t 隔开
        query只有文本，词之间用\s 分割
        title包含 文本 \t docid, \t xxxx \t xxxx, 因此设置title_steps=4，
        title文本出现在 1 + title_steps * ns位置
        title的docid出现在 2 + title_steps * ns 位置
        '''

        q = np.zeros((args.batchsize, args.max_sent_length), dtype=np.int32)
        qm = np.zeros_like(q, dtype=np.float32)
        t = np.zeros((args.batchsize, n_samples, args.max_sent_length), dtype=np.int32)
        tm = np.zeros_like(t, dtype=np.float32)

        cur_max_q_len = 0
        cur_max_t_len = 0

        for bid, qt_text_line in enumerate(qt_text_list):
            qt_text = qt_text_line.strip().split('\t')  # ['w w w', 'w w w w', ....]
            assert (len(qt_text) >= args.n_samples + 1)

            q_id_list, q_id_len = self.word2id(qt_text[0])
            q[bid, :q_id_len] = q_id_list
            qm[bid, :q_id_len] = 1.0
            cur_max_q_len = max(cur_max_q_len, q_id_len)

            for ns in range(n_samples):
                t_id_list, t_id_len = self.word2id(qt_text[1 + ns * title_steps])
                t[bid, ns, :t_id_len] = t_id_list
                tm[bid, ns, :t_id_len] = 1.0
                cur_max_t_len = max(cur_max_t_len, t_id_len)

        q = q[:, :cur_max_q_len]
        qm = qm[:, :cur_max_q_len]
        t = t[:, :, :cur_max_t_len]
        tm = tm[:, :, :cur_max_t_len]

        return q, qm, t, tm

    def base_feed(self):
        # -------------------------- Inputs (tensorflow placeholder) --------------------
        self.label = tf.placeholder(tf.float32, [None], 'label')
        self.keep_prob = tf.placeholder(tf.float32, [], 'keep_prob')
        self.is_training = tf.placeholder(tf.bool, [], 'train_flag')
        self.query_sent_in = tf.placeholder(tf.int32, [None, None],
                                            name='query_sent')
        self.title_sent_in = tf.placeholder(tf.int32, [None, None, None],
                                        name='title_sent')
        self.query_sent_mask = tf.placeholder(tf.float32, [None, None],
                                          name='query_mask')
        self.title_sent_mask = tf.placeholder(tf.float32, [None, None, None],
                                          name='title_mask')
        self.title_doc_mask = tf.placeholder(tf.float32, [None, None],
                                              name='doc_mask')

        self.extra_score = tf.placeholder(tf.float32, [None, None, 2],
                                         name='extra_score')

    def base_fetch(self):
        # definition of output nodes
        self.train_op = 0
        self.acc = 0
        self.loss = 0
        self.regu_loss = 0
        self.tol_cost = 0
        self.score = 0
        self.r_merge = 0
        self.r_final = 0

    def extra_feed(self):  # Add or Modify the inputs
        pass

    def extra_fetch(self):  # Add or Modify the outputs
        pass

    def init_global_step(self):
        # Global steps for asynchronous distributed training.
        with tf.device('/cpu:0'):
            self.global_step = tf.get_variable('global_step', [],
                               initializer=tf.constant_initializer(0), trainable=False)
        self.base_feed()
        self.base_fetch()
        self.extra_feed()
        self.extra_fetch()

    def model_graph(self):
        pass

    def model_saver(self):
        # For distributed version, assign/initialization/saver is not allowed on each GPU
        self.saver = tf.train.Saver(max_to_keep=args.max_to_keep,
                                    write_version=saver_pb2.SaverDef.V1)
        # save all, including word embeddings

    def model_setup(self):
        self.model_graph()
        self.model_saver()
        return tf.trainable_variables(), tf.all_variables(), tf.get_collection(tf.GraphKeys.NO_CLIP)

    def async_training_op(self, cost, var_list, grads=None):

        if not args.train_embd:
            # -------------------- calc & clip all gradients--------------
            if grads is None:
                grads = tf.gradients(cost, var_list)
            grads = [tf.clip_by_value(g, -args.grad_clip, args.grad_clip) for g in grads]
            grads, _ = tf.clip_by_global_norm(grads, args.max_norm)

        else:
            # ------ clip some of the gradients, embeddings are excluded ------
            grads = tf.gradients(cost, var_list)
            noclip_varlist, withclip_varlist, noclip_grads, withclip_grads = [], [], [], []  # empty
            noclip_varlist = tf.get_collection(key=tf.GraphKeys.NO_CLIP)
            for i, var in enumerate(var_list):
                if var not in noclip_varlist:
                    withclip_varlist.append(var)
                    withclip_grads.append(grads[i])
                else:
                    noclip_grads.append(grads[i])

            # -------------------- clip all gradients except word embedding gradient --------------
            if args.use_grad_clip:
                withclip_grads = [tf.clip_by_value(g, -args.grad_clip, args.grad_clip) for g in withclip_grads]
                withclip_grads, _ = tf.clip_by_global_norm(withclip_grads, args.max_norm)

            grads = noclip_grads + withclip_grads  # keep the same order
            var_list = noclip_varlist + withclip_varlist

        optimizer = tf.train.RMSPropOptimizer(args.lr, name='AsyncTrain')

        return optimizer.apply_gradients(zip(grads, var_list), global_step=self.global_step)  # a tf.bool

    def pair_loss_op(self, score, g, margin=0.1):
        pass
    def mse_loss_op(self, score, label):
        '''
        calc losses for pair matching
        :param score (bs, ns) score for each pair. We assume that score[:,0] > score[:,1]
        marginal hinge loss.
        '''
        gap = score - label  # (bs,)
        l = tf.reduce_mean(tf.square(gap))
        correct_prediction = tf.equal(tf.round(score), label)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # acc = tf.reduce_mean(tf.equal(tf.round(score), label))
        return l, acc
    def run_epoch(self, sess, inp, istrn, run_options=None, run_metadata=None):
        pass

    def run_score(self, sess, inp):
        pass







