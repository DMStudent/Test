import tensorflow as tf
from tensorflow.python.client import timeline
import time
from dev.config.final_config import *
from dev.model.final_model import graph_moudle
import numpy as np

args.embd_path = None
args.wdict_path = None
args.lr = 0.001
args.l2 = 1e-8
args.units = 384
args.train_embd = True
args.embd_dims = 100
args.vocab_size = 775000
args.batchsize = 400
args.num_layers = 1
args.use_intra = True
args.symmetric = False
args.pair_margin = 0.1
args.log_dir_path = 'results'
args.max_sent_length = 15

show_run_meta = True


def main():
    configproto = tf.ConfigProto()
    configproto.gpu_options.allow_growth = True
    configproto.log_device_placement = args.log_device
    configproto.allow_soft_placement = args.soft_placement
    configproto.inter_op_parallelism_threads = args.num_cores
    configproto.intra_op_parallelism_threads = args.num_cores
    with tf.Graph().as_default(), tf.Session(config=configproto) as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) if show_run_meta else None
        run_metadata = tf.RunMetadata() if show_run_meta else None

        model = graph_moudle.Model()
        model.init_global_step()

        vt, vs, vo = model.model_setup()
        tf.initialize_all_variables().run()
        cnt = 0
        for var in vt:
            cnt += 1
            str_line = str(cnt) + '. ' + str(var.name) + ': ' + str(var.get_shape())
            print(str_line)
        ssll = raw_input('aaaaa')
        np.random.seed(1234567890)
        qs = np.random.randint(0, args.vocab_size,
                               [10, args.batchsize, args.max_sent_length])
        qsm = np.ones_like(qs, dtype=np.float32)
        #qsm[:, :, -1:] = 0
        ts = np.random.randint(0, args.vocab_size, [10, args.batchsize, 2, args.max_sent_length])
        tsm = np.ones_like(ts, dtype=np.float32)
	url = np.ones([10, args.batchsize, 2], dtype=np.int32)
	kw = np.ones([10, args.batchsize, 2, 10], dtype=np.int32)
	kwm = np.ones([10, args.batchsize, 2, 10], dtype=np.float32)
        g = np.random.randint(1, 3, (10, args.batchsize))

        for i in range(400):
            bs = i % 10
            if bs == 0:
                idx = np.random.shuffle(np.arange(10 * args.batchsize))
                qs = np.reshape(qs, [10 * args.batchsize, -1])[idx]
                ts = np.reshape(ts, [10 * args.batchsize, -1])[idx]
                g = np.reshape(g, [10 * args.batchsize, -1])[idx]
                qs = np.reshape(qs, qsm.shape)
                ts = np.reshape(ts, tsm.shape)

            stime = time.time()
            loss, regu_loss = 0, 0
            
            step, loss, pair_loss, regu_loss, acc, acc01, score, _ = \
                model.run_epoch(sess, [qs[bs], qsm[bs], ts[bs], tsm[bs], url[bs], kw[bs], kwm[bs]], True,
                                run_options=run_options, run_metadata=run_metadata)
            '''
            pair_loss, acc, acc01, score = \
                model.run_epoch(sess,
                                [qs[bs], qsm[bs], ts[bs], tsm[bs], np.ones([args.batchsize, 100], dtype=np.int32)],
                                False,
                                run_options=run_options, run_metadata=run_metadata)
	    '''
            print(loss, pair_loss, regu_loss, acc, acc01, time.time() - stime)
            #print(score[0, :], g[bs][0])
	    
            if show_run_meta:
                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format(show_memory=True)
                with open(args.log_dir_path + '/timeline.json', 'w') as f:
                    f.write(ctf)
                    # ss = input("Press enter key to continue...")
                    # model.saver.save(sess, 'test', write_meta_graph=False)
