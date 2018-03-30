import sys
import os
import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter
import argparse
from dev.config.final_config import *
from dev.model.final_model import graph_moudle
import numpy as np
from datetime import datetime

def parseArgs(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default="."
                        , help="absolute path")
    parser.add_argument('--save_graph', type=bool, default=False
                        , help="save tf.graph")
    parser.add_argument('--only_trainable', type=bool, default=True
                        , help="save only trainable vars")
    # todo add arguments for hidden code dict which is used online.
    parser.add_argument('--extra_wdict', type=str, default=None
                        , help="Hidden code word dictionary which used online")
    parser.add_argument('--extra_testfnms', type=str, default='/data/webrank/convertDict/qt_test_hiddencode'
                        , help="Hidden code test file, which used online")
    command_line = parser.parse_args()

    args.checkpoint = command_line.ckpt_path
    args.save_graph = command_line.save_graph
    args.save_only_trainable = command_line.only_trainable
    args.extra_wdict = command_line.extra_wdict
    args.extra_testfnms = command_line.extra_testfnms

def main():
    parseArgs(args)
    args.log_dir_path = args.output + os.path.sep + 'test_' + os.path.split(args.checkpoint)[1]
    args.log_prefix = args.a0_model_name + '_'
    if not os.path.exists(args.log_dir_path):
        os.makedirs(args.log_dir_path)

    if args.save_graph:
        args.message("Resize batchsize to 1 for serving...")

    if args.extra_wdict is not None:
        args.wdict_path = args.extra_wdict
        args.testfnms = [args.extra_testfnms]
        args.message("Loading extra word dictionary from " + args.wdict_path)

    configproto = tf.ConfigProto()
    configproto.gpu_options.allow_growth = True
    configproto.allow_soft_placement = True
    args.batchsize = 400
    test_batchsize = args.batchsize
    #test_batchsize = 1

    with tf.Graph().as_default(), tf.Session(config=configproto) as sess:
        model = graph_moudle.Model()
        model.init_global_step()
        vt, vs, vo = model.model_setup()
        tf.initialize_all_variables().run()
        args.message('Loading trained model ' + str(args.checkpoint))
        model.saver.restore(sess, args.checkpoint)
        args.write_variables(vt)
        if args.save_graph:
            if args.save_only_trainable:
                exporter_saver = tf.train.Saver(var_list=tf.trainable_variables(), sharded=True)
            else:
                exporter_saver = tf.train.Saver(sharded=True)

            online_feed_dict = {'q': model.query_sent_in, 'qm': model.query_sent_mask,
                                't': model.title_sent_in, 'tm': model.title_sent_mask,
                                'is_training': model.is_training, 'keep_prob': model.keep_prob}
            online_fetch_dict = {'score': model.score}
            model_exporter = exporter.Exporter(exporter_saver)
            model_exporter.init(sess.graph.as_graph_def(),
                                named_graph_signatures={
                                    'inputs': exporter.generic_signature(online_feed_dict),
                                    'outputs': exporter.generic_signature(online_fetch_dict)})
            model_exporter.export(args.log_dir_path, tf.constant(0), sess)
            args.message("Successfully export graph to path:" + args.log_dir_path)
            #tf.train.write_graph(sess.graph_def, args.log_dir_path, 'graph.pbtxt', False)
            return

        args.write_args(args)
        for test_file in args.testfnms:
            print('Test ' + str(test_file))
            f = open(test_file)
            pairs = f.readlines()
            n_test = len(pairs)
            n_batches = int(n_test/test_batchsize)
            if n_test%test_batchsize != 0:
              n_batches += 1

            tpair_loss, tacc, tacc01 = 0.0, 0.0, 0.0
            for i in range(n_batches):
                q, qm, t, tm, g = model.data_proc(pairs[i*test_batchsize: (i+1)*test_batchsize])
                pair_loss, acc, acc01, score = \
                    model.run_epoch(sess, [q, qm, t, tm, g, False])
                tpair_loss, tacc, tacc01 = tpair_loss + pair_loss, tacc + acc, tacc01 + acc01
                out_str = "%f %f %f" % (pair_loss, acc, acc01)
                for j in range(0,len(score)):
                  if i*test_batchsize+j >= n_test:
                    break
                  pair_line = pairs[i*test_batchsize+j].rstrip()
                  pair_tokens = pair_line.split('\t')
                  if len(pair_tokens) == 3:
                    score_str1 = pair_tokens[0] + '\n' + pair_tokens[1] + '\n' + pair_tokens[2] + '\n' + str(score[j][0]) + '\t' + str(score[j][1]) + '\n\n'
                    #score_str2 = pair_tokens[0] + '\t' + pair_tokens[2] + '\t' + str(score[j][1]) + '\n'
                    #score_str = pair_line + '\t' + str(score[j][0])+'-'+str(score[j][1]) + '\n'
                    sys.stderr.write(score_str1)
                    #sys.stderr.write(score_str2)

                #print(out_str)

            out_str = "Test " + str(test_file) + " with checkpoint " + args.checkpoint
            args.message(out_str)
            n_batches = float(n_batches)
            out_str = "pair loss:%f acc:%f acc01:%f" \
                      % (tpair_loss / n_batches, tacc / n_batches, tacc01 / n_batches)
            args.message(out_str)
            f.close()



