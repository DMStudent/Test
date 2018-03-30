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
    # todo add arguments for hidden code dict which is used online.
    parser.add_argument('--extra_wdict', type=str, default=None
                        , help="Hidden code word dictionary which used online")
    parser.add_argument('--extra_testfnms', type=str, default='/data/webrank/convertDict/qt_test_hiddencode'
                        , help="Hidden code test file, which used online")
    command_line = parser.parse_args()

    args.checkpoint = command_line.ckpt_path
    args.extra_wdict = command_line.extra_wdict
    args.extra_testfnms = command_line.extra_testfnms

def main():
    parseArgs(args)
    args.log_dir_path = args.output + os.path.sep + 'test_' + os.path.split(args.checkpoint)[1]
    args.log_prefix = args.a0_model_name + '_'
    if not os.path.exists(args.log_dir_path):
        os.makedirs(args.log_dir_path)

    if args.extra_wdict is not None:
        args.wdict_path = args.extra_wdict
        args.testfnms = [args.extra_testfnms]
        args.message("Loading extra word dictionary from " + args.wdict_path)

    configproto = tf.ConfigProto()
    configproto.gpu_options.allow_growth = True
    configproto.allow_soft_placement = True
    args.batchsize = 400
    test_batchsize = args.batchsize

    with tf.Graph().as_default(), tf.Session(config=configproto) as sess:
        model = graph_moudle.Model()
        model.init_global_step()
        vt, vs, vo = model.model_setup()
        tf.initialize_all_variables().run()
        args.message('Loading trained model ' + str(args.checkpoint))
        model.saver.restore(sess, args.checkpoint)
        args.write_variables(vt)

        args.write_args(args)
        for test_file in args.testfnms:
            print('Test ' + str(test_file))
            f = open(test_file)
            pairs = f.readlines()
            n_test = len(pairs)
            n_batches = int(n_test/test_batchsize)
            if n_test % test_batchsize != 0:
              n_batches += 1

            for i in range(n_batches):
                data_proc_slice = model.data_proc(pairs[i*test_batchsize: (i+1)*test_batchsize], n_samples=1)
                score = model.run_score(sess, data_proc_slice)[0]
                for j in range(0, len(score)):
                  if i*test_batchsize+j >= n_test:
                    break
                  pair_line = pairs[i*test_batchsize+j].rstrip()
                  pair_tokens = pair_line.split('\t')
                  if len(pair_tokens) <= 9:
                    score_str1 = pair_tokens[0] + '\t' + pair_tokens[1] + '\t' + pair_tokens[2] + '\t' + str(score[j][0]) + '\n'
                  else:
                    score_str1 = pair_tokens[0] + '\t' + pair_tokens[3] + '\t' + pair_tokens[4] + '\t' + str(score[j][0]) + '\t' + pair_tokens[-1] + '\n'
                  sys.stderr.write(score_str1)

                #print(out_str)

            out_str = "Print log of " + str(test_file) + " with checkpoint " + args.checkpoint + " finished!"
            args.message(out_str)
            f.close()



