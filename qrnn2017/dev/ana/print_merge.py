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
    parser.add_argument('--ckpt', type=str, default="."
                        , help="absolute path")
    # todo add arguments for hidden code dict which is used online.
    parser.add_argument('--wdict', type=str, default=None
                        , help="Hidden code word dictionary which used online")
    parser.add_argument('--fake_input', type=str, default='fake_input.txt'
                        , help="Fake test files")
    parser.add_argument('--merge_data', type=str, default='merge_data'
                        , help="Merge features")
    command_line = parser.parse_args()

    args.checkpoint = command_line.ckpt
    if command_line.wdict is not None:
        args.wdict_path = command_line.wdict
    args.fake_input = command_line.fake_input
    args.merge_data = command_line.merge_data

def main():
    parseArgs(args)
    args.log_dir_path = args.output + os.path.sep + 'test_' + os.path.split(args.checkpoint)[1]
    args.log_prefix = args.a0_model_name + '_'
    if not os.path.exists(args.log_dir_path):
        os.makedirs(args.log_dir_path)

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

        print('Predicting ' + str(args.fake_input))
        fake_input_file = open(args.fake_input, 'rb')
        merge_data_file = open(args.merge_data, 'rb')
        new_feature_file = open(os.path.split(args.merge_data)[0] + os.path.sep +\
                                os.path.split(args.checkpoint)[1] + '.ltr', 'wb')
        pairs = fake_input_file.readlines()
        features = merge_data_file.readlines()
        n_test = len(pairs)
        n_batches = int(n_test/test_batchsize)
        if n_test % test_batchsize != 0:
          n_batches += 1

        for i in range(n_batches):
            data_proc_slice = model.data_proc(pairs[i*test_batchsize: (i+1)*test_batchsize], n_samples=1)
            score = model.run_score(sess, data_proc_slice)[0]
            for j in range(0, len(score)):
                line_num = i * test_batchsize + j
                if line_num >= n_test:
                    break
                feature_line = features[line_num]
                feature_text = feature_line.split('#')
                feature_text[0] += '6004:' + str(score[j][0]) + ' '
                new_line = '#'.join(feature_text)
                new_line = ''.join(new_line.split(','))  # special judgement
                new_feature_file.write(new_line)
        out_str = "Print log of " + str(args.fake_input) + " with checkpoint " + args.checkpoint + " finished!"
        args.message(out_str)
        fake_input_file.close()
        merge_data_file.close()
        new_feature_file.close()



