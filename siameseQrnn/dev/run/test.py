# -- coding:utf-8 --
import sys
import os
import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter
import argparse
from dev.config.final_config import *
from dev.model.final_model import graph_moudle
import numpy as np
from datetime import datetime
from operator import itemgetter
from itertools import groupby
from tensorflow.python.framework import ops

def read_input(file):
    """Read input and split."""
    data = []
    for line in file:
        data.append(line)
    return data


tf
def parseArgs(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default="./model/20180131/siamQrnnCkpt_2"
                        , help="absolute path")
    parser.add_argument('--save_graph', type=bool, default=False
                        , help="save tf.graph")
    parser.add_argument('--extra_testfnms', type=str, default='/search/odin/data/wangyuan/data-wenti/test'
                        , help="Hidden code test file, which used online")
    parser.add_argument('--predit_output', type=str, default='/search/odin/data/wangyuan/data-wenti/test_output2/output'
                        , help="predit output file path")
    command_line = parser.parse_args()

    args.checkpoint = command_line.ckpt_path
    # args.save_graph = command_line.save_graph
    # args.save_only_trainable = command_line.only_trainable
    # args.extra_wdict = command_line.extra_wdict
    args.extra_testfnms = command_line.extra_testfnms
    args.predit_output = command_line.predit_output

def main():
    parseArgs(args)
    args.log_dir_path = args.output + os.path.sep + 'test_' + os.path.split(args.checkpoint)[1]
    args.log_prefix = args.a0_model_name + '_'
    if not os.path.exists(args.log_dir_path):
        os.makedirs(args.log_dir_path)

    # if args.save_graph:
    #     args.message("Resize batchsize to 1 for serving...")
    #
    # if args.extra_wdict is not None:
    #     args.wdict_path = args.extra_wdict
    #     args.testfnms = [args.extra_testfnms]
    #     args.message("Loading extra word dictionary from " + args.wdict_path)
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 占用GPU90%的显存

    configproto = tf.ConfigProto()
    configproto.gpu_options.allow_growth = True
    configproto.allow_soft_placement = True
    test_batchsize = args.batchsize
    ops.reset_default_graph()
    with tf.Graph().as_default(), tf.Session(config=configproto) as sess:
        model = graph_moudle.Model()
        model.init_global_step()
        vt, vs, vo = model.model_setup()
        tf.global_variables_initializer().run()
        args.message('Loading trained model ' + str(args.checkpoint))
        model.saver.restore(sess, args.checkpoint)
        args.write_variables(vt)
        args.write_args(args)

        fw = open(args.predit_output,"w")

        for test_file in args.testfnms:
            print('Test ' + str(test_file))
            f = open(test_file)
            pairs = read_input(f.readlines())
            n_test = len(pairs)
            n_batches = int(n_test / test_batchsize)
            if n_test % test_batchsize != 0:
                n_batches += 1
            for i in range(n_batches):
                result = ""
                data_proc_slice = model.data_proc(pairs[i * test_batchsize: (i + 1) * test_batchsize])

                label, q, qm, t, tm, dm, _ = data_proc_slice

                out_str = "q: %f - %f - %f - %f - %f" % (q[0][0], q[0][1], q[0][2], q[0][3], q[0][4])
                args.message(out_str, True)
                out_str = "qm: %f - %f - %f - %f - %f" % (qm[0][0], qm[0][1], qm[0][2], qm[0][3], qm[0][4])
                args.message(out_str, True)
                out_str = "t: %f - %f - %f - %f - %f" % (t[0][0][0], t[0][0][1], t[0][0][2], t[0][0][3], t[0][0][4])
                args.message(out_str, True)
                out_str = "qm: %f - %f - %f - %f - %f" % (tm[0][0][0], tm[0][0][1], tm[0][0][2], tm[0][0][3], tm[0][0][4])
                args.message(out_str, True)
                out_str = "label: %f - %f - %f - %f - %f" % (label[0], label[1], label[2], label[3], label[4])
                args.message(out_str, True)

                acc, score = \
                    model.run_epoch(sess, data_proc_slice, False)
                print "acc: " + str(acc)

                for j in range(0, min(len(pairs)-(i*test_batchsize), test_batchsize)):

                    pair_line = pairs[i * test_batchsize + j].rstrip()
                    result = result + str(round(score[j],2)) + '\t' + pair_line + '\n'
                fw.write(result)

        fw.close()


