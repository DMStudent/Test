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
def read_input(file):
    """Read input and split."""
    for line in file:
        yield line.rstrip().split("\t")



tf
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
    parser.add_argument('--predit_output', type=str, default='./output.txt'
                        , help="predit output file path")
    command_line = parser.parse_args()

    args.checkpoint = command_line.ckpt_path
    args.save_graph = command_line.save_graph
    args.save_only_trainable = command_line.only_trainable
    args.extra_wdict = command_line.extra_wdict
    args.extra_testfnms = command_line.extra_testfnms
    args.predit_output = command_line.predit_output

def main():
    parseArgs(args)
    args.log_dir_path = args.output + os.path.sep + 'test_' + os.path.split(args.checkpoint)[1]
    args.log_prefix = args.a0_model_name + '_'
    if not os.path.exists(args.log_dir_path):
        os.makedirs(args.log_dir_path)

    if args.save_graph is not None:
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

        fw = open(args.predit_output,"w")
        fw2 = open("./data/detail.txt", "w")

        for test_file in args.testfnms:
            print('Test ' + str(test_file))
            f = open(test_file)
            data = read_input(f.readlines())

            for key, kviter in groupby(data, itemgetter(2)):
                pairs = []
                itemNum = 0
                for k in kviter:
                    #if(len(k[0])>15 and len(k[1])>15):
                    pairs.append("\t".join(k))
                    itemNum = itemNum + 1
                    if itemNum>20:
                        break
                if(itemNum<5):
                    continue
                n_test = len(pairs)
                n_batches = int(n_test / test_batchsize)
                if n_test % test_batchsize != 0:
                    n_batches += 1

                tpair_loss, tacc, tacc01 = 0.0, 0.0, 0.0
                score_sum = 0
                result = ""
                detail = ""
                for i in range(n_batches):
                    q, qm, t, tm, g = model.data_proc(pairs[i * test_batchsize: (i + 1) * test_batchsize])
                    pair_loss, acc, acc01, score = \
                        model.run_epoch(sess, [q, qm, t, tm, g, False])
                    for j in range(0, itemNum):
                        if i * test_batchsize + j >= n_test:
                            break
                        pair_line = pairs[i * test_batchsize + j].rstrip()
                        pair_tokens = pair_line.split('\t')
                        if len(pair_tokens) > 2:
                            score_sum = score_sum + score[j][0]
                            score_str1 = key + '\t' + pair_tokens[0] + '\t' + pair_tokens[1] + '\t' + str(score[j][0]) + '\n'
                            detail = detail + score_str1
                            # fw2.write(score_str1)

                    score_avg = score_sum/itemNum
                    result = result + key + '\t' + str(score_avg) + '\n'
                fw2.write(detail)
                fw.write(result)
            f.close()
        fw.close()
        fw2.close()


