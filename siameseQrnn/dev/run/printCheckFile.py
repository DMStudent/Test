# -- coding:utf-8 --
import sys
import os
import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter
import argparse
from dev.config.final_config import *
from dev.model.final_model import graph_moudle
from tensorflow.python.framework import ops


def parseArgs(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default="/search/odin/data/wangyuan/pycharmProjects/siameseQrnn/model/20180131/siamQrnnCkpt_5"
                        , help="absolute path")
def main():
    parseArgs(args)
    configproto = tf.ConfigProto()
    configproto.gpu_options.allow_growth = True
    configproto.allow_soft_placement = True
    ops.reset_default_graph()
    with tf.Graph().as_default(), tf.Session(config=configproto) as sess:
        model = graph_moudle.Model()
        model.init_global_step()
        vt, vs, vo = model.model_setup()
        tf.global_variables_initializer().run()
        args.message('Loading trained model ' + str(args.checkpoint))
        model.saver.restore(sess, args.checkpoint)
        for var in tf.trainable_varisbles():
            print var.name
        args.write_variables(vt)
        args.write_args(args)
