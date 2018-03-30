# -- coding:utf-8 --

# 打印模型的参数
import sys
import os
import tensorflow as tf
from dev.config.final_config import *
from dev.model.final_model import graph_moudle
from tensorflow.python.framework import ops


def main():
    checkName = "/search/odin/data/wangyuan/pycharmProjects/siameseQrnn/model/20180131/siamQrnnCkpt_1"
    configproto = tf.ConfigProto()
    configproto.gpu_options.allow_growth = True
    configproto.allow_soft_placement = True
    ops.reset_default_graph()
    with tf.Graph().as_default(), tf.Session(config=configproto) as sess:
        model = graph_moudle.Model()
        model.init_global_step()
        vt, vs, vo = model.model_setup()
        tf.global_variables_initializer().run()
        args.message('Loading trained model ' + checkName)
        model.saver.restore(sess, checkName)
        variable_names = tf.trainable_variables()
        values = sess.run(variable_names)
        for k, v in zip(variable_names, values):
            print("Variable: ", k)
            print("Shape: ", v.shape)
            print(v)


if __name__ == "__main__":
    main()