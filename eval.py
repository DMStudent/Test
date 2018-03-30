#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import numpy as np
import time
import datetime
import data_helpers
import codecs
from text_cnn import TextCNN
import data_reader
import pickle
import pandas as pd
import types
# Parameters
# ==================================================
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 1000, "Batch Size (default: 64)")
#如果模型位置移动，请记得修改模型里的checkpoint文件
tf.flags.DEFINE_string("checkpoint_dir", "./runs/seqing_final/checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("data_path", "./data/jinxin600w/", "data_path")
#tf.flags.DEFINE_string("data_path", ".", "data_path")
tf.flags.DEFINE_string("eval_file", "seqing_1k_label_1", "eval_file")
tf.flags.DEFINE_string("load_word2vec", "true", "Whether to load existing word2vec dictionary file.")
tf.flags.DEFINE_string("load_random", "false", "Whether to load existing word2vec dictionary file.")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

#for me
# FLAGS.batch_size = 2048
# FLAGS.checkpoint_dir = "./runs/checkpoints/"
# FLAGS.eval_train = True
# FLAGS.allow_soft_placement = True
# FLAGS.log_device_placement = False
# FLAGS.data_path = "./data/"
# FLAGS.load_word2vec = True
# FLAGS.load_random = False


print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
start = time.clock()

def load_word2id_embeddings(path1, path2):
    return pickle.load(open(path1, "r")), pickle.load(open(path2, "r"))

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    if FLAGS.load_word2vec=="true":
        # word2id, word_embeddings = data_reader.load_word2vec("./vector.skip.win2.100.float.for_python") #vector.skip.win2.100.float.for_python
        word2id, word_embeddings = load_word2id_embeddings("word2id", "word_embeddings")
        vocab_size = len(word2id) + 2

FLAGS.eval_file = FLAGS.data_path + FLAGS.eval_file
test_idsList, vocabulary ,max_sentence_length = data_reader.get_data_by_word2vec(word2id,
       data_file=FLAGS.eval_file, mode="predict")
print type(FLAGS.eval_file)
#print(len(test_idsList))
#for x in test_idsList: print x,
print type(test_idsList)
print test_idsList
print type(test_idsList[0][0])
x_test = np.array(test_idsList)
print '******'
print type(x_test)

# Map data into vocabulary
# if FLAGS.load_random:
#     vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
#     vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
#     x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")
print(FLAGS.checkpoint_dir)
# Evaluation
# ==================================================
#get the latest model
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    with tf.device("cpu:0"):
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables

            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
    #         saver = tf.train.Saver()
            saver.restore(sess, checkpoint_file)
            #for v in tf.global_variables():
             # print v.name
            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            # probs = graph.get_operation_by_name("output/probs").outputs[0]
            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("textCnn/output/predictions").outputs[0]
            # probs = graph.get_operation_by_name("output/probs").outputs[0]


            # Collect the predictions here
            print ("predicting .....")
            predictions = sess.run( predictions, {input_x:x_test, dropout_keep_prob: 1.0})
            print (predictions)
            print (len(predictions))
            print (type(predictions))
            print (predictions[0])
            print (len(predictions[0]))
            print (type(predictions[0]))
            # all_predictions = np.concatenate([all_predictions, batch_predictions])
            df = pd.read_csv(FLAGS.eval_file, names=["texts"], header=None)
            print ("writing out data>>>>>")
            # df['probs'] = probs
            df['predictions'] = predictions
            #只输出预测为1的case
            #df = df[df['predictions']==1]
            df.to_csv("predict_seqing.txt", header=None, index=False, sep="\t")
# Print accuracy if y_test is defined
# if y_test is not None:
#     correct_predictions = float(sum(all_predictions == y_test))
#     print("Total number of test examples: {}".format(len(y_test)))
#     print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
