#coding:utf-8
#! /usr/bin/env python
#单机
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import data_reader
import codecs
import pickle
import subprocess

# subprocess.call("export PATH=$PATH:/usr/local/cuda/bin", shell=True)
# subprocess.call("export LD_LIBRARY_PATH=/usr/local/cuda/lib64", shell=True)


# Parameters
# ==================================================

# Model Hyperparameters
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tf.flags.DEFINE_boolean("first_use", False, "First use")
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 3, "L2 regularizaion lambda (default: 0.0)")

tf.flags.DEFINE_string("model_save_path", "latest", "model save path")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_float("gpu_fraction", 0.5, "Memory gpu use")
tf.flags.DEFINE_boolean("allow_growth", True, "allow_growth")

#input data
tf.flags.DEFINE_string("data_path", "./data/", "data_path")
tf.flags.DEFINE_string("load_word2vec", "true", "Whether to load existing word2vec dictionary file.")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================

# Load data
print("Loading data...")


def save_word2id_embedding(word2id, path1, word_embeddings, path2):
    pickle.dump(word2id, open(path1, "w"))
    pickle.dump(word_embeddings, open(path2, "w"))

def load_word2id_embeddings(path1, path2):
    return pickle.load(open(path1, "r")), pickle.load(open(path2, "r"))

if FLAGS.load_word2vec == "true":
    if FLAGS.first_use:
        word2id, word_embeddings = data_reader.load_word2vec_utf8("word_embedding.txt.utf8")
        vocabulary_size = len(word2id) + 2
        save_word2id_embedding(word2id, "word2id", word_embeddings, "word_embeddings")
    else:
        word2id, word_embeddings = load_word2id_embeddings("word2id", "word_embeddings")
        vocabulary_size = len(word2id) + 2
    print("finish load embedding")

(train_idsList, train_lList), (test_idsList, test_lList), vocabulary ,max_sentence_length = \
    data_reader.get_data_by_word2vec(word2id, FLAGS.data_path)
print("Vocabulary Size: {:d}".format(vocabulary_size))
# for i in range(10):
#     print (train_idsList[i])
x_train = np.array(train_idsList)
y_train = train_lList

np.random.seed(7000000)
shuffle_indices = np.random.permutation(np.arange(x_train.shape[0]))

x_train = x_train[shuffle_indices]
y_train = y_train[shuffle_indices]

x_test = np.array(test_idsList)
y_test = test_lList

#过滤train和valid0
idx = np.sum(x_train!=vocabulary_size-1, axis=1)!=0
x_train = x_train[idx]
y_train = y_train[idx]

x_valid = x_test[:60000]
y_valid = y_test[:60000]

idx = np.sum(x_valid!=vocabulary_size-1, axis=1)!=0
x_valid = x_valid[idx]
y_valid = y_valid[idx]

#输出信息
print("x_train.shape", x_train.shape)
print("x_valid.shape", x_valid.shape)

#watch list
(test, label), _, _ = \
        data_reader.get_data_by_word2vec(word2id, FLAGS.data_path, mode='test', TEST_FILE='watch_test.txt')
watch_x= np.array(test)
watch_y = label

# Training
# ==================================================
fo = codecs.open("./model/pr_re", "w", "utf-8")
with tf.Graph().as_default():
    with tf.device("cpu:0"):
        session_conf = tf.ConfigProto(
          gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction,
                                    allow_growth=FLAGS.allow_growth),
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement
        )
        sess = tf.Session(config=session_conf)
        if FLAGS.load_word2vec == "true":
             cnn = TextCNN(
                sequence_length=max_sentence_length,
                num_classes=2,
                vocab_size=vocabulary_size,
                embedding_size=100,
                wordembedding=word_embeddings,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                pos_weight=1
             )
    #              sess.run(cnn.embedding.assign(word_embeddings))
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        # optimizer = tf.train.GradientDescentOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        # timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs/"+FLAGS.model_save_path))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        # Write vocabulary
    #         vocab_processor.save(os.path.join(out_dir, "vocab"))
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        #一个batch的训练
        #监控embeding的变化
        def train_step(x_batch, y_batch):
            global watch_x
            global watch_y
            global t
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy, prediction, target_y, embedded_chars = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.predictions, cnn.target_y, cnn.embedded_chars],
                feed_dict)
            if step % 20 == 0:
                true_positive = 0.0
                false_negative = 0.0
                false_positive = 0.0
                for i in range(len(prediction)):
                    if prediction[i] == 1 and target_y[i] == 1:
                        true_positive += 1
                    if prediction[i] == 0 and target_y[i] == 1:
                        false_negative += 1
                    if prediction[i] == 1 and target_y[i] == 0:
                        false_positive += 1
                # print("TP {}, FP {}, FN {}".
                #       format(true_positive, false_positive, false_negative))

                recal =  true_positive / (true_positive + false_negative)
                precision = true_positive / (true_positive + false_positive)
                time_str = datetime.datetime.now().isoformat()

                # 输出变化，如果embedding在变化说明在训练
                print("{}: step {}, loss {:g}, acc {:g}, pre {:g}, rec {:g}".format(time_str, step, loss,
                                                                                accuracy, precision, recal))

                feed_dict = {
                    cnn.input_x: watch_x,
                    cnn.input_y: watch_y,
                    cnn.dropout_keep_prob: 1
                }
                watch_prob, target_y = sess.run([cnn.probs, cnn.target_y], feed_dict)
                print(watch_prob)
            train_summary_writer.add_summary(summaries, step)
        #验证集的测试
        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy1, probs, target_y = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.probs, cnn.target_y],
                feed_dict)

            return probs, target_y

        # Generate batches
        batches = data_helpers.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs, mode="train"
                                          , train_batch_pos_size=128)
            # Training loop. For each batch...
        F1_s = 0.0
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                test_batches = data_helpers.batch_iter(list(zip(x_valid, y_valid)),
                                                       FLAGS.batch_size, 1, mode="test")
                probs_lst = []
                target_y_lst = []
                for batch in test_batches:
                    x_batch, y_batch = zip(*batch)
                    probs, target_y = dev_step(x_batch, y_batch, writer=dev_summary_writer)
                    probs_lst.append(probs)
                    target_y_lst.append(target_y)
                probs = np.vstack(probs_lst)
                target_y = np.hstack(target_y_lst)


                for thresold in [0.5, 0.6]:


                    thresold_prediction = np.array(probs[:,1] > thresold, int)
                    accuracy = np.mean(np.array(thresold_prediction == target_y, int))
                    true_positive = 0.0
                    false_negative = 0.0
                    false_positive = 0.0
                    for i in range(len(thresold_prediction)):
                        if thresold_prediction[i] == 1 and target_y[i] == 1:
                            true_positive += 1
                        if thresold_prediction[i] == 0 and target_y[i] == 1:
                            false_negative += 1
                        if thresold_prediction[i] == 1 and target_y[i] == 0:
                            false_positive += 1

                    recal = true_positive / (true_positive + false_negative)
                    precision = true_positive / (true_positive + false_positive)
                    F1 = (2 * recal * accuracy) / (recal + accuracy)
                # fo.write(str(accuracy))
                # fo.write("\t")
                # fo.write(str(recal))
                # fo.write("\n")

                    print("thresold:{:g}, acc {:g}, pre {:g},  rec {:g}, F1, {:g}, prediction, {:g}, target, {:g}"
                          .format(thresold, accuracy, precision, recal, F1, np.sum(thresold_prediction),
                                  np.sum(target_y)))
                    print("TP {}, FP {}, FN {}".
                          format(true_positive, false_positive, false_negative))
                    # if(thresold==0.5):
                    #     # 在测试集的F1大于现有的，则保存模型
                    #     if F1 > F1_s:
                    #         F1_s = F1
                    #         path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    #         print("Saved model checkpoint to {}\n".format(path))
                # if writer:
                #     writer.add_summary(summaries, step)

    #                 if current_step % FLAGS.checkpoint_every == 0:


