#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn import metrics

from cnn_model import TCNNConfig, TextCNN
from cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

base_dir = 'sexText/'
train_dir = os.path.join(base_dir, 'train.tfrecords')
vocab_dir = os.path.join(base_dir, 'vocab.txt')
# test_dir = os.path.join(base_dir, 'test.tfrecords')
# test_output = os.path.join(base_dir, 'test.output')
test_dir = os.path.join(base_dir, 'off2000/off2000.tfrecords')
test_output = os.path.join(base_dir, 'off2000/off2000.output')
save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def read_example(filename_queue):
    """Read one example from filename_queue"""
    reader = tf.TFRecordReader()
    key, value = reader.read(filename_queue)
    features = tf.parse_single_example(value, features={"title": tf.FixedLenFeature([config.seq_length], tf.int64),
                                                        "label": tf.FixedLenFeature([config.num_classes], tf.int64)})

    title = tf.cast(features["title"], tf.int32)
    label = tf.cast(features["label"], tf.int32)
    return title, label


def read_example_text(filename_queue):
    """Read one example from filename_queue"""
    reader = tf.TFRecordReader()
    key, value = reader.read(filename_queue)
    features = tf.parse_single_example(value, features={"text": tf.VarLenFeature(tf.string),
                                                        "title": tf.FixedLenFeature([config.seq_length], tf.int64),
                                                        "label": tf.FixedLenFeature([config.num_classes], tf.int64)})
    # image = tf.decode_raw(features["image"], tf.uint8)
    # image = tf.reshape(image, [28, 28])
    text = features["text"]
    title = tf.cast(features["title"], tf.int32)
    label = tf.cast(features["label"], tf.int32)
    return text, title, label

def train(config):
    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 100000  # 如果超过1000轮未提升，提前结束训练

    with tf.Graph().as_default(), tf.Session() as sess:
        queueTrain = tf.train.string_input_producer([train_dir], num_epochs=config.num_epochs)
        title, label = read_example(queueTrain)
        title_batch, label_batch = tf.train.shuffle_batch([title, label], batch_size=config.batch_size, capacity=5000,
                                                          min_after_dequeue=2000, num_threads=2)

        with tf.variable_scope("model", initializer=tf.random_uniform_initializer(-1 * 1, 1)):
            model = TextCNN(config=config, input_x=title_batch, input_y=label_batch)
        fetches = [model.embedding_inputs, model.loss, model.acc]
        feed_dict = {}
        # init
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        # 配置 Saver
        saver = tf.train.Saver()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        try:
            while not coord.should_stop():
                # Run training steps or whatever
                # titles, labels = sess.run([title_batch, label_batch])
                if total_batch % config.print_per_batch == 0:
                    # 每多少轮次输出在训练集和验证集上的性能
                    embedding_inputs, loss_val, acc_val = sess.run(fetches, feed_dict)
                    # loss_val, acc_val = evaluate(sess, titleVal_batch, labelVal_batch)  # todo

                    if acc_val > best_acc_val:
                        # 保存最好结果
                        best_acc_val = acc_val
                        last_improved = total_batch
                        saver.save(sess=sess, save_path=save_path)
                        improved_str = '*'
                    else:
                        improved_str = ''

                    time_dif = get_time_dif(start_time)
                    msg = 'Iter: {0:>6}, Val Loss: {1:>6.2}, Val Acc: {2:>7.2%}, Time: {3} {4}'
                    print(msg.format(total_batch, loss_val, acc_val, time_dif, improved_str))
                    # print(embedding_inputs)

                sess.run(model.optim, feed_dict)
                total_batch += 1

                if total_batch - last_improved > require_improvement:
                    # 验证集正确率长期不提升，提前结束训练
                    print("No optimization for a long time, auto-stopping...")
                    coord.should_stop()
                    break  # 跳出循环

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')

        coord.request_stop()
        coord.join(threads)


#
def test(config):
    print("Loading test data...")
    config.dropout_keep_prob = 1.0
    start_time = time.time()
    batch_size = config.batch_size
    count = 0
    data_len = 2973000
    y_test_cls = np.zeros(shape=data_len, dtype=np.int32)
    y_pred_cls = np.zeros(shape=data_len, dtype=np.int32)  # 保存预测结果

    with tf.Graph().as_default(), tf.Session() as sess:
        queueTest = tf.train.string_input_producer([test_dir], num_epochs=1)
        title, label = read_example(queueTest)
        title_batch, label_batch = tf.train.batch([title, label], batch_size=config.batch_size, capacity=5000,
                                                          num_threads=1)
        with tf.variable_scope("model", initializer=tf.random_uniform_initializer(-1 * 1, 1)):
            model = TextCNN(config=config, input_x=title_batch, input_y=label_batch)




        fetches = [model.y_pred_cls, model.input_y]
        feed_dict = {}
        # init
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        # 配置 Saver
        saver = tf.train.Saver()
        saver.restore(sess=sess, save_path=save_path)  # 读取保存的模型
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        try:
            while not coord.should_stop():
                start_id = count * batch_size
                end_id = min((count + 1) * batch_size, data_len)
                y_pred, y_test = sess.run(fetches, feed_dict=feed_dict)
                y_pred_cls[start_id:end_id] = y_pred
                y_test_cls[start_id:end_id] = np.argmax(y_test, 1)
                count = count+1
                print(count)


        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')

        coord.request_stop()
        coord.join(threads)

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

#
def testWithText(config):
    print("Loading test data...")
    config.dropout_keep_prob = 1.0
    start_time = time.time()
    batch_size = config.batch_size
    count = 0

    fw = file(test_output, "w")
    with tf.Graph().as_default(), tf.Session() as sess:
        queueTest = tf.train.string_input_producer([test_dir], num_epochs=1)
        text, title, label = read_example_text(queueTest)
        text_batch, title_batch, label_batch = tf.train.batch([text, title, label], batch_size=config.batch_size, capacity=5000,
                                                          num_threads=1)
        with tf.variable_scope("model", initializer=tf.random_uniform_initializer(-1 * 1, 1)):
            model = TextCNN(config=config, input_x=title_batch, input_y=label_batch)


        fetches = [text_batch, model.y_pred_cls, model.y_pred_score, model.input_y]
        feed_dict = {}
        # init
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        # 配置 Saver
        saver = tf.train.Saver()
        saver.restore(sess=sess, save_path=save_path)  # 读取保存的模型
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        try:
            while not coord.should_stop():
                texts, y_pred, score, y_test = sess.run(fetches, feed_dict=feed_dict)
                y_test = np.argmax(y_test, 1)
                texts = "".join(texts.values).split("\n")
                for i in range(len(texts)-1):
                    fw.write(texts[i]+"\t"+str(y_test[i])+"\t"+str(y_pred[i])+"\t"+str(score[i][0])+"\t"+str(score[i][1])+"\t"+str(score[i][2])+"\n")
                count = count+1
                # print(count)


        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')

        coord.request_stop()
        coord.join(threads)
    fw.close()



if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test', 'testWithText']:
        raise ValueError("""usage: python run_cnn.py [train / test / testWithText]""")

    print('Configuring CNN model...')
    config = TCNNConfig()
    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(train_dir, vocab_dir, config.vocab_size)
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)


    if sys.argv[1] == 'train':
        train(config)
    elif sys.argv[1] == 'test':
        test(config)
    else:
        testWithText(config)
