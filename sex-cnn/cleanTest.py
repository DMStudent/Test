#!/usr/bin/python
# -*- coding: UTF-8 -*-
import re
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import tensorflow as tf
from cnews_loader import read_vocab, read_category, process_file2, process_file, build_vocab
from cnn_model import TCNNConfig
import numpy as np
base_dir = 'sexText/'

# 清楚标点符号以及非中文字符
ruleZh = re.compile(ur"[^\u4e00-\u9fa5]")
ruleSp = re.compile(r" +")


test_raw = os.path.join(base_dir, 'off2000/off2000.raw')
test_txt = os.path.join(base_dir, 'off2000/off2000.txt')
test_tf = os.path.join(base_dir, 'off2000/off2000.tfrecords')
vocab_dir = os.path.join(base_dir, 'vocab.txt')



# int64
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
# byte
def _byte_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def convert_to_TFRecords_withText(dataset, name):
    """Convert mnist dataset to TFRecords"""
    texts, titles, labels = dataset
    n_examples = len(titles)

    filename = name
    print"Writing:"+filename

    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(n_examples):
            if(index%1000 == 0):
                print index
            text = "".join(texts[index]).encode("utf-8")+"\n"
            title = titles[index].tolist()
            label = labels[index]
            label = [int(x) for x in label]
            example = tf.train.Example(features=tf.train.Features(
                feature={"text": _byte_feature(text),
                         "title": _int64_feature(title),
                         "label": _int64_feature(label)}))
            writer.write(example.SerializeToString())


def convert_to_TFRecords(dataset, name):
    """Convert mnist dataset to TFRecords"""
    titles, labels = dataset
    n_examples = len(titles)

    filename = name + ".tfrecords"
    print"Writing:"+filename

    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(n_examples):
            title = titles[index].tolist()
            label = labels[index]
            label = [int(x) for x in label]
            example = tf.train.Example(features=tf.train.Features(
                feature={"title": _int64_feature(title),
                         "label": _int64_feature(label)}))
            writer.write(example.SerializeToString())

def read_TFRecords_test(name):
    filename = name
    record_itr = tf.python_io.tf_record_iterator(path=filename)
    for r in record_itr:
        example = tf.train.Example()
        example.ParseFromString(r)

        text = example.features.feature["text"].bytes_list.value
        print"Text:" + "".join(text)
        label = example.features.feature["label"].int64_list.value[0]
        print label
        title = example.features.feature["title"].int64_list.value
        print title
        break  # 只读取一个Example

def read_example(filename_queue):
    """Read one example from filename_queue"""
    reader = tf.TFRecordReader()
    config = TCNNConfig()
    key, value = reader.read(filename_queue)
    features = tf.parse_single_example(value, features={"text": tf.VarLenFeature(tf.string),
                                                        "title": tf.FixedLenFeature([config.seq_length], tf.int64),
                                                        "label": tf.FixedLenFeature([config.num_classes], tf.int64)})

    text = features["text"]
    title = tf.cast(features["title"], tf.int32)
    label = tf.cast(features["label"], tf.int32)
    return text, title, label


def remove_punctuation(line):
    line = ruleZh.sub('',line)
    line = ruleSp.sub(' ', line)
    return line


def preprocess(filenameSex):
    count = 0
    print "preprocess:"+filenameSex
    fr = file(filenameSex)
    fw = file(test_txt, 'w')
    for line in fr.readlines():
        line = line.decode("utf-8").split("\t")

        if(len(line)<2):
            pass
        title = line[0]
        label = line[1].split(" ")[0]
        if(len(title)<1 or len(label)<1):
            pass
        title = remove_punctuation(title)
        if (len(title) > 5):
            count = count + 1
            fw.write(label+"\t"+title+"\n")
    fr.close()
    fw.close()

    print "done."



if __name__ == '__main__':
    preprocess(test_raw)
    config = TCNNConfig()
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    text_test, x_test, y_test = process_file2(test_txt, word_to_id, cat_to_id, config.seq_length)

    convert_to_TFRecords_withText([text_test, x_test, y_test], test_tf)

    queue = tf.train.string_input_producer([test_tf], num_epochs=1)
    text, title, label = read_example(queue)

    text_batch, title_batch, label_batch = tf.train.batch([text, title, label], batch_size=1000, capacity=5000,
                                                    num_threads=1)
    count = 0
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                # Run training steps or whatever
                texts, titles, labels = sess.run([text_batch, title_batch, label_batch])
                print "--------------------------"
                print count
                count = count + 1
                # print titles
                # print labels
                # texts = "".join(texts.values)
                # print texts
                print(titles.shape, labels.shape)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')

        coord.request_stop()
        coord.join(threads)


