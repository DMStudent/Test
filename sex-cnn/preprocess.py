# --coding:utf-8 --
"""Script to illustrate usage of tf.estimator.Estimator in TF v1.3"""
"https://zhuanlan.zhihu.com/p/34918066"

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import tensorflow as tf
from cnews_loader import read_vocab, read_category, process_file2, process_file, build_vocab
from cnn_model import TCNNConfig
import numpy as np
base_dir = 'sexText/'
train_dir = os.path.join(base_dir, 'train.txt')
test_dir = os.path.join(base_dir, 'test.txt')
val_dir = os.path.join(base_dir, 'val.txt')
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

    filename = name + ".tfrecords"
    print"Writing:"+filename

    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(n_examples):
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
    config = TCNNConfig()
    reader = tf.TFRecordReader()
    key, value = reader.read(filename_queue)
    features = tf.parse_single_example(value, features={"text": tf.VarLenFeature(tf.string),
                                                        "title": tf.FixedLenFeature([config.seq_length], tf.int64),
                                                        "label": tf.FixedLenFeature([config.num_classes], tf.int64)})

    text = features["text"]
    title = tf.cast(features["title"], tf.int32)
    label = tf.cast(features["label"], tf.int32)
    return text, title, label

if __name__ == "__main__":
    # 数据格式转换
    config = TCNNConfig()
    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(train_dir, vocab_dir, config.vocab_size)
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    text_test, x_test, y_test = process_file2(test_dir, word_to_id, cat_to_id, config.seq_length)

    convert_to_TFRecords([x_train, y_train], base_dir+"/train")
    # convert_to_TFRecords_withText([text_test, x_test, y_test], base_dir+"/test")



    # 读取测试
    # read_TFRecords_test("sexText/train10.tfrecords")
    # queue = tf.train.string_input_producer(["sexText/test.tfrecords"], num_epochs=10)
    # text, title, label = read_example(queue)
    #
    # text_batch, title_batch, label_batch = tf.train.batch([text, title, label], batch_size=1, capacity=5000,
    #                                                 num_threads=1)
    # count = 0
    # with tf.Session() as sess:
    #     sess.run(tf.local_variables_initializer())
    #     sess.run(tf.global_variables_initializer())
    #
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #
    #     try:
    #         while not coord.should_stop():
    #             # Run training steps or whatever
    #             texts, titles, labels = sess.run([text_batch, title_batch, label_batch])
    #             print "--------------------------"
    #             print count
    #             count = count + 1
    #             # print titles
    #             # print labels
    #             texts = "".join(texts.values)
    #             print texts
    #             print(titles.shape, labels.shape)
    #
    #     except tf.errors.OutOfRangeError:
    #         print('Done training -- epoch limit reached')
    #
    #     coord.request_stop()
    #     coord.join(threads)

