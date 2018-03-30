# --coding:utf-8 --
"""Script to illustrate usage of tf.estimator.Estimator in TF v1.3"""
"https://zhuanlan.zhihu.com/p/34918066"

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

import numpy as np



# int64
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))

# bytes
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_TFRecords(dataset, name):
    """Convert mnist dataset to TFRecords"""
    images, labels = dataset.images, dataset.labels
    n_examples = dataset.num_examples

    filename = name + ".tfrecords"
    print("Writing", filename)

    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(n_examples):
            image_bytes = images[index].tostring()
            label = labels[index]
            example = tf.train.Example(features=tf.train.Features(
                feature={"image": _bytes_feature(image_bytes),
                         "label": _int64_feature(label)}))
            writer.write(example.SerializeToString())


def read_TFRecords_test(name):
    filename = name
    record_itr = tf.python_io.tf_record_iterator(path=filename)
    for r in record_itr:
        example = tf.train.Example()
        example.ParseFromString(r)

        label = example.features.feature["label"].int64_list.value[0]
        print("Label", label)
        image_bytes = example.features.feature["image"].bytes_list.value[0]
        print (image_bytes)
        # img = np.fromstring(image_bytes, dtype=np.uint8).reshape(28, 28)
        # print(img)
        # plt.imshow(img, cmap="gray")
        # plt.show()
        break  # 只读取一个Example

def read_example(filename_queue):
    """Read one example from filename_queue"""
    reader = tf.TFRecordReader()
    key, value = reader.read(filename_queue)
    features = tf.parse_single_example(value, features={"image": tf.FixedLenFeature([], tf.string),
                                                        "label": tf.FixedLenFeature([], tf.int64)})
    image = tf.decode_raw(features["image"], tf.uint8)
    image = tf.reshape(image, [28, 28])
    label = tf.cast(features["label"], tf.int32)
    return image, label

if __name__ == "__main__":

    # 数据格式转换
    # mnist_datasets = mnist_data.read_data_sets("mnist_data", dtype=tf.uint8, reshape=False)
    # convert_to_TFRecords(mnist_datasets.train, "train")
    # convert_to_TFRecords(mnist_datasets.validation, "validation")
    # convert_to_TFRecords(mnist_datasets.test, "test")
    #
    # read_TFRecords_test("train.tfrecords")

    queue = tf.train.string_input_producer(["train.tfrecords"], num_epochs=10)
    image, label = read_example(queue)

    img_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=32, capacity=5000,
                                                    min_after_dequeue=2000, num_threads=4)
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                # Run training steps or whatever
                images, labels = sess.run([img_batch, label_batch])
                print(images.shape, labels.shape)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')

        coord.request_stop()
        coord.join(threads)

