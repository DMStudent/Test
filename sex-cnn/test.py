# -*- coding: utf-8 -*-
# Author: wangyuan
# mail: wyxidian@gmail.com
# Created Time: 
#!/bin/bash

import os

import tensorflow as tf

# our NN's output
logits = tf.constant([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
# step1:do softmax
y = tf.nn.softmax(logits)
# true label
y_ = tf.constant([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
# step2:do cross_entropy
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# do cross_entropy just one step
cross_entropy2 = tf.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y_, pos_weight=0.5))  # dont forget tf.reduce_sum()!!

with tf.Session() as sess:
    softmax = sess.run(y)
    c_e = sess.run(cross_entropy)
    c_e2 = sess.run(cross_entropy2)
    print("step1:softmax result=")
    print(softmax)
    print("step2:cross_entropy result=")
    print(c_e)
    print("Function(softmax_cross_entropy_with_logits) result=")
    print(c_e2)
