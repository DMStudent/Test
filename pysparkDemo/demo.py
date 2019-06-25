# -*- coding: utf-8 -*-
# File : demo.py
# Author: wangyuan
# mail: wyxidian@gmail.com
# Created Time: 2018/12/25
#!/bin/bash

import logging
from pyspark import SparkContext,SparkConf
import sys
import os
os.environ['PYSPARK_PYTHON'] = '/search/anaconda/envs/py2/bin/python'

logging.basicConfig(format='%(message)s', level=logging.INFO)
# test_file_name = sys.argv[1]
# out_file_name = sys.argv[2]

log_file_name = "input/log.txt"
type_file_name = "input/type.txt"

conf = SparkConf()
# conf.setMaster("local")
# conf.setMaster("yarn-client")
# conf.setAppName(out_file_name)

sc = SparkContext(conf=conf)
log_file = sc.textFile(log_file_name)
type_file = sc.textFile(type_file_name)

log_file = log_file.map(lambda line: (line.split("\t")[1], line.split("\t")[0]))
type_file = type_file.map(lambda line: (line.split("\t")[0], line.split("\t")[1]))


def g(x):
    print x

print("Join:")
log_join_type = log_file.join(type_file)
log_join_type.foreach(g)

print("right Outer Join:")
log_join_type = log_file.rightOuterJoin(type_file)
log_join_type.foreach(g)

print("left Outer Join:")
log_join_type = log_file.leftOuterJoin(type_file)
log_join_type.foreach(g)

print("full Outer Join:")
log_join_type = log_file.fullOuterJoin(type_file)
log_join_type.foreach(g)

print("log subtract type:")
log_join_type = log_file.subtractByKey(type_file)
log_join_type.foreach(g)

print("type subtract log:")
log_join_type = type_file.subtractByKey(log_file)
log_join_type.foreach(g)