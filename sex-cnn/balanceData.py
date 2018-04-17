# -*- coding: utf-8 -*-
# Author: wangyuan
# mail: wyxidian@gmail.com
# Created Time: 
#!/bin/bash

# coding: utf-8

import sys
from collections import Counter
import os
import numpy as np
import tensorflow.contrib.keras as kr

if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False


def native_word(word, encoding='utf-8'):
    """如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
    if not is_py3:
        return word.encode(encoding)
    else:
        return word


def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content


def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)


def read_file(filename):
    """读取文件数据"""
    contents0, contents1, contents2 = [], [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    if label == "0":
                        contents0.append(list(native_content(content)))
                    elif label == "1":
                        contents1.append(list(native_content(content)))
                    elif label == "2":
                        contents2.append(list(native_content(content)))
            except:
                pass
    return contents0, contents1, contents2


def Static_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    contents0, contents1, contents2 = read_file(train_dir)

    all_data = []
    for content in contents0:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    count_pairs = map(lambda x: x[0] + "\t" + str(x[1]), count_pairs)
    open_file(vocab_dir+"0", mode='w').write('\n'.join(count_pairs) + '\n')

    all_data = []
    for content in contents1:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    count_pairs = map(lambda x: x[0] + "\t" + str(x[1]), count_pairs)
    open_file(vocab_dir + "1", mode='w').write('\n'.join(count_pairs) + '\n')

    all_data = []
    for content in contents2:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    count_pairs = map(lambda x: x[0] + "\t" + str(x[1]), count_pairs)
    open_file(vocab_dir + "2", mode='w').write('\n'.join(count_pairs) + '\n')


# coding: utf-8

import sys
from collections import Counter

import numpy as np
import tensorflow.contrib.keras as kr

if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False


def native_word(word, encoding='utf-8'):
    """如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
    if not is_py3:
        return word.encode(encoding)
    else:
        return word


def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content


def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)



def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    contents0, contents1, contents2 = read_file(train_dir)

    all_data = []
    for content in contents0:
        all_data.extend(content)

    for content in contents1:
        all_data.extend(content)

    for content in contents2:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')

def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [native_content(_.strip()) for _ in fp.readlines()]
    ids = []
    for i in range(len(words)):
        ids.append([0, 0, 0])
    word_to_id = dict(zip(words, ids))
    return word_to_id

def isTooMuch(contentList, word_to_id):
    for i in contentList:
        if word_to_id[i] > 200000:
            return True
    return False
def sample(filename, vocab_dir, file_output_name):
    """读取文件数据"""
    contents, labels = [], []
    open_file(filename)
    word_to_id = read_vocab(vocab_dir)

    fw = open_file(file_output_name, 'w')
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    if label == "0":
                        contentList = list(native_content(content))
                        too_much = False
                        for i in contentList:
                            if word_to_id[i][0] > 200000:
                                too_much = True

                        if(not too_much):
                            fw.write(line)
                            for i in contentList:
                                word_to_id[i][0] = word_to_id[i][0] + 1
                    elif label == "1":
                        contentList = list(native_content(content))
                        too_much = False
                        for i in contentList:
                            if word_to_id[i][1] > 200000:
                                too_much = True

                        if (not too_much):
                            fw.write(line)
                            for i in contentList:
                                word_to_id[i][1] = word_to_id[i][1] + 1
                    elif label == "2":
                        contentList = list(native_content(content))
                        too_much = False
                        for i in contentList:
                            if word_to_id[i][2] > 200000:
                                too_much = True
                        if (not too_much):
                            fw.write(line)
                            for i in contentList:
                                word_to_id[i][2] = word_to_id[i][2] + 1
            except:
                pass
    fw.close()



if __name__ == '__main__':
    base_dir = 'sexText/'
    train_dir = os.path.join(base_dir, 'raw/train.txt')
    vocab_dir = os.path.join(base_dir, 'train.top')
    vocab_size = 1000
    file_output_name = os.path.join(base_dir, 'train.sample')
    # print "processing: "+train_dir
    # Static_vocab(train_dir, vocab_dir, vocab_size)
    # print "done"
    vocab_dir = os.path.join(base_dir, 'vocab.txt')
    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(train_dir, vocab_dir, vocab_size)

    sample(train_dir, vocab_dir, file_output_name)
