# -*- coding: utf-8 -*-
#########################################################################
# File Name: wordCount.py
# Author: wangyuan
# mail: wangyuan214159@sogou-inc.com
# Created Time: 2018年01月16日 星期二 13时44分02秒
#########################################################################
#!/bin/bash
import logging
from pyspark import SparkContext,SparkConf
import sys
import math
import pickle
import urllib

logging.basicConfig(format='%(message)s', level=logging.INFO)


conf=SparkConf()
conf.setMaster("yarn-client")
conf.setAppName("extensive domain application")

test_file_name = sys.argv[1]
out_file_name = sys.argv[2]
sc = SparkContext(conf=conf)
# text_file rdd object
text_file = sc.textFile(test_file_name)



# 色情模型
pkl_file_sex = open('sex_model.pki', 'rb')
model_sex = pickle.load(pkl_file_sex)
counts_sex = model_sex['mat']
thresh_sex = model_sex['thresh']
pos_sex = model_sex['pos']

#乱码模型
model_gib = pickle.load(open('gib_model.pki', 'rb'))
mat_gib = model_gib['mat']
thresh_gib = model_gib['thresh']
accepted_chars = 'abcdefghijklmnopqrstuvwxyz '
pos_gib = dict([(char, idx) for idx, char in enumerate(accepted_chars)])

# 色情相关函数
def ngram(n, l):
    """ Return all n grams from l after normalizing """
    # filtered = normalize(l)
    for start in range(0, len(l) - n + 1):
        yield ''.join(l[start:start + n])

def removeOov(line):
    newLine = ""
    for i in line:
        if pos_sex.has_key(i):
           newLine = newLine+i
    return newLine
def avg_transition_prob(l):
    """ Return the average transition prob from l through log_prob_mat. """
    log_prob = 0.0
    transition_ct = 0
    l = removeOov(l)
    for a, b in ngram(2, l):
        log_prob += counts_sex[pos_sex[a]][pos_sex[b]]
        transition_ct += 1
    # The exponentiation translates from log probs to probs.
    return math.exp(log_prob / (transition_ct or 1))


# *****************************************************
# 乱码相关函数
def normalize(line):
    """ Return only the subset of chars from accepted_chars.
    This helps keep the  model relatively small by ignoring punctuation,
    infrequenty symbols, etc. """
    newLine = ""
    for c in line:
        if c.lower() in accepted_chars:
            newLine = newLine + c.lower()
        else:
            newLine = newLine + " "
    return newLine

def ngram_gib(n, l):
    """ Return all n grams from l after normalizing """
    filtered = normalize(l)
    for start in range(0, len(filtered) - n + 1):
        yield ''.join(filtered[start:start + n])

def avg_gib_prob(l):
    """ Return the average transition prob from l through log_prob_mat. """
    log_prob = 0.0
    transition_ct = 0
    for a, b in ngram_gib(2, l):
        log_prob += mat_gib[pos_gib[a]][pos_gib[b]]
        transition_ct += 1
    # The exponentiation translates from log probs to probs.
    return math.exp(log_prob / (transition_ct or 1))

def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        pass
    return False

def getHost(url):
    protocol, s1 = urllib.splittype(url)
    # ('http', '//www.freedom.com:8001/img/people')
    host, s2 = urllib.splithost(s1)
    # ('www.freedom.com:8001', '/img/people')
    host, port = urllib.splitport(host)
    return host
def isHostGib(host):
    isGib = "0"
    if host.startswith("www"):
        isGib = "0"
    else:
        extList = host.split(".")
        if len(extList) > 0:
            if(is_number(extList[0])):
                isGib = "1"
            elif (len(extList[0])<4):
                isGib = "0"
            else:
                isGib = str(avg_gib_prob(extList[0]))
    return isGib
# counts
def mapFun(line):
    lineList = line.split("\t")
    if(len(lineList)<2):
        return None
    url = lineList[0]
    if(len(url)<1):
        return None
    host = getHost(url)
    if (len(host) < 1):
        return None

    isGib = isHostGib(url)
    title = lineList[1]
    return url+"\t"+title+"\t"+isGib+"\t"+str(avg_transition_prob(title))






query = text_file.map(lambda line: mapFun(line)).saveAsTextFile(out_file_name)