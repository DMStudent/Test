#!/usr/bin/python
# -*- coding: UTF-8 -*-
import re
import random
# 清楚标点符号以及非中文字符
ruleZh = re.compile(ur"[^\u4e00-\u9fa5]")
ruleSp = re.compile(r" +")
baseDir = "./sexText"


def remove_punctuation(line):
    line = ruleZh.sub('',line)
    line = ruleSp.sub(' ', line)
    return line


def preprocess(filename):
    count = 0
    titleList = []
    print "preprocess:"+filename
    fr = file(filename)
    for line in fr.readlines():
        lineList = line.split("\t")
        if len(lineList) < 2:
            pass
        line = lineList[0].decode("utf-8")
        label = lineList[1].strip()
        line = remove_punctuation(line)
        if len(line) > 5:
            count = count + 1
            titleList.append(label+"\t"+line.encode("utf-8")+"\n")

    fr.close()

    random.shuffle(titleList)

    print "generate train val test files."
    fw1 = file(baseDir+"/train.txt", 'w')
    fw2 = file(baseDir+"/test.txt", 'w')
    idx = 0
    for line in titleList:
        if idx < count*0.95:
            fw1.write(line)
        else:
            fw2.write(line)
        idx = idx + 1
        if idx % 10000 == 0:
            print idx
    fw1.close()
    fw2.close()
    print "done."


if __name__ == '__main__':
    preprocess(baseDir+"/raw/train.raw")
