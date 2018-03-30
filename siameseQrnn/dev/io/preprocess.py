# -*- coding: utf-8 -*-
#########################################################################
# File Name: preprocess.py
# Author: wangyuan
# mail: wangyuan214159@sogou-inc.com
# Created Time: 2018年01彜~H22彗¥ 彘~_彜~_䷾@ 15彗¶16佈~F15禾R
#########################################################################
#!/bin/bash
import random
queryList = []
titleList = []

fr = file('/search/odin/data/wangyuan/data-wenti/raw', 'r')
count = 0
all = []
for line in fr.readlines():
    query = "\t".join(line.split("\t")[:2])
    title = "\t".join(line.split("\t")[2:])
    queryList.append(query)
    titleList.append(title)
    all.append("1" + "\t" + query + "\t" + title)
    count = count + 1
random.shuffle(queryList)
fr.close()

for i in range(count):
    all.append("0"+"\t"+queryList[i]+"\t"+titleList[i])

random.shuffle(all)
fw = file('/search/odin/data/wangyuan/data-wenti/train', 'w')
for i in range(count*2):
    fw.write(all[i])
fw.close()