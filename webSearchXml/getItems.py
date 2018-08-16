# -*- coding: utf-8 -*-
# Author: wangyuan
# mail: wyxidian@gmail.com
# Created Time: 
#!/bin/bash

import os
import urllib2
import re
import time
import sys
from WebSearchXml import WebSearchXml
PUBLIC_PATH="./"
sys.path.append(PUBLIC_PATH)


pattern_zh = re.compile(u"[\u4E00-\u9FA5]+")
pattern_space=re.compile('\s+')

def filterQuery(query):
    if len(query)>5 or len(query)<1:
        return False
    match = pattern_zh.search(query)
    if match != None:
        return True
    else:
        return False


def readFile(filename):
    fileList = []
    fr = file(filename)
    for line in fr.readlines():
        line = line.strip()
        if len(line)>0 and filterQuery(line.decode('utf-8')):
            fileList.append(line)
    fr.close()
    return fileList

if __name__ == '__main__':
    filename = 'input/sex.word'
    outname = 'output/items.sex.word'
    fw = file(outname, 'w')
    fileList = readFile(filename)
    count = 0
    for line in fileList:
        ret = WebSearchXml().getItems(line)
        for word in ret:
            fw.write(line+"\t"+word+"\n")
        count = count + 1
        if(count % 100 == 0):
            time.sleep(2)
            print count
    fw.close()







