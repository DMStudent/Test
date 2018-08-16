# -*- coding: utf-8 -*-
# Author: wangyuan
# mail: wyxidian@gmail.com
# Created Time: 
#!/bin/bash

import os
import re
rulePunc = re.compile(ur"[^\u4e00-\u9fa5|\w|]")

if __name__ == '__main__':
    inputname = '/search/odin/data/wangyuan/pycharmProjects/badWordCount/output/sex.word2'
    outputname = '/search/odin/data/wangyuan/pycharmProjects/badWordCount/output/sex.word3'
    fr = file(inputname)
    fw = file(outputname, 'w')
    for line in fr.readlines():
        line = line.strip().decode('utf-8')
        line = re.sub(rulePunc,'', line)
        if len(line) > 1:
            fw.write(line.encode('utf-8')+'\n')
    fr.close()
    fw.close()
    print "done"
