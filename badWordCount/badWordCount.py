# -*- coding: utf-8 -*-
# Author: wangyuan
# mail: wyxidian@gmail.com
# Created Time: 
#!/bin/bash

import os
import re



def readDict(filename):
    dic = {}
    fr = file(filename)
    idx = 0
    for line in fr.readlines():
        line = line.strip().split("\t")[0]
        if len(line)>0:
            dic[line.decode('utf-8')] = idx
            idx = idx + 1
    fr.close()
    return dic

def filterQuery(filename, outputname, dic_sex, dic_bet):
    fr = file(filename)
    fw = file(outputname, 'w')
    count = 0
    for line in fr.readlines():
        line = line.decode('utf-8')
        query = line.split("\t")[0].encode('utf-8')
        sex_count = 0
        bet_count = 0
        for word in dic_sex:
            pattern = re.compile("("+word+")")
            sex_count = sex_count + len(re.findall(pattern, line))
        for word in dic_bet:
            pattern = re.compile("("+word+")")
            bet_count = bet_count + len(re.findall(pattern, line))
        fw.write(query+"\t"+str(sex_count)+"\t"+str(bet_count))
        if(count % 100 ==0):
            print count
        count = count + 1
    fw.close()
    fr.close()


if __name__ == '__main__':
    sex_word_file = "input/sex.word"
    dic_sex = readDict(sex_word_file)
    bet_word_file = "input/bet.word"
    dic_bet = readDict(bet_word_file)
    test_file = "/search/odin/data/wangyuan/pycharmProjects/webSearchXml/output/out.txt"
    outputname = "output/out.txt"
    filterQuery(test_file, outputname, dic_sex, dic_bet)

    print "done"
