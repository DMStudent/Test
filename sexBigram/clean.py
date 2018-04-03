#!/usr/bin/python
# -*- coding: UTF-8 -*-
import re

def remove_punctuation(line):
    rule = re.compile(ur"[^\u4e00-\u9fa5]")
    line = rule.sub('',line)
    return line


def cleanFile(oldF, newF):
    fr = file(oldF)
    fw = file(newF, 'w')
    zhPattern = re.compile(u'[\u4e00-\u9fa5]+')

    for line in fr.readlines():
        if line.find("http") > 0:
            continue
        line = line.decode("utf-8")
        match = zhPattern.search(line)
        if match == None:
            continue
        line = remove_punctuation(line).encode("utf-8")
        fw.write(line.strip() + "\n")

    fw.close()
    fr.close()


if __name__ == '__main__':
    cleanFile("train.txt.bkp", "train.txt")
    cleanFile("good.txt.bkp", "good.txt")
    cleanFile("bad.txt", "bad1.txt")
# import re
# zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
# contents = u'中文'
# match = zhPattern.search(contents)
# print (match.group(0))
