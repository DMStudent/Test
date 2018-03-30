#!/usr/bin/python
# -*- coding: UTF-8 -*-
import re



def cleanFile(oldF, newF):
    fr = file(oldF)
    fw = file(newF, 'w')
    zhPattern = re.compile(u'[\u4e00-\u9fa5]+')

    for line in fr.readlines():
        if line.find("http") > 0:
            continue
        match = zhPattern.search(line.decode("utf-8"))
        if match == None:
            continue
        fw.write(line.strip() + "\n")

    fw.close()
    fr.close()


if __name__ == '__main__':
    cleanFile("raw.txt", "train.txt")
    cleanFile("good.txt", "good1.txt")
    cleanFile("bad.txt", "bad1.txt")
# import re
# zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
# contents = u'中文'
# match = zhPattern.search(contents)
# print (match.group(0))
