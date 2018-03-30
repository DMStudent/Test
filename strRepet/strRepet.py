#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys

if __name__ == '__main__':
    filename = '51test'
    f = open(filename)
    for line in f.readlines():
        text = line.strip().split('\t')[0].decode("utf-8")
        textLen = len(text)
        textList = [i for i in text]
        uniqChar = set(textList)
        uniqLen = len(uniqChar)

        print text, textLen, uniqLen, format(float(textLen)/float(uniqLen),'.2f')

