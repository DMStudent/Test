#!/usr/bin/python

import re
import sys

if __name__ == '__main__':
  while True:
    line = sys.stdin.readline()
    if line == '':
      break
    line = line.rstrip()
    tokens = line.split('\t')
    assert len(tokens) == 5
    query_hidden = re.sub('^query_termid:','',tokens[2])
    title_hidden = re.sub('^cnn_title:','',tokens[4])
    if query_hidden != '' and title_hidden != '':
      print query_hidden + '\t' + title_hidden + '\t' + title_hidden

