#!/bin/python
'''Reducer for sort'''
import sys
from operator import itemgetter
from itertools import groupby

def read_input(file):
    """Read input and split."""
    for line in file:
        yield line.rstrip().split('\t')

data = read_input(sys.stdin)

for key, kviter in groupby(data, itemgetter(0)):
    values = ""
    for k in kviter:
        if(len(k)==3):
            if(k[1].startswith("a")):
                values = k[2] + "\t" + values
            else:
                values = values + "\t".join(k[1:])
        else:
            values = values + "\t".join(k[1:])
    print values

