''' reducer of pagerank algorithm'''
import sys
from operator import itemgetter
from itertools import groupby

values = 0.0
alpha = 0.8
N = 4# Size of the web pages


def read_input(file):
    """Read input and split."""
    for line in file:
        yield line.rstrip().split('\t')

data = read_input(sys.stdin)
for key, kviter in groupby(data, itemgetter(0)):
    values = 0
    for k in kviter:
        values += float(k[1])
    values = alpha * values + (1 - alpha) / N
    print '%s\ta\t%s' % (key, values)

