''' mapper of pangerank algorithm'''
import sys

for line in sys.stdin:
    data = line.strip().split('\t')
    if(len(data)>2):
        key = data[0]
        value = float(data[1])
        heros = data[2:]
        for hero in heros:
            v = value / len(heros)
            print '%s\t%s' % (hero, v)
        print '%s\t%s' % (key, 0.0)