# _*_ coding: utf-8 _*_
import sys
import multiprocessing as mp
import logging
from ishub import genLabelFromHtml
#filename = sys.argv[1]
filename = "head10"
#featurefilename = sys.argv[2]
featurefilename = "titleWithWeight.txt"
lock = mp.Lock()
counter = mp.Value('i', 0) # int type，相当于java里面的原子变量

def genLabel(line):
    global lock, counter
    with lock:
        print "row:" + str(counter.value)
        counter.value += 1
        result = genLabelFromHtml(line)
    return result


def multicore():
    urlDict = {}
    with open(featurefilename, 'r') as f:
        for line in f.readlines():
            line = line.split("\t")
            url = line[0]
            w = line[1]
            urlDict[url] = w

    with open(filename, 'r') as f:                  #以读方式打开文件
        pool = mp.Pool()
        multi_res =[pool.apply_async(genLabel, (line,)) for line in f.readlines()]
        result = [res.get() for res in multi_res]
        page_type_feature = []
        for item in result:
            url = item.split("\t")[0]
            if urlDict.has_key(url):
                item = item + "\t" + urlDict[url]
                page_type_feature.append(item)



        #result = "".join([res.get() for res in multi_res])
        output="page_type.txt"
        with open(output, 'w') as f:
            f.write("".join(page_type_feature))
        print "successs!"

if __name__ == '__main__':
    multicore()

