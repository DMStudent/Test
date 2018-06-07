# -*- coding: utf-8 -*-
# Author: wangyuan
# mail: wyxidian@gmail.com
# Created Time: 
#!/bin/bash

from nltk.probability import FreqDist
import math
import re


min_entropy = 0.8
min_p = 7
max_gram = 4
count_appear = 30
freq_all = [0]
word_all = [0]

def gram(text, max_gram):
    t1 = [i for i in text]
    loop = len(t1) + 1 - max_gram
    t = []
    for i in range(loop):
        t.append(text[i:i + max_gram])
    if max_gram == 1:
        return t1
    else:
        return t



def pro(word):
    len_word = len(word)
    total_count = len(word_all[len_word])
    pro = float(freq_all[len_word][word]) / total_count
    return pro


def entropy(alist):
    f = FreqDist(alist)
    ent = (-1) * sum([float(i) / len(alist) * math.log(float(i) / len(alist)) for i in f.values()])
    return ent

def save_result(filename, final_word):
    print(u'正在对结果进行排序...')
    for i in range(len(final_word)):
        final_word[i] = (final_word[i], str(freq_all[len(final_word[i])][final_word[i]]))
    final_word = sorted(final_word, key=lambda a: a[1], reverse=True)

    print(u'正在保存...')
    with open(filename,'w') as fw:
        for i in final_word:
            fw.write(i[0].encode('utf-8')+"\t"+i[1]+"\n")




if __name__ == '__main__':
    f = open("/search/odin/data/wangyuan/pycharmProjects/sex-cnn/sexText/sex-long/sex.raw2", 'r')
    text = f.read().decode("utf-8")
    f.close()
    stop_word = [u'，', u'\n', u'。', u'、', u'：', u'(', u')', u'[', u']',
                 u'.', u',', u' ', u'\u3000', u'”', u'“', u'？', u'?', u'！',
                 u'‘', u'’', u'…', u'【', u'】', u'《', u'》', u' ', u'-',
                 u'\'', u'：', u'/', u'"', u'．']

    for i in stop_word:
        text = text.replace(i, "")

    for i in range(1, max_gram + 1):
        print(u'正在生成%s字词...' % i)
        t = gram(text, i)
        freq = FreqDist(t)
        word_all.append(t)
        freq_all.append(freq)

    # 筛选一部分符合互信息的单词
    final_word = [[] for i in range(max_gram+1)]
    for i in range(2, max_gram + 1):
        print(u'正在进行%s字词的互信息筛选(%s)...' % (i, len(word_all[i])))
        for j in word_all[i]:
            if freq_all[i][j] < count_appear:
                pass
            else:
                p = min([pro(j[:t]) * pro(j[t:]) for t in range(1, len(j))])
                if math.log(pro(j) / p) > min_p:
                    final_word[i].append(j)
        final_word[i] = list(set(final_word[i]))


    # 筛选左右熵
    final_word2 = []
    for i in range(2, max_gram + 1):
        print(u'正在进行%s字词的左右熵筛选(%s)...' % (i, len(final_word[i])))
        for j in final_word[i]:

            lr = re.findall('(.)%s(.)' % j, text)
            left_entropy = entropy([w[0] for w in lr])
            right_entropy = entropy([w[1] for w in lr])
            if min([right_entropy, left_entropy]) > min_entropy:
                final_word2.append(j)
        filename = 'result'
    save_result(filename, final_word2)
    print(u'ok')


