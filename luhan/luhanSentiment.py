# _*_ coding: utf-8 _*_
import pandas as pd
from pandas import Series, DataFrame
from snownlp import SnowNLP


def sentimentAnalysis(textlist):
    count = 1
    sentimentslist = []
    for li in textlist:
        print('正在判断第 %s 条评论：' % count)
        count = count + 1
        s = SnowNLP(li)
        sentimentslist.append([li,s.sentiments])
    return sentimentslist

comment = pd.read_csv("data/comment.csv", encoding='utf-8', dtype={'text':str})
textlist = comment["text"]
sentimentslist = sentimentAnalysis(textlist)
sentiments = DataFrame(sentimentslist,columns=['text','sentiment'])
sentiments.to_csv("data/sentiment.csv", encoding='utf-8')