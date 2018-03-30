# --coding:utf-8 --
from snownlp import SnowNLP


def tup2str(dictList):
    result = ""
    for sample_dic in dictList:
        key, value = sample_dic
        result  = result + key + ":" + value + " "
    return  result
def dict2str(dictList):
    result = ""
    for sample_dic in dictList:
        for key, value in sample_dic.items():
            result  = result + key + ":" + unicode(value) + " "
    return  result
text1 = u"百度是一家高科技公司"
text2 = u"百度不是一家高科技公司"


s1 = SnowNLP(text1)
s2 = SnowNLP(text2)
# s.words # [u'这个', u'东西', u'真心',
# u'很', u'赞']

text = u"搜狗是一家高科技公司"

s = SnowNLP(text)
words = " ".join(s.words)
tags = tup2str(s.tags)
sentiments = s.sentiments
pinyin = " ".join(s.pinyin)
keywords = " ".join(s.keywords())
summary = " ".join(s.summary())
http_response = words + '\t' + tags + '\t' + str(sentiments) + '\t' + pinyin + '\t' + keywords + '\t' + summary
print http_response
# print " ".join(s2.tags)
# print " ".join(s2.idf)
# print " ".join(s2.tf)

