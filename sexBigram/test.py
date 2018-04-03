#!/usr/bin/python
# -*- coding: UTF-8 -*-

import math
import pickle
import re

pkl_file = open('model.pki', 'rb')
model = pickle.load(pkl_file)
counts = model['mat']
thresh = model['thresh']
pos = model['pos']



def ngram(n, l):
    """ Return all n grams from l after normalizing """
    # filtered = normalize(l)
    for start in range(0, len(l) - n + 1):
        yield ''.join(l[start:start + n])

def removeOov(line):
    newLine = ""
    for i in line:
        if pos.has_key(i):
           newLine = newLine+i
    return newLine
def remove_punctuation(line):
    rule = re.compile(ur"[^\u4e00-\u9fa5]")
    line = rule.sub('',line)
    return line
def avg_transition_prob(l, log_prob_mat):
    """ Return the average transition prob from l through log_prob_mat. """
    log_prob = 0.0
    transition_ct = 0
    l = removeOov(l)
    l = remove_punctuation(l)
    if(len(l)<8):
        return 0.0001
    for a, b in ngram(2, l):
        # if not pos.has_key(a):
        #     a = " "
        # if not pos.has_key(b):
        #     b = " "
        log_prob += log_prob_mat[pos[a]][pos[b]]
        transition_ct += 1
    # The exponentiation translates from log probs to probs.
    return math.exp(log_prob / (transition_ct or 1))


def test():
    print "thresh=" + str(thresh)
    print avg_transition_prob(u"嗯不要快点插花核车震", counts)
    print avg_transition_prob(u"出包王女邪恶漫画本子", counts)
    print avg_transition_prob(u"姐孕在手机线观看", counts)
    print avg_transition_prob(u"乡村乱来大杂烩", counts)
    print avg_transition_prob(u"女主在小孩时被男主侵犯", counts)
    print avg_transition_prob(u"还珠格格尔泰风流", counts)
    print avg_transition_prob(u"我与爸爸在厨房嗯嗯", counts)
    print avg_transition_prob(u"囚妃传番外相公的惩罚", counts)
    print avg_transition_prob(u"深春弄潮瑶池被同学征服的母亲", counts)
    print avg_transition_prob(u"凤玲公车系列辣文安琪儿", counts)
    print avg_transition_prob(u"女主在小孩时被男主侵犯轩辕大宝全文阅读重生还珠格格尔泰风流我与爸爸在厨房嗯嗯", counts)
    print avg_transition_prob(u"囚妃传番外相公的惩罚深春弄潮瑶池被同学征服的母亲凤玲公车系列辣文安琪儿", counts)
    print avg_transition_prob(u"公公操儿媳口述男女缠欢绵爱视频小萝莉被绑架侮辱图片漫画张大春征服王雪", counts)
    print avg_transition_prob(u"爸爸轻点胀死我了超级对对碰９７视频贞芸劫全文阅读目录鬼父２在线观看１至６集", counts)
    print avg_transition_prob(u"皇上不要了奴婢好疼爸爸不要塞了太涨了特级蔩片视频频日本变类态综艺", counts)
    print avg_transition_prob(u"失控的生理课全文阅读江疏影奶咪子全曝光美丽的新娘阿龟原文", counts)
    print avg_transition_prob(u"荒岛母子", counts)
    print avg_transition_prob(u"1000部啪啪视频辣妞范", counts)
    print "--------------------------------------------------"
    print avg_transition_prob(u"提示信息　－　长沙业主论坛－齐聚长沙社区，共策长沙楼市　－　Ｐｏｗｅｒｅｄ　ｂｙ　Ｄｉｓｃｕｚ！", counts)
    print avg_transition_prob(u"０７３１楼市通", counts)
    print avg_transition_prob(u"楼市楼盘　－　新房网　－　０７３１房产网　－　长沙购房必选", counts)
    print avg_transition_prob(u"长沙市投标监管网", counts)
    print avg_transition_prob(u"０７３１家装网　－　０７３１房产网", counts)
    print avg_transition_prob(u"票信通　发票４００６", counts)
    print avg_transition_prob(u"Ｅｘｃｅｐｔｉｏｎ", counts)
    print avg_transition_prob(u"刘强在办公室干小念", counts)
    print avg_transition_prob(u"１０２４基地手机直接看片，手机看片１０２４版免费，１０２４ｘｐ核工厂，ｘｐ３０３", counts)
    print avg_transition_prob(u"乱情家庭王丽霞章局长，乱情家庭王丽霞第３部，乱情家庭王丽霞尚莉华，王丽霞第三部第７章", counts)
    print avg_transition_prob(u"乡村爱情刘能操刘英", counts)
    print avg_transition_prob(u"管守祖的空间", counts)
    print avg_transition_prob(u"小学生作文大全｜小学生作文４００字", counts)
    print avg_transition_prob(u"帮助中心　－　小学生作文大全网（ｗｗｗ．ｚｉｘｕｅｋａｏｓｈｉ．ｎｅｔ）", counts)
    print avg_transition_prob(u"云霞小说网－更新快，无弹窗，绿色小说在线阅读", counts)



    print "--------------------------------------------------"

    print avg_transition_prob(u"食品安全和流感知识图片", counts)
    print avg_transition_prob(u"黄灯亮时3秒内过停止线", counts)

    fr = file("train.txt")
    fw = file("train2.txt", "w")
    for line in fr.readlines():
        score = avg_transition_prob(line.decode("utf-8"), counts)
        if(score>0.001):
            fw.write(line)
    fr.close()
    fw.close()

    fr = file("good.txt")
    fw = file("good2.txt", "w")
    for line in fr.readlines():
        score = avg_transition_prob(line.decode("utf-8"), counts)
        if (score <= 0.008):
            fw.write(line)
    fr.close()
    fw.close()

    # fr = file("train.txt")
    # fw = file("train2.txt", "w")
    # for line in fr.readlines():
    #     score = avg_transition_prob(line.decode("utf-8"), counts)
    #     if(score<0.001):
    #         fw.write(str(score)+"\t"+ line)
    # fr.close()
    # fw.close()
    #
    # fr = file("good.txt")
    # fw = file("good2.txt", "w")
    # for line in fr.readlines():
    #     score = avg_transition_prob(line.decode("utf-8"), counts)
    #     if (score >= 0.008):
    #         fw.write(str(score) + "\t" + line)
    # fr.close()
    # fw.close()


if __name__ == '__main__':
    test()


