#!/usr/bin/python
# -*- coding: UTF-8 -*-

import math
import pickle

accepted_chars = 'abcdefghijklmnopqrstuvwxyz '

trainList = []
wordSet = set(" ")
fr = file('train.txt')
for line in fr.readlines():
    line = line.decode("utf-8")
    trainList.append(line)
    for i in range(len(line)):
        wordSet.add(line[i])
fr.close()

fr = file('good.txt')
for line in fr.readlines():
    line = line.decode("utf-8")
    for i in range(len(line)):
        wordSet.add(line[i])
fr.close()

fr = file('bad.txt')
for line in fr.readlines():
    line = line.decode("utf-8")
    for i in range(len(line)):
        wordSet.add(line[i])
fr.close()



k = len(wordSet)
pos = dict([(char, idx) for idx, char in enumerate(wordSet)])



def normalize(line):
    """ Return only the subset of chars from accepted_chars.
    This helps keep the  model relatively small by ignoring punctuation,
    infrequenty symbols, etc. """
    return [c.lower() for c in line if c.lower() in accepted_chars]

def ngram(n, l):
    """ Return all n grams from l after normalizing """
    # filtered = normalize(l)
    for start in range(0, len(l) - n + 1):
        yield ''.join(l[start:start + n])

def train():
    """ Write a simple model as a pickle file """

    # Assume we have seen 10 of each character pair.  This acts as a kind of
    # prior or smoothing factor.  This way, if we see a character transition
    # live that we've never observed in the past, we won't assume the entire
    # string has 0 probability.
    counts = [[1 for i in xrange(k)] for i in xrange(k)]

    # Count transitions from big text file, taken
    # from http://norvig.com/spell-correct.html
    for line in trainList:
        for a, b in ngram(2, line):
            counts[pos[a]][pos[b]] += 1

    # Normalize the counts so that they become log probabilities.
    # We use log probabilities rather than straight probabilities to avoid
    # numeric underflow issues with long texts.
    # This contains a justification:
    # http://squarecog.wordpress.com/2009/01/10/dealing-with-underflow-in-joint-probability-calculations/
    for i, row in enumerate(counts):
        s = float(sum(row))
        for j in xrange(len(row)):
            row[j] = math.log(row[j] / s)

    # Find the probability of generating a few arbitrarily choosen good and
    # bad phrases.
    good_probs = [avg_transition_prob(l.decode("utf-8"), counts) for l in open('good.txt')]
    bad_probs = [avg_transition_prob(l.decode("utf-8"), counts) for l in open('bad.txt')]

    # Assert that we actually are capable of detecting the junk.
    # assert min(bad_probs) > max(good_probs)

    # And pick a threshold halfway between the worst good and best bad inputs.
    # thresh = (max(good_probs) + min(bad_probs)) / 2
    thresh = (sum(good_probs)/len(good_probs) + sum(bad_probs)/len(bad_probs)) / 2
    pickle.dump({'mat': counts, 'thresh': thresh, 'pos': pos}, open('model.pki', 'wb'))
    print "min(bad_probs)=" + str(min(bad_probs))
    print "max(good_probs)=" + str(max(good_probs))
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
    print avg_transition_prob(u"爸爸轻点胀死我了超级对对碰视频贞芸劫全文阅读目录鬼父在线观看至集", counts)
    print avg_transition_prob(u"皇上不要了奴婢好疼爸爸不要塞了太涨了特级蔩片视频频日本变类态综艺", counts)
    print avg_transition_prob(u"失控的生理课全文阅读江疏影奶咪子全曝光美丽的新娘阿龟原文", counts)
    print avg_transition_prob(u"白娜白波青春放纵全文娜美罗宾女帝人齐上阵嗯爸爸不要快点花核狼车诗晴喂奶书包", counts)

    print "--------------------------------------------------"

    print avg_transition_prob(u"食品安全和流感知识图片", counts)
    print avg_transition_prob(u"黄灯亮时秒内过停止线", counts)


    #
    # with open("model.txt", "w") as f:
    #     for i in xrange(k):
    #         for j in xrange(k):
    #             f.write(str(counts[i][j])+"\n")
    # with open("model.txt", "w") as f:
    #     f.write("{")
    #     for i in xrange(k):
    #         l = "{"
    #         for j in xrange(k):
    #             l = l+str(counts[i][j])+", "
    #         l = l[:-3]
    #         if i < k-1:
    #             l = l+"}, "+"\n"
    #         else:
    #             l = l + "}};"
    #         f.write(l)


def avg_transition_prob(l, log_prob_mat):
    """ Return the average transition prob from l through log_prob_mat. """
    log_prob = 0.0
    transition_ct = 0
    for a, b in ngram(2, l):
        log_prob += log_prob_mat[pos[a]][pos[b]]
        transition_ct += 1
    # The exponentiation translates from log probs to probs.
    return math.exp(log_prob / (transition_ct or 1))

if __name__ == '__main__':
    train()


