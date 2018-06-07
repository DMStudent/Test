# -*- coding: utf-8 -*-
# Author: wangyuan
# mail: wyxidian@gmail.com
# Created Time: 
#!/bin/bash




def read_new_word(filename):
    print "read new word..."
    f = file(filename, 'r')
    word = {}
    for line in f.readlines():
        lines = line.strip().split(",")
        word[lines[0].decode("utf-8")] = lines[1]

    f.close()
    return word

def cal_idf(filename, word):
    print "calculate idf..."
    f = file(filename, 'r')
    idf = {}
    count = 0
    for w in word.keys():
        idf[w] = 1
    for line in f.readlines():
        if count % 1000 == 0:
            print count
        count = count + 1
        line = line.decode("utf-8")
        for w in word.keys():
            if line.find(w) >= 0:
                idf[w] = idf[w]+1

    f.close()

    for w in idf.keys():
        idf[w] = float(idf[w])/count
    return idf

def filter_good_word(inputdir, outputdir, word, idf):
    print "filter..."
    f = open(inputdir, 'r')  # 读取文章
    good_text = f.read().decode("utf-8")  # 读取为一个字符串
    f.close()
    fw = file(outputdir, 'w')
    count_all = 0
    count_bad = 0
    for w in word.keys():
        if good_text.find(w) < 0:
            count_bad = count_bad + 1
            line = w + "\t" + word[w] + "\t" + str(idf[w]) + "\t" + str(int(word[w])*idf[w]) + "\n"
            fw.write(line.encode("utf-8"))
        if count_all % 1000 == 0:
            print count_all, count_bad
        count_all = count_all + 1
    fw.close()


if __name__ == '__main__':
    word_dir = "bad_good.word"
    word = read_new_word(word_dir)
    bad_file_dir = "/search/odin/data/wangyuan/pycharmProjects/sex-cnn/sexText/sex-long/sex.raw2"
    idf = cal_idf(bad_file_dir, word)

    inputdir = '/search/odin/data/wangyuan/pycharmProjects/sex-cnn/sexText/sex-long/good.raw2'
    outputdir = "bad.word"
    filter_good_word(inputdir, outputdir, word, idf)






    # f = open('/search/odin/data/wangyuan/pycharmProjects/sex-cnn/sexText/sex-long/good.raw2', 'r')  # 读取文章
    # good_text = f.read().decode("utf-8")  # 读取为一个字符串
    # f.close()
    # f = file("bad_good.word",'r')
    # fw = file("bad.word",'w')
    # count_all = 0
    # count_bad = 0
    # for line in f.readlines():
    #     lines = line.split(",")
    #     if good_text.find(lines[0].decode("utf-8"))<0:
    #         count_bad = count_bad + 1
    #         fw.write(line)
    #     if count_all%1000 == 0:
    #         print count_all,count_bad
    #     count_all = count_all + 1
    # f.close()
    # fw.close()
    # print "done"

