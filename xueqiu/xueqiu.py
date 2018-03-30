# _*_ coding: utf-8 _*_
from __future__ import print_function
import re
import time
import requests
import numpy as np
import json
import io
# from pandas import Series, DataFrame

import requests
from bs4 import BeautifulSoup
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}




def getInfoFromXuqiu(group_id):
    result = {}
    # group_id = 'ZH846956'
    # url='http://xueqiu.com/stock/quotep.json?stockid=1000861%2C1000084%2C1000823%2C1000014%2C1000772%2C1000931%2C1001306%2C1000870%2C1000928%2C1000857'
    url='http://xueqiu.com/stock/quotep.json?stockid='
    referer = 'http://xueqiu.com/P/'+group_id

    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}
    r = requests.get(referer, headers = headers)
    content = r.text
    soup = BeautifulSoup(content, 'lxml')
    divs = soup.find_all(class_ = 'stock fn-clear no-tooltip')
    data_sid_all = ''
    for div in divs:
        joke = div.span.get_text()
        data_sid = div.attrs['data-sid']
        data_sid_all = data_sid_all + '2C' + data_sid + '%'
        stock_name = div.find_all(class_='name')[0].get_text()
        stock_symbol = div.find_all(class_='price')[0].get_text()
        percentage = div.find_all(class_='stock-weight weight')[0].get_text()
        stock_str = stock_name + "\t" + unicode(percentage) + '\t' + data_sid
        result[stock_symbol] = stock_str
    data_sid_all = data_sid_all[2:-1]
    url = url + data_sid_all
    # print (url)
    # http://xueqiu.com/P/ZH1088307
    headers = {
        'User-agent' : 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.221 Safari/537.36 SE 2.X MetaSr 1.0',
        'Host' : 'xueqiu.com',
        'Accept' : '*/*',
        'Accept-Language' : 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
        'Accept-Encoding' : 'gzip, deflate, br',
        'Referer' : referer,
        'Cookie' : 'device_id=99f3959a94c120ca967a3fe66227daf8; s=ef17e1t0sf; xq_a_token=71158a691fbe727b322a141a00aeaa76a9341145; xqat=71158a691fbe727b322a141a00aeaa76a9341145; xq_r_token=e32ca1d4946286a99c4864ea5860bcf379072cd0; xq_token_expire=Wed%20Feb%2007%202018%2016%3A25%3A35%20GMT%2B0800%20(CST); xq_is_login=1; u=3264356413; bid=1a64018b0b08f5dcd5c16e984904c5f9_jcd3739d; __utma=1.1685280481.1515831803.1515831803.1515831803.1; __utmb=1.2.10.1515831803; __utmc=1; __utmz=1.1515831803.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); Hm_lvt_1db88642e346389874251b5a1eded6e3=1515735281,1515816814; Hm_lpvt_1db88642e346389874251b5a1eded6e3=1515832772',
        'DNT' : '1',
        'Connection' : 'keep-alive',
        }


    r = requests.get(url = url,headers = headers)

    try:
        rebalancing_histories = r.json()
        if r.status_code == 200:
            try:
                # print('正在读取')
                for key in rebalancing_histories.keys():
                    stock_symbol = rebalancing_histories[key]['symbol']
                    price = rebalancing_histories[key]['current']
                    result[stock_symbol] = unicode(price) + "\t" + result[stock_symbol]
                    # print (stock_symbol+'\t'+result[stock_symbol])
                time.sleep(1)
            except :
                pass

    except:
        pass
    return result

def genTopK(id_info,n):
    stock_rank = {}
    for group_id,info in id_info:
        for key in info.keys():
            stock_name = info[key].split('\t')[1]
            if stock_rank.has_key(stock_name):
                stock_rank[stock_name].append('http://xueqiu.com/P/'+group_id)
            else:
                stock_rank[stock_name] = ['http://xueqiu.com/P/'+group_id]
    stock_rank = sorted(stock_rank.items(), key=lambda d: len(d[1]), reverse=True)
    stock_dict = {}
    for i in stock_rank[:n]:
        item = {}
        stock_name = i[0]
        item['num'] = len(i[1])
        item['groups'] = ' '.join(i[1])
        stock_dict[stock_name] = item
    return stock_dict


if __name__ == '__main__':
    group_if_list = 'ZH1088307,ZH1239563,ZH1186266,ZH1214202,ZH1218679,ZH1218677,ZH796048,ZH1004909,ZH013874,ZH1235083,ZH907732,ZH563983,ZH846956,ZH992971,ZH002191,ZH912497,ZH140459,ZH1004904,ZH739090,ZH201809,ZH764846,ZH1129503,ZH1092959,ZH864735,ZH1218280'.split(',')
    # group_if_list = 'ZH1088307,ZH1239563'.split(',')
    id_info = []
    for group_id in group_if_list:
        info = getInfoFromXuqiu(group_id)
        id_info.append([group_id,info])
    topK = genTopK(id_info,len(id_info))
    for key, value in topK.items():
        print (key,value['num'],value['groups'])
    json_str = json.dumps(topK, ensure_ascii=False)
    # Writing JSON data
    with io.open('xueqiu.json', 'w', encoding='utf-8') as f:
        f.write(json_str)

    # # Reading data back
    # with open('data.json', 'r') as f:
    #     data = json.load(f)





# # i = 0
# comment_num = 1
# commentList = []
# while True:
#     if comment_num>20000:
#         break;
#     # if i==1:     #爬热门评论
#     #     r = requests.get(url = url.format(i),headers = headers)
#     #     comment_page = r.json()[1]['card_group']
#     # else:
#     #     r = requests.get(url = url.format(i),headers = headers)
#     #     comment_page = r.json()[0]['card_group']
#
#     r = requests.get(url = url.format(i),headers = headers)  #爬时间排序评论
#     try:
#         comment_page = r.json()['data']
#         if r.status_code ==200:
#             try:
#                 print('正在读取第 %s 页评论：' % i)
#                 for j in range(0,len(comment_page)):
#                     print('第 %s 条评论' % comment_num)
#                     user = comment_page[j]
#                     comment_id = user['user']['id']
#                     # print(comment_id)
#                     user_name = user['user']['screen_name']
#                     # print(user_name)
#                     created_at = user['created_at']
#                     # print(created_at)
#                     text = re.sub('<.*?>|回复<.*?>:|[\U00010000-\U0010ffff]|[\uD800-\uDBFF][\uDC00-\uDFFF]','',user['text'])
#                     if len(text)<1:
#                         text = "#"
#                     # print(text)
#                     likenum = user['like_counts']
#                     # print(likenum)
#                     source = re.sub('[\U00010000-\U0010ffff]|[\uD800-\uDBFF][\uDC00-\uDFFF]','',user['source'])
#                     if len(source)<1:
#                         source = "#"
#                     # print(source + '\r\n')
#                     comment_num+=1
#                     singleComment = [comment_id,user_name,created_at,text,likenum,source]
#                     commentList.append(singleComment)
#
#
#                 i+=1
#                 time.sleep(3)
#             except:
#                 i+1
#                 pass
#         else:
#             break
#     except:
#         pass
# # data = {'comment_id': [comment_id],
# #         'screen_name': [user_name],
# #         'created_at': [created_at],
# #         'text': [text],
# #         'likenum': [likenum],
# #         'source': [source]}
# comment = DataFrame(commentList,columns=['comment_id','user_name','created_at','text','likenum','source'])
# comment.to_csv("data/comment.csv", encoding='utf-8')
