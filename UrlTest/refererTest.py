# coding=utf-8

import urllib2
import random

url = "http://research.ruc.edu.cn/index.asp"
# html=urllib2.urlopen(url)
# print html.getcode()   #直接访问出现403错误


# 【追加header方法1】：req.add_header()
req = urllib2.Request(url)
req.add_header("Host", "research.ruc.edu.cn")
req.add_header("Referer", "https://m.sogou.com/web/searchList.jsp?uID=LpqMxosFqnXZ_mOy&v=5&from=index&w=1274&t=1520566303545&s_t=1520566309406&s_from=index&keyword=%E5%85%AD%E5%90%88%E7%8E%B0%E5%9C%BA%E5%BC%80%E5%A5%96%E7%BB%93%E6%9E%9C&pg=webSearchList&suguuid=b4a6bbc4-7a3a-403e-99b5-2ba08917846d&sugsuv=AAEQ0jOMHgAAAAqUKXUNaQ0AZAM%3D&sugtime=1520566309408")
req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 5.1; rv:37.0) Gecko/20100101 Firefox/37.0")

content = urllib2.urlopen(req)
# print content.read()

# 【追加header方法2】:req=urllib2.Request(url,headers)
# my_headers = {
#     "Host": "research.ruc.edu.cn",
#     "Referer": "https://m.sogou.com/web/searchList.jsp?uID=LpqMxosFqnXZ_mOy&v=5&from=index&w=1274&t=1520566303545&s_t=1520566309406&s_from=index&keyword=%E5%85%AD%E5%90%88%E7%8E%B0%E5%9C%BA%E5%BC%80%E5%A5%96%E7%BB%93%E6%9E%9C&pg=webSearchList&suguuid=b4a6bbc4-7a3a-403e-99b5-2ba08917846d&sugsuv=AAEQ0jOMHgAAAAqUKXUNaQ0AZAM%3D&sugtime=1520566309408",
#     "User-Agent": "Mozilla/5.0 (Windows NT 5.1; rv:37.0) Gecko/20100101 Firefox/37.0"
# }
# req = urllib2.Request(url, my_headers)

# content = urllib2.urlopen(req)
print content.read()

#
# # 【使用随机表头】，函数封装
# my_userAgent = [
#     "Mozilla/5.0 (Windows NT 5.1; rv:37.0) Gecko/20100101 Firefox/37.0",
#     "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:6.0) Gecko/20100101 Firefox/6.0",
#     "Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 5.1; Trident/4.0; GTB7.0)",
#     "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.1 (KHTML, like Gecko) Chrome/14.0.835.163 Safari/535.1",
#     "Opera/9.80 (Windows NT 6.1; U; zh-cn) Presto/2.9.168 Version/11.50"]
#
#
# def get_content(url, userAgent):
#     '''
#     @获取403禁止访问网页
#     '''
#     random_userAgent = random.choice(userAgent)
#
#     req = urllib2.Request(url)
#     req.add_header("User-Agent", random_userAgent)
#     req.add_header("Host", "blog.csdn.net")
#     req.add_header("Referer", "http://blog.csdn.net/")
#
#     content = urllib2.urlopen(req).read()
#     return content
#
#
# print get_content(url, my_userAgent)

