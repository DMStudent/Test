# _*_ coding: utf-8 _*_
from __future__ import print_function
import re
import time
import requests
import numpy as np
import json
import io
import xml.etree.ElementTree as ET
# from pandas import Series, DataFrame

import requests
from bs4 import BeautifulSoup
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}


result = {}
# group_id = 'ZH846956'
# url='http://xueqiu.com/stock/quotep.json?stockid=1000861%2C1000084%2C1000823%2C1000014%2C1000772%2C1000931%2C1001306%2C1000870%2C1000928%2C1000857'
url='https://www.sogou.com/sogou?query=王小川&wxc=on'

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}
r = requests.get(url, headers = headers)
content = r.text
soup = BeautifulSoup(content, 'lxml')
divs = soup.find_all(class_ = 'sogou-debug')
for div in divs:
    if div.span == None:
        continue

    idx = unicode(div.span.text).index(u'处理有的展现XML')
    dnn1 = BeautifulSoup(unicode(div.span.text)[6:idx], 'lxml').find('rank').attrs['dnn1']
    title = BeautifulSoup(unicode(div.span.text)[6:idx], 'lxml').find('title').text

    # for child in div.span.children:
    #     if unicode(child).strip().startswith("<rank F"):
    #         print (unicode(child))
    #     elif unicode(child).strip().startswith("<title> "):
    #         print(unicode(child))
