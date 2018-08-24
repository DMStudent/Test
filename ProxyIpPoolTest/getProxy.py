# -*- coding: utf-8 -*-
# Author: wangyuan
# mail: wyxidian@gmail.com
# Created Time: 
#!/bin/bash

import urllib2, time, datetime
from lxml import etree
from bs4 import BeautifulSoup

class getProxy():

    def __init__(self):
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36"
        self.header = {"User-Agent": self.user_agent}
        self.now = time.strftime("%Y-%m-%d")
        self.outname = "output/ip.lst"

        # 查看爬到的代理IP是否还能用
    def isAlive(self, ip_type, ip, port):
        proxy = {ip_type: ip + ':' + port}
        print proxy

        # 使用这个方式是全局方法。
        proxy_support = urllib2.ProxyHandler(proxy)
        opener = urllib2.build_opener(proxy_support)
        urllib2.install_opener(opener)
        # 使用代理访问腾讯官网，进行验证代理是否有效
        test_url = "http://www.whatismyip.com.tw/"
        req = urllib2.Request(test_url, headers=self.header)
        try:
            # timeout 设置为10，如果你不能忍受你的代理延时超过10，就修改timeout的数字
            resp = urllib2.urlopen(req, timeout=5)

            if resp.code == 200:
                print "work"
                return True
            else:
                print "not work"
                return False
        except:
            print "Not work"
            return False

    def getContent(self, num):
        nn_url = "http://www.xicidaili.com/nn/" + str(num)
        #国内高匿
        req = urllib2.Request(nn_url, headers=self.header)
        resp = urllib2.urlopen(req, timeout=10)
        content = resp.read()
        et = etree.HTML(content)
        result_even = et.xpath('//tr[@class=""]')
        result_odd = et.xpath('//tr[@class="odd"]')
        #因为网页源码中class 分开了奇偶两个class，所以使用lxml最方便的方式就是分开获取。
        #刚开始我使用一个方式获取，因而出现很多不对称的情况，估计是网站会经常修改源码，怕被其他爬虫的抓到
        #使用上面的方法可以不管网页怎么改，都可以抓到ip 和port
        ipLst = []
        for i in result_even:
            t1 = i.xpath("./td/text()")[:]
            ip_type = t1[5].strip().lower()
            ip = t1[0].strip()
            port = t1[1].strip()
            if len(ip)>0 and len(port)>0 and len(ip_type)>0:
                if self.isAlive(ip_type, ip, port):
                    ipLst.append(ip_type+"\t"+ip+":"+port)
        for i in result_odd:
            t2 = i.xpath("./td/text()")[:]
            ip_type = t2[5].strip()
            ip = t2[0].strip()
            port = t2[1].strip()
            if len(ip) > 0 and len(port) > 0 and len(ip_type) > 0:
                if self.isAlive(ip_type, ip, port):
                    ipLst.append(ip_type.lower() + "\t" + ip + ":" + port)

        return ipLst

    # 针对网站"https://ip.ihuan.me/"
    def getContentXiaohuan(self, num):
        nn_url = "https://ip.ihuan.me/?page=" + str(num)
        #国内高匿
        content = ""
        try:
            # timeout 设置为10，如果你不能忍受你的代理延时超过10，就修改timeout的数字
            req = urllib2.Request(nn_url, headers=self.header)
            resp = urllib2.urlopen(req, timeout=10)
            if resp.code == 200:
                content = resp.read()
            else:
                return []
        except:
            print "Not work"
            return []
        soup = BeautifulSoup(content, 'lxml')
        trs = soup.select("tbody > tr")
        ipLst = []
        for tr in trs:
            tds = tr.select('td')
            if len(tds) == 10:
                ip = tds[0].text.strip().encode('utf-8')
                port = tds[1].text.strip().encode('utf-8')
                ip_type = tds[4].text.strip()
                if len(ip_type) == 3:
                    ip_type = 'http'.encode('utf-8')
                else:
                    ip_type = 'https'.encode('utf-8')
                if len(ip) > 0 and len(port) > 0 and len(ip_type) > 0:
                    if self.isAlive(ip_type, ip, port):
                        ipLst.append(ip_type.lower() + "\t" + ip + ":" + port)

        return ipLst

        # 针对网站"http://www.66ip.cn/"
    def getContent66ip(self, num):
        nn_url = "http://www.66ip.cn/" + str(num)+".html"
        # 国内高匿
        content = ""
        try:
            # timeout 设置为10，如果你不能忍受你的代理延时超过10，就修改timeout的数字
            req = urllib2.Request(nn_url, headers=self.header)
            resp = urllib2.urlopen(req, timeout=10)
            if resp.code == 200:
                content = resp.read().decode('gb18030')
            else:
                return []
        except:
            print "Not work"
            return []
        soup = BeautifulSoup(content, 'lxml')

        trs = soup.select("table > tr")
        ipLst = []
        for tr in trs:
            tds = tr.select('td')
            if len(tds) == 5:
                ip = tds[0].text.strip().encode('utf-8')
                port = tds[1].text.strip().encode('utf-8')
                ip_type = "http"
                if len(ip) > 7 and len(port) > 0 and len(ip_type) > 0:
                    if self.isAlive(ip_type, ip, port):
                        ipLst.append(ip_type.lower() + "\t" + ip + ":" + port)
                    ip_type = "https"
                    if self.isAlive(ip_type, ip, port):
                        ipLst.append(ip_type.lower() + "\t" + ip + ":" + port)

        return ipLst

    def loop(self,page):
        f = file(self.outname, 'w')
        for i in range(1,page):
            print "page:" + str(i)
            ipLst = self.getContent66ip(i)
            for line in ipLst:
                if len(line)<1:
                    continue
                f.write(line + "\n")
        f.close()



if __name__ == "__main__":
    now = datetime.datetime.now()
    print "Start at %s" % now
    obj=getProxy()
    obj.loop(100)
