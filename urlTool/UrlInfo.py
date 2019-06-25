# -*- coding: utf-8 -*-
#########################################################################
# File Name: /search/odin/util/hfgrep/UrlInfo.py
# Author: wangyuan
# mail: wangyuan214159@sogou-inc.com
# Created Time: 2018年04月20日 星期五 17时50分48秒
#########################################################################
#!/bin/bash
#!/usr/bin/env python
""" generated source for module UrlInfo """
import re
class UrlInfo(object):
    """ generated source for class UrlInfo """
    url = ""
    b = []
    host_head = 0
    host_tail = 0
    domain_head = -1
    domain_tail = 0
    FIRST_PAGE_NAME = ["index", "default", "main", "home", "homepage"]
    ipUrlPattern = r"^http://(\d+\.\d+\.\d+\.\d+)(:\d+)?$"

    def getHostHead(self):
        """ generated source for method getHostHead """
        return self.host_head

    def getHostTail(self):
        """ generated source for method getHostTail """
        return self.host_tail

    def getDomainHead(self):
        """ generated source for method getDomainHead """
        return self.domain_head

    def getDomainTail(self):
        """ generated source for method getDomainTail """
        return self.domain_tail

    def getDomainSign(self, var1, var2, var3, var4):
        """ generated source for method getDomainSign """
        var1.reset()
        var1.update(self.b, self.domain_head, self.domain_tail - self.domain_head)
        return var1.digest(var2, var3, var4)

    def getHostSign(self, var1, var2, var3, var4):
        """ generated source for method getHostSign """
        var1.reset()
        var1.update(self.b, self.host_head, self.host_tail - self.host_head)
        return var1.digest(var2, var3, var4)

    def getUrlSign(self, var1, var2, var3, var4):
        """ generated source for method getUrlSign """
        var1.reset()
        var1.update(self.b)
        return var1.digest(var2, var3, var4)

    def __init__(self, var1):
        """ generated source for method __init__ """
        self.url = var1
        self.b = [ord(i) for i in var1]
        var2 = self.domain_head
        var3 = self.domain_head
        self.domain_tail = self.domain_head
        var4 = False
        var5 = False
        var6 = 0
        while var6 < len(self.b):
            if self.b[var6] == 46:
                var5 = True
            else:
                if self.b[var6] == 47:
                    break
                if self.b[var6] == 58:
                    if var6 + 2 < len(self.b) and self.b[var6 + 1] == 47 and self.b[var6 + 2] == 47:
                        var6 = var6 + 2
                        self.host_head = var6
                        self.domain_head = var6
                        var2 = var6
                        var3 = var6
                        self.domain_tail = var6
                        var6 = var6 + 1
                        continue
                    if not var4:
                        var5 = True
                        var4 = True
            if var5:
                var2 = self.domain_head
                self.domain_head = var3
                var3 = self.domain_tail
                self.domain_tail = var6
                var5 = False
            var6 = var6 + 1
        self.host_tail = var6
        if not var4:
            var2 = self.domain_head
            self.domain_head = var3
            var3 = self.domain_tail
            self.domain_tail = var6
        if self.in_second_domain_set(self.b, var3 - self.domain_head - 1, self.domain_head + 1) > 0 and self.in_top_domain_set(self.b, self.domain_tail - var3 - 1, var3 + 1) == 0:
            self.domain_head = var2
        self.domain_head = self.domain_head + 1
        self.host_head = self.host_head + 1

    def getPrefix(self):
        url = self.url
        types = ["?", "/"]
        pos = -1
        type = -1;

        for i in range(len(types)):
            if (url[-1] == types[i]):
                url = url[:-1]
            _pos = url.rfind(types[i])
            if (_pos > pos):
                pos = _pos
                type = i
        if (pos >= 8 and type >= 0):
            return url[:pos] + types[type]
        else:
            return self.url
    def in_top_domain_set(self, var1, var2, var3):
        """ generated source for method in_top_domain_set """
        var4 = ["ac", "co", "cn"]
        var5 = ["cat", "edu", "net", "biz", "mil", "int", "com", "gov", "org", "pro"]
        var6 = ["name", "aero", "info", "coop", "jobs", "mobi", "arpa"]
        var7 = ["travel", "museum"]
        var8 = [None, None, var4, var5, var6, None, var7]
        var9 = [0, 0, len(var4), len(var5), len(var6), 0, len(var7)]
        if var2==2:
            return self.in_domain_set(var1, var2, var8[var2], var9[var2], var3)
        elif var2==3:
            return self.in_domain_set(var1, var2, var8[var2], var9[var2], var3)
        elif var2==4:
            return self.in_domain_set(var1, var2, var8[var2], var9[var2], var3)
        elif var2==6:
            return self.in_domain_set(var1, var2, var8[var2], var9[var2], var3)
        elif var2==5:
            pass
        else:
            return 0

    def in_second_domain_set(self, var1, var2, var3):
        """ generated source for method in_second_domain_set """
        var4 = ["ha", "hb", "ac", "sc", "gd", "sd", "he", "ah", "qh", "sh", "hi", "bj", "fj", "tj", "xj", "zj", "hk", "hl", "jl", "nm", "hn", "ln", "sn", "yn", "co", "mo", "cq", "gs", "js", "tw", "gx", "jx", "nx", "sx", "gz", "xz"]
        var5 = ["cat", "edu", "net", "biz", "mil", "int", "com", "gov", "org", "pro"]
        var6 = ["name", "aero", "info", "coop", "jobs", "mobi", "arpa"]
        var7 = ["travel", "museum"]
        var8 = [None, None, var4, var5, var6, None, var7]
        var9 = [0, 0, len(var4), len(var5), len(var6), 0, len(var7)]
        if var2==2:
            return self.in_domain_set(var1, var2, var8[var2], var9[var2], var3)
        elif var2==3:
            return self.in_domain_set(var1, var2, var8[var2], var9[var2], var3)
        elif var2==4:
            return self.in_domain_set(var1, var2, var8[var2], var9[var2], var3)
        elif var2==6:
            return self.in_domain_set(var1, var2, var8[var2], var9[var2], var3)
        elif var2==5:
            pass
        else:
            return 0

    def in_domain_set(self, var1, var2, var3, var4, var5):
        """ generated source for method in_domain_set """
        var6 = 0
        var7 = var4 - 1
        var8 = -1
        var10 = 2
        while var6 <= var7:
            var8 = (var6 + var7) / 2
            var9 = var1[var5 + 1] - ord(var3[var8][1])
            if var9 > 0:
                var6 = var8 + 1
            elif var9 < 0:
                var7 = var8 - 1
            else:
                var9 = var1[var5] - ord(var3[var8][0])
                if var9 > 0:
                    var6 = var8 + 1
                else:
                    if var9 >= 0:
                        break
                    var7 = var8 - 1
        if var6 > var7:
            return 0
        else:
            while var10 < var2 and var1[var5 + var10] == ord(var3[var8][var10]):
                var10 = var10 + 1
            return 1 if var10 == var2 else 0


    def getDomain(self):
        """ generated source for method getDomain """
        # ui = UrlInfo(url)
        # urlBytes = [ord(i) for i in url]
        host = self.url[self.host_head:self.host_tail]
        domain = self.url[self.domain_head:self.domain_tail]

        # domain = url[ui.getDomainHead():ui.getDomainTail()]
        m = re.match(self.ipUrlPattern, host)
        if m:
            domain = m.group(1)
        return domain


    def getHome(self):
        """ generated source for method getHome """
        # host = getHostWithPort(url)
        # ui = UrlInfo(url)
        # host = url[ui.getHostHead():ui.getHostTail()]
        host = self.url[self.host_head:self.host_tail]
        return "http://" + host + "/"


    def getHost(self):
        """ generated source for method getHost """
        host = self.url[self.host_head:self.host_tail]
        return host


    @classmethod
    def isTopHomePage(self):
        """ generated source for method isTopHomePage """
        domain = self.getDomain()
        if self.url == "http://" + domain + "/":
            return True
        if self.url == "http://www." + domain + "/":
            return True
        if self.url == "http://" + domain:
            return True
        if self.url == "http://www." + domain:
            return True
        return False

    @classmethod
    def isHomePage(self):
        """ generated source for method isHomePage """
        home = self.getHome(self.url)
        if home == self.url:
            return True
        if home == self.url + '/':
            return True
        return False


    @classmethod
    def main(cls, var0):
        """ generated source for method main """
        var1 = "http://sports.163.com/10/0506/07/6601D42S00051C8V.html"
        var2 = UrlInfo(var1)
        var3 = var1[var2.host_head:var2.host_tail]
        print var1
        print var3
        var4 = var1[var2.domain_head:var2.domain_tail]
        print var4
        url = "https://www.github.com/wanpark/hadoop-hbase-streaming/tree/master/src/org/childtv/hadoop/hbase/mapred"
        print "Url:" + url
        urlInfoObj = UrlInfo(url)
        print "Domain:" + urlInfoObj.getDomain()
        print "Home:" + urlInfoObj.getHome()
        print "Host:" + urlInfoObj.getHost()
        print "Prefix:" + urlInfoObj.getPrefix()

        url = "https://abc.gov.cn/"
        print "Url:" + url
        urlInfoObj = UrlInfo(url)
        print "Domain:" + urlInfoObj.getDomain()
        print "Home:" + urlInfoObj.getHome()
        print "Host:" + urlInfoObj.getHost()
        print "Prefix:" + urlInfoObj.getPrefix()


if __name__ == '__main__':
    import sys
    UrlInfo.main(sys.argv)


