# -*- coding: utf-8 -*-
# Author: wangyuan
# mail: wyxidian@gmail.com
# Created Time: 
#!/bin/bash

import os
import urllib2
import re


class WebSearchXml():
    prefix = "http://www.sogou.com.inner/websearch/xml/xml4union.jsp?query="
    page = "&page={0}&ie=utf8"
    pattern_zh = re.compile(u"[\u4E00-\u9FA5]+")
    pattern_item = re.compile(r'<item>\s*?<title>(.*?)</title>\s*?<summary>(.*?)</summary>\s*?<link>(.*?)</link>')
    pattern_hint = re.compile(r'<hintword>(.*?)</hintword>')
    pattern_space = re.compile('\s+')
    def getItems(self, query, page=1):
        url = self.prefix + query + self.page.format(page)
        content = urllib2.urlopen(url).read().decode('gb18030').encode('utf-8')
        match = self.pattern_item.findall(content)
        ret = []
        if match:
            for item in match:
                if len(item) == 3:
                    title, summary, link = item
                ret.append(link + "\t" + title + "\t" + summary)
        return ret

    def getHint(self, query):
        url = self.prefix + query + self.page
        content = urllib2.urlopen(url).read().decode('gb18030').encode('utf-8')
        match = self.pattern_hint.findall(content)
        hints = []
        if match:
            for item in match:
                if len(item) > 0:
                    hints.append(item.strip())
        return hints


if __name__ == '__main__':
    query = "百度"
    hints = WebSearchXml().getHint(query)
    print " ".join(hints)
    print "-------------------------------------"
    items = WebSearchXml().getItems(query)
    print "\n".join(items)