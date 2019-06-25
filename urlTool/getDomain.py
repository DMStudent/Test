# -*- coding: utf-8 -*-
#########################################################################
# File Name: /search/odin/util/hfgrep/getHost.py
# Author: wangyuan
# mail: wangyuan214159@sogou-inc.com
# Created Time: 2018年04月20日 星期五 17时52分54秒
#########################################################################
#!/bin/bash
import sys
import os
curDir = os.path.abspath(os.path.dirname(sys.argv[0]))
sys.path.append(curDir)

import UrlInfo


for url in sys.stdin:
    urlInfoObj = UrlInfo.UrlInfo(url)
    print url.strip() + "\t" + urlInfoObj.getDomain()

