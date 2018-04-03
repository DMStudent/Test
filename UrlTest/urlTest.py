# -*- encoding: utf-8 -*-
import urlparse
import re
url = 'http://www.6080.tv/'
if len(url)>2 and url.endswith("/") :
    url = url[:-1]
parsed_url = urlparse.urlparse(url)

path = parsed_url.path
pathList = path.split("/")
if len(pathList)>2:
    subpath = "/".join(pathList[:-1])
    prefix = parsed_url.scheme+"://"+parsed_url.netloc+subpath
    print prefix
else:
    prefix = parsed_url.scheme+"://"+parsed_url.netloc
    print prefix

# print subpath
# print path.split("/")
# print type(path.split())
# print parsed_url.path

s = u"圆圆你们这么大尺度赵又廷造么？>！"
#r = (re.sub("[\`\~\!\@\#\$\?\^\&\*\%\(\)\=\|\{\}\'\:\;\,\[\]\.\<\>\/\~\\！\？\>\！？《》]", "", s))
r = (re.sub(u"[？>！\`\~\!\@\#\$\?\^\&\*\%\(\)\=\|\{\}\'\:\;\,\[\]\.\<\>\/\~\\！\？\>]", "", s))
# r = (re.sub(u"", "", r))
# r = (re.sub(u"\u2000-\u206f", "", r))
# r = (re.sub(u"\u3000-\u303f", "", r))
# r = (re.sub(u"\uff00-\uffef", "", r))

print r

url = 'www.6080.tv'
ipUrlPattern = r"^http://(\d+\.\d+\.\d+\.\d+)(:\d+)?$"
m = re.match(ipUrlPattern, url)
def remove_punctuation(line):
    rule = re.compile(ur"[^a-zA-Z0-9\u4e00-\u9fa5]")
    line = rule.sub('',line)
    return line
line = u"１００８６．ｃｏｍ　｜　１００分的中国　１００分品质服务"
print remove_punctuation(line)





