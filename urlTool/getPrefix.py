import sys
#url = "https://jingyan.baidu.com/article/5bbb5a1b1f4c7613eba1790d.html"

def getPrefix(url):
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
        print url + "\t" +  url[:pos] + types[type]


for url in sys.stdin:
    getPrefix(url.strip())
