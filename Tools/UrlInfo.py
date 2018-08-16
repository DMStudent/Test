# -*- coding:utf-8 -*-
import re
import os,sys

class UrlInfo:
    host_head = 0
    host_tail = 0
    domain_head = -1
    domain_tail = 0
    url = None  
    isIp = False

    def getSite(self):  
        result = self.url[ self.host_head : self.host_tail ]
        if result.endswith("/"):
            return result
        else:
            return result + "/"

    def getDomain(self):
        if self.isIp:
            toReturn_str = self.url
            if toReturn_str.startswith("http://"):
                start_loc = 7
            elif toReturn_str.startswith("https://"):
                start_loc = 8
            else:
                start_loc = 0
            toReturn_str = self.url[ start_loc : ]

            loc = toReturn_str.find("/")
            if loc > -1:
                return toReturn_str[ : loc ]
            else:
                return toReturn_str

        else:
            return self.url[ self.domain_head : self.domain_tail ]

    def isTop(self):
        if self.url == "http://" + self.getDomain() + "/":
            return True
        if self.url == "http://" + self.getDomain():
            return True
        if self.url == "http://www." + self.getDomain() + "/":
            return True
        if self.url == "http://www." + self.getDomain() :
            return True

        if self.url == "https://" + self.getDomain() + "/":
            return True
        if self.url == "https://" + self.getDomain():
            return True
        if self.url == "https://www." + self.getDomain() + "/":
            return True
        if self.url == "https://www." + self.getDomain() :
            return True

        return self.isTopPage()

    def isTopPage(self):
        idx_path_start = self.url.find( '/' )
        if idx_path_start <= 0: return False
        if ( idx_path_start+1 == len(self.url) ): return True
        for ch in self.url:
            if ch in [ '?', '/', '=' ]:
                return False
        path = self.url[ idx_path_start+1 : ]
        if len(path) > 15: return False
        if ( path.startswith("index.") ): return True
        if ( path.startswith("default.") ): return True
        if ( path.startswith("home.") ): return True
        if ( path.startswith("homepage.") ): return True
        if ( path.startswith("main.") ): return True
        if ( path == "home") : return True
        return False


    def isSite(self):
        site = self.getSite()
        if self.url.endswith("/"):
            if self.url ==  site:
                return True
        else:
            if self.url + '/' == site:
                return True
        return False

    def isHostNohttp(self):
        site = self.getSite()
        if ('http://' + self.url + '/'  == site)  or ("https://" + self.url + '/' == site):
            return True
        else:
            return False

    def in_domain_set(self,url,length,domain_set,domain_set_length,start):      
        begin = 0
        end = domain_set_length - 1
        mid = -1
        #ret = -1
        i = 2

        while begin <= end:
            mid = (begin+end)/2
            #ret = url[start+1] - domain_set[mid][1]
            if url[start+1] > domain_set[mid][1]:
            #if ret > 0:
                begin = mid + 1
            elif url[start+1] < domain_set[mid][1]:
            #elif ret < 0:
                end = mid - 1
            else:
                #ret = url[start] - domain_set[mid][0]
                if url[start] > domain_set[mid][0]:
                #if ret > 0:
                    begin = mid + 1
                elif url[start] < domain_set[mid][0]:
                    end = mid - 1;
                else:
                    break       
        if begin > end:
            return 0

        while i<length and url[start+i]==domain_set[mid][i]:
            i += 1
        if i == length:
            return 1
        else:
            return 0    

    def in_second_domain_set(self,url,length,start):
        domain2 = list( ("ha", "hb", "ac", "sc", "gd", "sd", "he", "ah", "qh",
                        "sh", "hi", "bj", "fj", "tj", "xj", "zj", "hk", "hl",
                        "jl", "nm", "hn", "ln", "sn", "yn", "co", "mo", "cq",
                        "gs", "js", "tw", "gx", "jx", "nx", "sx", "gz", "xz") ) # domain in china & ac & co
        domain3 = list( ("cat", "edu", "net", "biz", "mil", "int", "com", "gov", "org", "pro") )
        domain4 = list( ("name", "aero", "info", "coop", "jobs", "mobi", "arpa") )
        domain6 = list( ("travel", "museum") )
        domain_set = list( (None,None,domain2,domain3,domain4,None,domain6) )
        domain_set_length = (0,0,len(domain2),len(domain3),len(domain4),0,len(domain6))

        if length in (2,3,4,6):
            x = self.in_domain_set(url, length, domain_set[length], domain_set_length[length], start)
            return x
        else:
                    return 0

    def in_top_domain_set(self,url,length,start):
        domain2 = list( ("ac", "co") ) # ac & co
        domain3 = list( ("cat", "edu", "net", "biz", "mil", "int", "com", "gov", "org", "pro") )
        domain4 = list( ("name", "aero", "info", "coop", "jobs", "mobi", "arpa") )
        domain6 = list( ("travel", "museum") )
        domain_set = list( (None,None,domain2,domain3,domain4,None,domain6) )
        domain_set_length = (0,0,len(domain2),len(domain3),len(domain4),0,len(domain6))

        if length in (2,3,4,6):
            return self.in_domain_set(url, length, domain_set[length], domain_set_length[length], start)
        else:
                    return False

    def __init__(self):               
        pass

    def process(self,url): 
        if url is None or url == "": return None
        url = url.strip() 

        self.host_head = 0
        self.host_tail = 0
        self.domain_head = -1
        self.domain_tail = 0
        self.url = url

        if re.match( r"^(http[s]?://)?\d+.\d+.\d+.\d/|^(http[s]?://)?\d+.\d+.\d+.\d$", url ) is not None: # that is IP address
            self.isIp = True
        else:
            self.isIp = False

        domain_pre_head = self.domain_head
        domain_post_head = self.domain_head
        self.domain_tail = self.domain_head
        find_domain = False
        deal_domain = False
        i = 0
        while i < len(url):
        #for i in range( len(url) ):
            if url[i] == ".":
                deal_domain = True
            elif url[i] == "/":
                break
            elif url[i] == ":":
                if ( (i+2)<len(url) ) and ( url[i+1]=="/" ) and ( url[i+2]=="/" ):
                    i += 2
                    self.domain_head = i
                    domain_pre_head = i
                    domain_post_head = i
                    self.domain_tail = i
                    i += 1
                    continue;
                elif not find_domain:
                    deal_domain = True
                    find_domain = True
            if deal_domain: 
                domain_pre_head = self.domain_head
                self.domain_head = domain_post_head
                domain_post_head = self.domain_tail
                self.domain_tail = i
                deal_domain = False
            i += 1
        self.host_tail = i
        
        if not find_domain:
            domain_pre_head = self.domain_head
            self.domain_head = domain_post_head
            domain_post_head = self.domain_tail
            self.domain_tail = i
        if ( self.in_second_domain_set(url,domain_post_head-self.domain_head-1,self.domain_head+1) > 0 ) and ( self.in_top_domain_set(url, self.domain_tail-domain_post_head-1, domain_post_head+1) == 0 ):
            self.domain_head = domain_pre_head
        self.domain_head += 1

def main():
    ui = UrlInfo()
    for line in open( sys.argv[1] ):
        try:
            line = line.decode("utf8").strip()
        except:
            try:
                line = line.decode("gbk").strip()
            except:
                continue
    
        splits = line.split()
        url = splits[0]
        ui.process(url)
        print ui.getDomain()

        #print ("http://www." + domain + "/").encode('utf8')
        #print ("http://" + domain + "/").encode('utf8')
        #print ui.getDomain()

if __name__ == "__main__":
    main()

