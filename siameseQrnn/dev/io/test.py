# -- coding:utf-8 --

import pickle as pkl

wdict_path="/search/odin/data/wangyuan/pycharmProjects/wenti/word_list/cnn_online_wdict.pkl"
wdict = pkl.load(open(wdict_path, 'rb'))
newWdict = {}
for key in wdict.keys():
    newKey = key.decode('gb18030').encode("utf-8")
    newWdict[newKey] = wdict[key]
print "start dump ..."
file = open('/search/odin/data/wangyuan/pycharmProjects/wenti/word_list/cnn_online_wdict_utf8.pkl', 'wb')
pkl.dump(newWdict, file)
print "fineshed"

