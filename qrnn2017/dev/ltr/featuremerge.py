import sys

merge_path=""

#src_path="train/merge_data"
add_path=sys.argv[1]
src_path=sys.argv[2]

add_dic={}
file=open(add_path,"r")
while 1:
    line=file.readline()
    if not line:
        break
    line=line.strip()
    ts=line.split("\t")
    if len(ts)!=6:
        continue
    query= ts[0][20:]
    docid=ts[1][6:]
    feature=ts[5]
    key=query+"\t"+docid
    add_dic[key]=feature


file=open(src_path,"r")
while 1:
    line=file.readline()
    if not line:
        break
    line=line.strip()
    ts=line.split("\t")
    if len(ts)!=3:
        continue

    index=ts[2].replace(",","")
    docid=ts[1]
    feature=ts[0]
    query=feature[feature.index("#")+1:]
    feature=feature[:feature.index("#")]
    key=query+"\t"+docid
    if key not in add_dic:
        continue
    
    line=feature+" "+add_dic[key]+" #"+query+"\t"+docid+"\t"+index
    print line

#mergre_file=open(merge_path,"w")
