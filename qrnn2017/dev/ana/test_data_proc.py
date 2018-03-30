from dev.config.final_config import *
from dev.model.final_model import graph_moudle

data_file_path = '/search/odin/data/session-data-v3/demo'
#data_file_path = '/search/odin/data/strict_anchor_data/demo'
model = graph_moudle.Model()
data_proc_fun = model.data_proc

f = open(data_file_path)
pairs = f.readlines()
#q, qp, qt, qm, t, tp, tt, tm, url, k, km = data_proc_fun(pairs[:args.batchsize])
q, qm, t, tm, url, k, km, g = data_proc_fun(pairs[:args.batchsize])
id = 0
print(pairs[id])
print('----query ----')
print(q[id])
print(qm[id])
#print(qp[id])
#print(qt[id])
print('---- title ----')
print(t[id])
print(tm[id])
#print(tp[id])
#print(tt[id])
print(url[id])
print('----keyword----')
print(k[id])
print(g[id])
