from dev.config.final_config import *
from dev.model.final_model import graph_moudle

data_file_path = '/data/webrank/baidu_method.v1/demo'
model = graph_moudle.Model()
data_proc_fun = model.data_proc

f = open(data_file_path)
pairs = f.readlines()
q, qm, t, tm, g = data_proc_fun(pairs)

print(q[0])
print(qm[0])
print(t[0])
