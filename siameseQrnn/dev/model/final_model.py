# -- coding:utf-8 --
from dev.config.dynamic_setting import *
model_file = 'dev.model.model_' + str(ds.model)
graph_moudle = __import__(model_file, fromlist=['dev.model'])