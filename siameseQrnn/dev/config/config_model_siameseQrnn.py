# -- coding:utf-8 --
import os
from dev.config.dynamic_setting import *

gpu_config_file = 'dev.config.config_gpu_' + str(ds.gpu)
cg = __import__(gpu_config_file, fromlist=['dev.config'])

class ConfigModel(cg.ConfigGPU):
    def __init__(self):
        super(ConfigModel, self).__init__()
        self.gpu_config_file = gpu_config_file

        self.docid_path = self.word_embd_dir + os.path.sep + 'docid_new_map.dict'
        self.docid_size = 110000
        self.n_keyword = 10

        self.batchsize = 400
        self.showperbatch = int(20000 / self.batchsize)  # 120000
        self.ckptperbatch = int(10000000 / self.batchsize)  # 1.8e
        self.testperbatch = int(1900000 / self.batchsize)
        self.use_g = True
        if self.use_g:
            self.lr = 0.0006  # Adam learning rate 0.0004 - 0.0012 for 8 GPUs, 0.001 - 0.0015 for 4 GPUs,
                          # RMSProp = 0.001 as default
        else:
            self.lr = 0.0005

        self.l2 = 1e-8
        self.use_grad_clip = False
        self.keep_prob = 0.6
        self.pair_margin = 0.1  # 0.05 - 0.2, estimate the average margin in original data feature

        self.train_embd = False
