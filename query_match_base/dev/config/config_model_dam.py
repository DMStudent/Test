from dev.config.dynamic_setting import *
gpu_config_file = 'dev.config.config_gpu_' + str(ds.gpu)
cg = __import__(gpu_config_file, fromlist=['dev.config'])

class ConfigModel(cg.ConfigGPU):
    def __init__(self):
        super(ConfigModel, self).__init__()
        self.gpu_config_file = gpu_config_file

        self.batchsize = 400
        self.showperbatch = int(120000 / self.batchsize)  # 120000
        self.ckptperbatch = int(19200000 / self.batchsize)  # 1.8e

        self.lr = 0.0005  # Adam learning rate 0.0004 - 0.0012 for 8 GPUs, 0.001 - 0.0015 for 4 GPUs,
                          # RMSProp = 0.001 as default
        self.l2 = 1e-8
        self.keep_prob = 0.6
        self.pair_margin = 0.1  # 0.05 - 0.2, estimate the average margin in original data feature

        self.use_intra = True
        self.num_layers = 1
        self.units = 512
        self.train_embd = True



