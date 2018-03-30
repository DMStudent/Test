# -- coding:utf-8 --
from dev.config.dynamic_setting import *
data_config_file = 'dev.config.config_data_' + str(ds.data)
cd = __import__(data_config_file, fromlist=['dev.config'])

class ConfigGPU(cd.ConfigData):
    def __init__(self):
        super(ConfigGPU, self).__init__()
        self.data_config_file = data_config_file
        # 8 workers
        self.workerhosts = ['localhost:' + str(i) for i in range(2513, 2521)]  # from 2213
        self.numworker = len(self.workerhosts)
        self.pshosts = ['localhost:' + str(i) for i in range(2505, 2513)]  # up to 2212



