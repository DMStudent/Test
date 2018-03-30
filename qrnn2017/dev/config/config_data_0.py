from dev.config.config_base import *

class ConfigData(ConfigBase):
    def __init__(self):
        '''
        log_dir_path
        train_files
        '''
        self.output = '/search/odin/data/wangyuan/pycharmProjects/qrnn2017/model/0108/rnnatt01'
        super(ConfigData, self).__init__('', ldir=self.output)

        self.use_g = False
        train_dir = '/search/odin/data/wangyuan/keyword-data-2/'
        train_suffix = 10
        train_prefix = 'part'
	self.trainfnms = [train_dir + os.path.sep +
                          train_prefix + '-%05d'%i for i in range(0, train_suffix)]
