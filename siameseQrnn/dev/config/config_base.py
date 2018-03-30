# -- coding:utf-8 --
from dev.config.logging import *

class ConfigBase(ExpLogging):
    def __init__(self, lp='', ldir=None):
        super(ConfigBase, self).__init__(lp, ldir)

        self.sKeychunklistlck = "chunklist.lck"
        self.sKeychunklistinfo = ".info"
        self.sKeychunklist = "chunklist"
        self.chunksize = 16
        self.shufflechunk = 8
        self.sKeyckptfnm = "siamQrnnCkpt"
        self.max_to_keep = 200
        self.checkpoint = None

        self.distributed = True
        self.allow_growth = True
        self.num_cores = 8
        self.log_device = True
        self.soft_placement = True
        self.ckptpersecs = 0
        self.devicemem = 1.0
        self.isps = False
        self.ischief = False
        self.job_name = 'worker'
        self.task_index = 0
        self.deviceid = -1

        self.maxepoch = 20
        self.grad_clip = 1.0
        self.max_norm = 100.0  # clip ops only for adam
        self.n_samples = 5
        self.max_sent_length = 20


        test_dir = '/search/odin/data/wangyuan/data-wenti/test'
        self.testfnms = ['test']
        self.testfnms = [test_dir + os.path.sep + fnms for fnms in self.testfnms]

        self.word_embd_dir = '/search/odin/data/wangyuan/pycharmProjects/wenti/word_list'
        self.embd_path = self.word_embd_dir + os.path.sep + 'cnn_online_word_vec_4.2M.pkl'
        self.wdict_path = self.word_embd_dir + os.path.sep + 'cnn_online_wdict_utf8.pkl'
        self.embd_dims = 100
        # 775354
        self.vocab_size = 4200000





