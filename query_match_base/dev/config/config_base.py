from dev.config.logging import *

class ConfigBase(ExpLogging):
    def __init__(self, lp='', ldir=None):
        '''
        word embedding path
        word dict path
        test pair path
        '''
        super(ConfigBase, self).__init__(lp, ldir)

        self.sKeychunklistlck = "chunklist.lck"
        self.sKeychunklistinfo = ".info"
        self.sKeychunklist = "chunklist"
        self.chunksize = 16
        self.shufflechunk = 8
        self.sKeyckptfnm = "ckpt"
        self.max_to_keep = 50
        self.checkpoint = None

        self.distributed = True
        self.allow_growth = True
        self.num_cores = 4
        self.log_device = True
        self.soft_placement = True
        self.ckptpersecs = 0
        self.devicemem = 1.0
        self.isps = False
        self.ischief = False
        self.job_name = 'worker'
        self.task_index = 0
        self.deviceid = -1

        self.maxepoch = 10
        self.grad_clip = 5.0
        self.max_norm = 250.0  # clip ops only for adam
        self.n_samples = 2
        self.max_sent_length = 30


        test_dir = '/search/odin/lianghuashen/tensor_exp/lstm_relevance/test_set'
        self.testfnms = ['markos_qt_56k', 'prdt1216_1', 'prdt1216_2','prdt1216_3', 'prdt1227_1', 'prdt1227_2','prdt1227_3','markos_qd_2m', 'qt_test']
        self.testfnms = [test_dir + os.path.sep + fnms for fnms in self.testfnms]

        word_embd_dir = '/search/odin/lianghuashen/tensor_exp/query_match_base/word_list'
        #self.embd_path = word_embd_dir + os.path.sep + 'cnn_online_word_vec_4.2M.pkl'
        self.embd_path = None
        self.wdict_path = word_embd_dir + os.path.sep + 'vocab_explict_775354.pkl'
        #self.wdict_path = word_embd_dir + os.path.sep + 'cnn_online_wdict.pkl'
        self.embd_dims = 100
        #self.vocab_size = 775354
        self.vocab_size = 4200000




