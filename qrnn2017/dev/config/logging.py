import os
import time

class ExpLogging(object):

    def __init__(self, lp='', ldir=None):
        self.log_prefix = lp
        if ldir is None:
            self.log_dir_path = './results'
        else:
            self.log_dir_path = ldir
        if not os.path.exists(self.log_dir_path):
            os.makedirs(self.log_dir_path)

    def write_args(self, args):
        self.message("-------- Parameter Info --------")
        name_dict = sorted(args.__dict__)
        for i, name in enumerate(name_dict):
            self.message("{}: {} = {}".format(str(i), name, args.__dict__[name]))
        self.message('--------------------')

    def message(self, str_line, write_result=False):
        str_stream = "{}, {}: {}".format(time.strftime("%Y-%m-%d %H:%M:%S"), self.log_prefix, str_line)
        print(str_stream)
        if not write_result:
            wf = self.log_dir_path + os.path.sep + self.log_prefix + 'log.log'
            with open(wf, 'a') as f:
                f.write(str_stream + '\n')
        if write_result:
            wf = self.log_dir_path + os.path.sep + self.log_prefix + 'results.log'
            with open(wf, 'a') as f:
                f.write(str_stream + '\n')

    def write_variables(self, var_list):
        self.message("-------- Model Variables --------")
        cnt = 0
        for var in var_list:
            cnt += 1
	    try:
                var_shape = var.get_shape()
	    except:
                var_shape = var.dense_shape
            str_line = str(cnt) + '. ' + str(var.name) + ': ' + str(var_shape)
            self.message(str_line)
        self.message('--------------------')

    def file_copy(self, file_list):
        backup_path = self.log_dir_path + os.path.sep + 'backup.' + time.strftime('%Y%m%d.%H%M')
        if not os.path.exists(backup_path):
            os.makedirs(backup_path)
        for f in file_list:
            os.system('\\cp -rf ' + f + ' ' + backup_path + os.path.sep)


