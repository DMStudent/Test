from dev.config.dynamic_setting import *
model_config_file = 'dev.config.config_model_' + str(ds.model)
model_file = 'dev.model.model_' + str(ds.model)
cm = __import__(model_config_file, fromlist=['dev.config'])

args = cm.ConfigModel()
args.a0_model_name = str(ds.model).upper()
args.model_config_file = model_config_file
args.model_file = model_file