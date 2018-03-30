import sys
import os
import argparse
from dev.config.final_config import *
from dev.model.final_model import graph_moudle
from dev.io.datasets import *
from dev.run.async_train_utils import *

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_name', type=str, default="worker"
                        , help="One of 'ps', 'worker'")
    parser.add_argument('--task_index', type=int, default=0
                        , help="Index of task within the job")
    parser.add_argument('--deviceid', type=int, default=-1
                        , help="GPU ID: 0-7")
    command_line = parser.parse_args()

    args.deviceid = command_line.deviceid
    args.task_index = command_line.task_index
    args.job_name = command_line.job_name

    if args.task_index == 0:
        args.ischief = True

    if args.job_name == "ps":
        args.isps = True


def main():
    if len(sys.argv) == 1:
        print("Usage: train.py --job_name --task_index --deviceid\n"
              "                --job_name: One of 'ps', 'worker'\n"
              "                --task_index: Index of task within the job\n"
              "                --deviceid: GPU ID: 0-7\n")

        exit(1)
    # with global variable 'log' and 'args'
    parseArgs()
    args.log_prefix = args.job_name + ':' + str(args.task_index) + ':' + str(args.deviceid) + '_'
    initenviroment()
    server, cluster = initDistributed()


    if args.isps is True:
        args.message("Parameter server joined!")
        args.write_args(args)
        server.join()
    else:
        model = graph_moudle.Model()
        ## Step 1: Init tensorflow:
        sess = initworker(server, cluster, model)

        ## Step 3: Init dataset
        dataset = initDataset(model.data_proc)
        # modified by sam
        #dataset = initDataset(model.data_proc_url)


        args.write_args(args)
        if args.ischief:
            args.file_copy(['dev'])

        ## Step 4: Run it
        trainLoop(model, sess, dataset)
