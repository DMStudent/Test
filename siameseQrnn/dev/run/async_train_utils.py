# -- coding:utf-8 --
import time
import os
import tensorflow as tf
import numpy as np
from dev.config.final_config import *

def initenviroment():
    if args.isps is True:
        os.environ["CUDA_VISIBLE_DEVICES"] = " "
        return

    gpuid = int(args.deviceid)
    if (gpuid < 0) or (gpuid > 7):
        raise Exception("Bad deviceid value {}, expect 0-7".format(gpuid))

    args.message("Setup gpu id: {}".format(gpuid))
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpuid)

def initDistributed():
    args.message("Init distributed cluster ......")
    cluster = tf.train.ClusterSpec({"ps": args.pshosts, "worker": args.workerhosts})
    ## create server
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = args.allow_growth  # False if only one worker on one GPU.
    config.allow_soft_placement = True
    server = tf.train.Server(cluster, job_name=args.job_name, task_index=args.task_index, config=config)
    return server, cluster

def initworker(server, cluster, model):
    with tf.Graph().as_default(), tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % args.task_index, cluster=cluster)):
        model.init_global_step()

        with tf.device("/gpu:{}".format(args.deviceid)):
             vt, vs, vo = model.model_setup()
        args.write_variables(vt)
        args.write_variables(vo)
        # create supervisor
        args.message("Create supervisor ......")

        sv = tf.train.Supervisor(is_chief=args.ischief,
                                 logdir=args.output,
                                 saver=model.saver,
                                 checkpoint_basename=args.sKeyckptfnm,
                                 init_op=tf.initialize_all_variables(),
                                 save_model_secs=0,
                                 global_step=model.global_step)

        # create session
        configproto = tf.ConfigProto()
        configproto.gpu_options.allow_growth = args.allow_growth    # False if only one worker on one GPU.
        configproto.log_device_placement = args.log_device
        configproto.allow_soft_placement = args.soft_placement
        configproto.inter_op_parallelism_threads = args.num_cores
        configproto.intra_op_parallelism_threads = args.num_cores

        if args.ischief is True:
            args.message(("Worker %d: Initializing session..." % args.task_index))
        else:
            args.message(("Worker %d: Waiting for session to be initialized..." % args.task_index))

        sess = sv.prepare_or_wait_for_session(server.target, config=configproto, start_standard_services=False)

        return sess

def trainLoop(model, sess, dataset):
    if args.task_index != 1 and args.numworker > 1:
        args.message("Warming up for stable training, only task=1 is working .......")
        time.sleep(100.0 + np.random.randint(0, 100) * ((args.task_index + 1) % 10))

    loops = 0
    tik = -1  # trigger every args.ckptperbatch time
    tloss, tscore_loss, tacc = 0.0, 0.0, 0.0
    n_batches = float(args.showperbatch)
    args.message("Start train loop ......")
    while 1:
        loops += 1
        data_proc_slice = dataset.get_batch()
        label, q, qm, t, tm, dm, extra_score = data_proc_slice


        t_time = time.time()
        if loops % args.testperbatch == 0:
            acc, score = \
                model.run_epoch(sess, data_proc_slice, False)
            out_str = "TEST : acc:%f Time:%f" \
                      % (acc, t_time)
            args.message(out_str, True)
            out_str = "Score: %f - %f - %f - %f - %f" % (score[0], score[1], score[2], score[3], score[4])
            args.message(out_str, True)
            out_str = "label: %f - %f - %f - %f - %f" % (label[0], label[1], label[2], label[3], label[4])
            args.message(out_str, True)
            continue


        gstep, loss, score_loss, acc, score, _ = \
            model.run_epoch(sess, data_proc_slice, True)

        tloss, tscore_loss, tacc = \
            tloss + loss, tscore_loss + score_loss, tacc + acc
        t_time = time.time() - t_time

        if loops % args.showperbatch == 0:
            out_str = "G/L %d/%d: loss:%f l2:%f acc:%f Time:%f" \
                      % (gstep + 1, loops, tscore_loss/n_batches, tloss/n_batches,
                         tacc/n_batches, t_time)
            tloss, tscore_loss, tacc = 0.0, 0.0, 0.0
            args.message(out_str, True)
            out_str = "Score: %f - %f - %f - %f - %f" % (score[0], score[1], score[2], score[3], score[4])
            args.message(out_str, True)
            out_str = "label: %f - %f - %f - %f - %f" % (label[0], label[1], label[2], label[3], label[4])
            args.message(out_str, True)
            # out_str = "r_final: %f - %f - %f - %f - %f" % (r_final[0], r_final[1], r_final[2], r_final[3], r_final[4])
            # args.message(out_str, True)
            # out_str = "r_merge: %f - %f - %f - %f - %f" % (r_merge[0][0], r_merge[0][1], r_merge[0][2], r_merge[1][0], r_merge[1][1])
            # args.message(out_str, True)

        # if args.ischief:
        if not args.ischief:
            new_tik = (gstep + 1) % args.ckptperbatch
            ckpt_id = int((gstep + 1) / args.ckptperbatch)
            if new_tik <= tik:
                time.sleep(((new_tik + 1) % 20) * 10)
                fnm = str("{}_{}".format(args.sKeyckptfnm, ckpt_id))
                ckptfnm = os.path.join(args.output, fnm)
                if not os.path.exists(ckptfnm):
                    args.message("Saving checkpoint model: {} ......".format(ckptfnm))
                    model.saver.save(sess, ckptfnm, write_meta_graph=False)
            tik = new_tik

