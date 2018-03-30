# -- coding:utf-8 --
import math
import time
import os
from dev.config.final_config import *
from dev.io.flock import FLOCK

chunkfnm = os.path.join(args.output, args.sKeychunklist)
chunkfnminfo = chunkfnm + args.sKeychunklistinfo
chunklckfnm = os.path.join(args.output, args.sKeychunklistlck)


def filterfiles(files):
    if len(files) == 0:
        raise Exception("No train data file be defined?")
    badfiles, goodfiles = [], []
    filenm = [f.strip() for f in files]
    for f in filenm:
        if not os.path.exists(f):
            badfiles.append(f)
        else:
            goodfiles.append(f)

    if not len(badfiles) == 0:
        args.message("Can't find data files:")
        for i in range(len(badfiles)):
            args.message("\t'{}'".format(badfiles[i]))
    if len(goodfiles) == 0:
        raise Exception("No available data files!")

    args.message("Data files: ")
    readablefiles = []
    for i in range(len(goodfiles)):
        curflen = os.path.getsize(goodfiles[i])
        curflen /= (1024 * 1024)
        if curflen < 10:
            args.message("File: {} {}M, too small?".format(goodfiles[i], curflen))
        else:
            readablefiles.append(goodfiles[i])
            args.message("\tNo.{} file:{}".format(i + 1, goodfiles[i]))
    if len(readablefiles) == 0:
        raise Exception("Found 0 records?")
    return readablefiles


trainfilelist = filterfiles(args.trainfnms)


def doCreateChunkList():
    assert not os.path.exists(chunkfnm)
    args.message("Build train dataset chunklist {} ...".format(chunkfnm))

    fileid, filelen = [], []
    for i in range(len(trainfilelist)):
        args.message("No.{} file {} ......".format(i + 1, trainfilelist[i]))
        curflen = os.path.getsize(trainfilelist[i])
        curflen /= 1024 * 1024
        fileid.append(trainfilelist[i])
        nchunk = int(math.ceil(curflen / args.chunksize))  # chunk size == 64M
        for num in range(nchunk):
            offset = args.chunksize * num
            filelen.append("{}\t{}".format(i, offset * 1024 * 1024))

    args.message("Write chunk list file info {} ...".format(chunkfnminfo))
    with open(chunkfnminfo, "w") as f:
        for item in fileid:
            f.write("%s\n" % item)

    args.message("Write chunk list file {} ...".format(chunkfnm))
    with open(chunkfnm, "w") as f:
        ## f.write("%s\n" % len(filelen))
        for item in filelen:
            f.write("%s\n" % item)


def createChunkList():
    lck = FLOCK(chunklckfnm)
    while 1:
        ret = lck.lock()
        if ret:
            if not os.path.exists(chunkfnm):
                try:
                    doCreateChunkList()
                    lck.unlock()
                    return
                except Exception as e:
                    lck.unlock()
                    raise Exception("Can't create file {}?\n{}".format(chunkfnm, e))
            else:
                lck.unlock()
                return
        else:
            time.sleep(1)
            continue


def loadChunkList(workid, worknum):
    args.message("Load chunk list from file {} ......".format(chunkfnm))
    if (not os.path.exists(chunkfnminfo)) or (not os.path.exists(chunkfnm)):
        raise Exception("Cant find file {} or {}?".format(chunkfnm, chunkfnminfo))

    with open(chunkfnminfo) as f:
        flist = f.readlines()
    for i in range(len(flist)):
        flist[i] = flist[i].strip()
    assert len(flist) > 0

    with open(chunkfnm) as f:
        chunklist = f.readlines()
    for i in range(len(chunklist)):
        chunklist[i] = chunklist[i].strip()

    assert len(chunklist) > worknum
    nchunkblock = int(math.ceil(len(chunklist) / worknum))
    startid = nchunkblock * workid
    stopid = nchunkblock
    if workid == worknum - 1:
        stopid = len(chunklist) - nchunkblock * (worknum - 1)

    retchunklist = chunklist[startid: startid + stopid - 1]

    return flist, retchunklist


def getChunkList():
    assert args.output is not None
    assert args.distributed is True
    assert (args.task_index != -1)
    assert (args.numworker != 0)
    createChunkList()
    return loadChunkList(args.task_index, args.numworker)
