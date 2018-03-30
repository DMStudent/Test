# -- coding:utf-8 --
import time
import os
import random
import Queue as lQueue
from dev.config.final_config import *


def loadNextChunk(flist, chunklist, ptr):
    ## TODO: load more chunk, do shuffer
    dchunk = []
    for i in range(args.shufflechunk):
        dchunk_tmp, ptr = loadChunkData(flist, chunklist, ptr)
        assert len(dchunk_tmp) >= args.batchsize
        dchunk.extend(dchunk_tmp)
    random.shuffle(dchunk)
    return dchunk, ptr


def loadChunkData(flist, chunklist, ptr):
    count = 0
    while 1:
        indata, ptr = doLoadChunkData(flist, chunklist, ptr)
        ## TODO: if indata size < batch size, need more loader!
        if (indata is None) or (len(indata) < args.batchsize):
            count += 1
            if count >= len(chunklist):
                raise Exception("Can't get any data from chunk list")
            continue
        else:
            return indata, ptr


def doLoadChunkData(flist, chunklist, ptr):
    if ptr >= len(chunklist):
        ptr = 0
    cl = chunklist[ptr]
    ptr += 1

    info = cl.split('\t')
    assert len(info) == 2
    fileno = int(info[0])
    if fileno >= len(flist):
        raise Exception("Bad chunklist file, expect {} data file, found {} data file"
                        .format(fileno + 1, len(flist)))
    filenm = flist[fileno]
    offset = int(info[1])

    flen = os.path.getsize(filenm)
    if offset >= flen:
        raise Exception("Bad chunklist file, file {} size = {}, expect {}"
                        .format(filenm, flen, offset))

    with open(filenm) as f:
        f.seek(offset, 0)
        readlen = args.chunksize * 1024 * 1024  # want load 64M
        if flen - offset < readlen:
            readlen = flen - offset

        indata = f.read(readlen)
        recodes = indata.split('\n')
        if len(recodes) <= 2:
            return None, ptr

        return recodes[1: len(recodes) - 1], ptr


def loadNextBatch(buffer):
    assert len(buffer) >= args.batchsize
    ret = []
    if len(buffer) == args.batchsize:
        return buffer, ret

    return buffer[:args.batchsize], buffer[args.batchsize:]


def readerProc(proc_fun, flist, chunklist, queue, quitEvent):
    nchunk = len(chunklist)
    random.shuffle(chunklist)

    curepoch = 0
    buffer = []
    curptr = 0
    numchunk = 0
    while 1:
        if quitEvent.is_set():
            return
        ## load next batch
        if len(buffer) < args.batchsize:
            chunkdata, curptr = loadNextChunk(flist, chunklist, curptr)
            buffer.extend(chunkdata)
            numchunk += args.shufflechunk

            if numchunk >= nchunk:
                curepoch += 1
                if curepoch >= args.maxepoch:
                    quitEvent.set()
                    return
                random.shuffle(chunklist)
                curptr = 0
                numchunk = 0
            else:
                args.message("Loader: {}/{} chunks loaded ......".format(numchunk, nchunk))

        batch, buffer = loadNextBatch(buffer)
        ## push into queue
        while 1:
            try:
                queue.put_nowait(proc_fun(batch))
                break
            except lQueue.Full:
                if quitEvent.is_set():
                    return
                time.sleep(0.1)
                continue
