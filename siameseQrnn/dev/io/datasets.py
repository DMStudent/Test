# -- coding:utf-8 --
from dev.config.final_config import *
import time
import multiprocessing as mp
import Queue as lQueue
from dev.io.mpreader import readerProc
from dev.io.chunklist import getChunkList


class initDataset(object):
    def __init__(self, proc_fun):
        flist, chunklist = getChunkList()
        self.quitEvent = mp.Event()
        self.mpQueue = mp.Queue(100)
        self.producer = mp.Process(name="Loader", target=readerProc,
                                   args=(proc_fun, flist, chunklist,
                                         self.mpQueue, self.quitEvent))
        self.producer.start()
        self.isrunning = True

    def __del__(self):
        self.setstop()

    def get_batch(self):
        if not self.isrunning:
            raise Exception("You should start Producer first!")
        while 1:
            try:
                return self.mpQueue.get_nowait()
            except lQueue.Empty:
                if self.quitEvent.is_set():
                    raise StopIteration("Get quit event, IO finished!")
                time.sleep(0.1)
                continue

    def setstop(self):
        if not self.isrunning:
            return
        self.quitEvent.set()
        self.producer.join()
        self.isrunning = False
        pass


