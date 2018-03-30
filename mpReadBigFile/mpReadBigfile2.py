import time
import multiprocessing as mp
import Queue as lQueue
import os

class initDataset(object):
    def __init__(self, filename, proc_fun):
        self.quitEvent = mp.Event()
        self.mpQueue = mp.Queue(10*10)
        self.producer = mp.Process(name="Loader", target=proc_fun,
                                   args=(filename, self.mpQueue, self.quitEvent))
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
        self.producer.terminate()
        self.producer.join()
        # print "self.isrunning = False"
        self.isrunning = False
        pass


def read_fun(filename,queue,quitEvent):
    offset = 0
    while 1:
        if quitEvent.is_set():
            return

        flen = os.path.getsize(filename)
        if offset >= flen:
            return None

        with open(filename) as f:
            f.seek(offset, 0)
            readlen = 1024 * 1024  # want load 64M
            if flen - offset < readlen:
                readlen = flen - offset

            indata = f.read(readlen)
            recodes = indata.split('\n')
            if len(recodes) < 2:
                return None
            lastLen = len(recodes[-1])
            offset = offset+readlen-lastLen
            recodes[: len(recodes) - 1]

        ## push into queue
        while 1:
            try:
                for item in recodes[: len(recodes) - 1]:
                    queue.put_nowait(item)
                break
            except lQueue.Full:
                if quitEvent.is_set():
                    return
                time.sleep(0.1)
                continue

    return True


def read_data(dataset, batch_size=64):
    data_set = []
    for i in range(batch_size):
        source = dataset.get_batch()
        data_set.append(source)

    return data_set

if __name__ == '__main__':
    data_dir = '/search/odin/data/wangyuan/pycharmProjects/textsum/data'
    train_path = os.path.join(data_dir, "train")
    src_train_path = os.path.join(train_path, "content-train.txt")
    dest_train_path = os.path.join(train_path, "title-train.txt")

    src_dataset = initDataset(src_train_path, read_fun)
    dest_dataset = initDataset(dest_train_path, read_fun)
    for i in range(10):
        src_batch = read_data(src_dataset, batch_size = 10)
        print ("batch num: %d" % i)
        for item in src_batch:
            print "".join(item)
        dest_batch = read_data(dest_dataset, batch_size=10)
        print ("batch num: %d" % i)
        for item in dest_batch:
            print "".join(item)
    src_dataset.setstop()
    dest_dataset.setstop()