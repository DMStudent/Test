import fcntl

class FLOCK(object):
    def __init__(self, name):
        self.name = name
        self.fobj = open(name, 'w')
        self.fd = self.fobj.fileno()

    def lock(self):
        try:
            fcntl.lockf(self.fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            return True
        except IOError:
            return False
        except Exception as e:
            self.fobj.close()
            raise Exception("Lock {} failed? {}".format(self.name, e))

    def unlock(self):
        self.fobj.close()
