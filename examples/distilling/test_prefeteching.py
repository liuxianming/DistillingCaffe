"""test_prefetching.py
Test prefetching functions and multiprocessing
"""
import numpy as np
import time
from multiprocessing import Process, Pipe

class Prefetcher(Process):
    def __init__(self, size, conn):
        super(Prefetcher, self).__init__()
        self._size = size
        self._conn = conn
        print "Initialization done"

    def _get_next_batch(self):
        # generate a random numpy array
        start = time.time()
        batch = np.zeros((self._size, ))
        end = time.time()
        print "A new batch is generated, using {} seconds".format(end-start)
        return batch

    def run(self):
        print "Prefetcher started..."
        while True:
            batch = self._get_next_batch()
            print "[Run] A new batch fetched"
            self._conn.send(batch)


class Tester:
    def setup(self, batch_size=64):
        self._batch_size = batch_size
        self._conn, conn = Pipe()
        self._prefetch_process = Prefetcher(self._batch_size, conn)
        self._prefetch_process.start()

        def cleanup():
            print "Terminating Prefetcher..."
            self._prefetch_process.terminate()
            self._prefetch_process.join()
            self._conn.close()
        import atexit
        atexit.register(cleanup)

    def Test(self, iteration):
        print "Beging Testing Prefetcher"
        for i in range(iteration):
            print "[-----------Iteration {}-------------]".format(i)
            start = time.time()
            batch = self._conn.recv()
            print batch
            end = time.time()
            print "Received a batch from prefetcher.\
            [{} seconds]".format(end-start)

        print "Test ended"

if __name__ == "__main__":
    tester = Tester()
    tester.setup()
    tester.Test(10)
