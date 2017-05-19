#!/urs/bin/env python

from theano_process import process
from multiprocessing import Process, Queue


def main():
    # Run theano processing and print elapsed time
    # CPU
    q = Queue()
    p = Process(target=process, args=(q,False))
    p.start()
    p.join()
    timecpu = q.get()
    # GPU
    q = Queue()
    p = Process(target=process, args=(q,True))
    p.start()
    p.join()
    timegpu = q.get()
    # Print times
    print('Time used with CPU: %f' % timecpu)
    print('Time used with GPU: %f' % timegpu)


if __name__ == "__main__":
    main()
