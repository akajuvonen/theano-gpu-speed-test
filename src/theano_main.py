#!/urs/bin/env python

from theano_process import process
import os
from multiprocessing import Process, Queue


def main():
    # Run theano processing and print elapsed time
    # CPU
    print('--- USING CPU ---')
    os.environ['THEANO_FLAGS'] = 'device=cpu'
    q = Queue()
    p = Process(target=process, args=(q,))
    p.start()
    p.join()
    timecpu = q.get()
    # GPU
    print('--- USING GPU ---')
    os.environ['THEANO_FLAGS'] = 'device=cuda'
    q = Queue()
    p = Process(target=process, args=(q,))
    p.start()
    p.join()
    timegpu = q.get()
    # Print times
    print('Time used with CPU: %f' % timecpu)
    print('Time used with GPU: %f' % timegpu)


if __name__ == "__main__":
    main()
