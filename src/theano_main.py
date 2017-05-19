#!/urs/bin/env python

from theano_process import process
import os
from multiprocessing import Process, Queue


def main():
    # Run theano processing and print elapsed time
    # CPU
    os.environ['THEANO_FLAGS'] = 'device=cpu'
    q = Queue()
    p = Process(target=process, args=(q,))
    p.start()
    p.join()
    timecpu = q.get()
    # GPU
    os.environ['THEANO_FLAGS'] = 'device=cuda'
    #time2 = process()
    # Print times
    # NOTE: This will not work yet, need to spawn separate Processes
    # that init theano again
    print('Time used with CPU: %f' % timecpu)
    #print('Time used with GPU: %f' % time2)


if __name__ == "__main__":
    main()
