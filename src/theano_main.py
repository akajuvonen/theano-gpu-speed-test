#!/urs/bin/env python

from theano_process import process
from multiprocessing import Process, Queue


def main():
    # Run theano processing and print elapsed time
    # CPU
    q = Queue()
    p = Process(target=process, args=(q, False))
    p.start()
    p.join()
    # The time used for cpu analysis
    timecpu = q.get()
    # GPU
    q = Queue()
    p = Process(target=process, args=(q, True))
    p.start()
    p.join()
    # The time used for gpu analysis
    timegpu = q.get()
    # Print times
    print('----- RESULTS -----')
    print('Time used with CPU: %f' % timecpu)
    print('Time used with GPU: %f' % timegpu)
    print("GPU was %.2f times faster" % (timecpu / timegpu))


if __name__ == "__main__":
    main()
