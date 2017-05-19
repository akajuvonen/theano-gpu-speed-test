#!/urs/bin/env python

from theano_process import process
import os


def main():
    # Run theano processing and print elapsed time
    # CPU
    os.environ['THEANO_FLAGS'] = 'device=cpu'
    time1 = process()
    # GPU
    os.environ['THEANO_FLAGS'] = 'device=cuda'
    time2 = process()
    # Print times
    # NOTE: This will not work yet, need to spawn separate Processes
    # that init theano again
    print('Time used with CPU: %f' % time1)
    print('Time used with GPU: %f' % time2)


if __name__ == "__main__":
    main()
