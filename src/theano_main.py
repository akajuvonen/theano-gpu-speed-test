#!/urs/bin/env python

from theano_process import process
import os


def main():
    # Run theano processing and print elapsed time
    # CPU
    os.environ['THEANO_FLAGS'] = 'device=cpu'
    time = process()
    print(time)
    # GPU
    os.environ['THEANO_FLAGS'] = 'device=cuda'
    time = process()
    print(time)


if __name__ == "__main__":
    main()
