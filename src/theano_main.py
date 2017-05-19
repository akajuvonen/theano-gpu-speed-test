#!/urs/bin/env python

from theano_process import process


def main():
    # Run theano processing and print elapsed time
    time = process()
    print(time)

if __name__ == "__main__":
    main()
