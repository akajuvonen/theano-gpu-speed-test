#!/urs/bin/env python

import theano.tensor as T
from theano import function
from theano import config

import numpy as np
import time


def process(f):
    """Processes some calculations with or without gpu depending on config.
    """
    # A simple theano function to multiply matrices
    x = T.dmatrix('x')
    y = T.dmatrix('y')
    z = x * y
    f = function([x, y], z)

    # Multiply two random matrices
    m1 = np.random.rand(10000, 10000)
    m2 = np.random.rand(10000, 10000)
    result = f(m1, m2)


def main():
    # Start time
    time0 = time.time()
    process(f)
    # End time
    time1 = time.time()
    # Elapsed time
    print('Elapsed time: %f' % (time1 - time0))

if __name__ == "__main__":
    main()
