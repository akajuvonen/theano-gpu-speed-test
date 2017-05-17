#!/urs/bin/env python

import theano.tensor as T
from theano import function
from theano import config

import numpy as np
import time


def process():
    """Processes some calculations with or without gpu depending on config.
    Returns:
    The elapsed time
    """
    # A simple theano function to multiply matrices
    x = T.dmatrix('x')
    y = T.dmatrix('y')
    z = x * y
    f = function([x, y], z)

    # Start time
    time0 = time.time()

    # Multiply two random matrices
    m1 = np.random.rand(10000, 10000)
    m2 = np.random.rand(10000, 10000)
    result = f(m1, m2)

    # End time
    time1 = time.time()

    # Return elapsed time
    return (time1 - time0)

def main():
    timedelta = process()
    print('Elapsed time: %f' % timedelta)

if __name__ == "__main__":
    main()
