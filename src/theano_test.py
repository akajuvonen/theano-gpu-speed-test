#!/urs/bin/env python

import numpy as np
import theano.tensor as T
from theano import function
import time


def main():
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

    # Elapsed time
    print('Elapsed time: %f' % (time1 - time0))

if __name__ == "__main__":
    main()
