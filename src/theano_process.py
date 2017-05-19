#!/urs/bin/env python

import numpy as np
import time
import os


def process(q,use_gpu):
    """Processes some calculations with or without gpu depending on config.
    Arguments:
    q -- The multiprocessing queue
    use_gpu -- Boolean, if True, uses gpu
    Returns:
    The elapsed time
    """
    if use_gpu:
        os.environ['THEANO_FLAGS'] = 'device=cuda'
        print('--- USING GPU ---')
    else:
        os.environ['THEANO_FLAGS'] = 'device=cpu'
        print('--- USING CPU ---')

    # Have to import after setting the THEANO_FLAGS param
    import theano.tensor as T
    from theano import function

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
    q.put(time1 - time0)
