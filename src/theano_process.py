#!/urs/bin/env python

import numpy as np
import time
import os


def process(q, use_gpu):
    """Processes some calculations with or without gpu depending on config.
    Arguments:
    q -- The multiprocessing queue
    use_gpu -- Boolean, if True, uses gpu
    Returns:
    The elapsed time
    """
    if use_gpu:
        os.environ['THEANO_FLAGS'] = 'device=cuda,floatX=float32'
        print('--- USING GPU ---')
    else:
        os.environ['THEANO_FLAGS'] = 'device=cpu,floatX=float32'
        print('--- USING CPU ---')

    # Have to import after setting the THEANO_FLAGS param
    import theano
    import theano.tensor as T

    m = np.random.rand(1000, 1000)
    n = np.random.rand(1000, 1000)

    tm = theano.shared(m)
    tn = theano.shared(n)

    f = theano.function([], tm * tn)

    # Start time
    time0 = time.time()

    # Repeat the multiplication
    for i in range(1000):
        f()

    # End time
    time1 = time.time()

    # Return elapsed time
    q.put(time1 - time0)
