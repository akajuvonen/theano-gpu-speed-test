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
        os.environ['THEANO_FLAGS'] = 'device=cuda'
        print('--- USING GPU ---')
    else:
        os.environ['THEANO_FLAGS'] = 'device=cpu'
        print('--- USING CPU ---')

    # Have to import after setting the THEANO_FLAGS param
    import theano
    import theano.tensor as T

    # Random matrices
    m = np.random.rand(1000, 1000).astype('float32')
    n = np.random.rand(1000, 1000).astype('float32')
    # The output variable, zeros for now
    y = np.zeros((1000, 1000)).astype('float32')

    # Make these into theano shared variables, stored in GPU
    t_m = theano.shared(m)
    t_n = theano.shared(n)
    t_y = theano.shared(y)

    # Function for element-wise addition
    t_add = T.add(t_m, t_n)

    # Update the output variable using the function
    f = theano.function(inputs=[], updates={t_y : t_add})

    # Start time
    time0 = time.time()

    # Repeat the multiplication
    for i in range(1000):
        f()

    # End time
    time1 = time.time()

    # Return elapsed time
    q.put(time1 - time0)
