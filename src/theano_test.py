#!/urs/bin/env python

import numpy as np
import theano.tensor as T
from theano import function

def main():
    # A simple theano function to multiply matrices
    x = T.dmatrix('x')
    y = T.dmatrix('y')
    z = x * y
    f = function([x, y], z)

    # Multiply two random matrices
    m1 = np.random.rand(1000, 1000)
    m2 = np.random.rand(1000, 1000)
    result = f(m1, m2)

if __name__=="__main__":
  main()
