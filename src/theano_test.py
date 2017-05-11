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
    m1 = np.random.rand(10, 10)
    m2 = np.random.rand(10, 10)
    result = f(m1, m2)
    print(result)

if __name__=="__main__":
  main()
