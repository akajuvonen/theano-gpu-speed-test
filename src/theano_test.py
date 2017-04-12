#!/urs/bin/env python

import numpy as np
import theano.tensor as T
from theano import function

def main():
  # A simple theano function to add to matrices
  x = T.dmatrix('x')
  y = T.dmatrix('y')
  z = x + y
  f = function([x, y], z)

if __name__=="__main__":
  main()
