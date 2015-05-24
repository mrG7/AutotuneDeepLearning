__author__ = 'Roman'

import numpy as np

a = np.arange(400).reshape((40,10))
print a
a.reshape(len(a)*10,)
print a
