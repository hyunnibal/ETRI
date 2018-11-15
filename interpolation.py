import eval
import numpy as np

identifier = 'sine'
for nn in np.arange(50,1005,50):
    eval.view_interpolation(identifier, nn)