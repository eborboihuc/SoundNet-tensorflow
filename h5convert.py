import numpy as np
import h5py
import sys


th = h5py.File(sys.argv[1], 'r')
print th.keys()


if len(th.keys()) <= 1:
    key = th.keys()[0]
    npy = np.array(th[key])
else:
    npy = {}
    for key in th.keys():
        npy[key] = np.array(th[key])

np.save(sys.argv[2], npy)


