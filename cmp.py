import numpy as np
import sys

name = sys.argv[1]
dec  = int(sys.argv[2]) if len(sys.argv) >= 3 else 4

th = np.load('output/demo_th.npy').item()['layer{}'.format(name)].T
tf = np.load('output/tf_fea{}.npy'.format(name))
if name == '25':
    tf = np.concatenate([tf, np.load('output/tf_fea26.npy')], 1)

print 'Layer {}: tf.shape={}, th.shape={}'.format(name, tf.shape, th.shape)
print 'TF:'
print tf
print 'Torch:'
print th

size = tf.shape[0] if tf.shape[0] < th.shape[0] else th.shape[0]

print 'Round to {} decimals'.format(dec)
tf = np.round(tf, decimals=dec)
th = np.round(th, decimals=dec)
print 'Total Diff:', np.sum(abs(tf[:size] - th[:size])), \
      '\nMax Diff:', np.max(tf[:size] - th[:size]), \
      '\nMin Diff:', np.min(tf[:size] - th[:size])
