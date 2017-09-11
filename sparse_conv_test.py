import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

import numpy as np
import time

sparse_conv_module = tf.load_op_library('sparse_conv.so')

def create_img(height, width, batch_size=1):
	return np.float32(np.random.rand(batch_size, height, width, 1))

def create_filter(filter_height, filter_width, in_channels, out_channels):
	return np.float32(np.random.rand(filter_height, filter_width, in_channels, out_channels))

strides = [1,3,2,1]
padding = 'SAME'
filter_shape = [5, 3, 1, 1]

X = create_img(14, 19, 4)
W = create_filter(filter_shape[0], filter_shape[1], 
				  filter_shape[2], filter_shape[3])
filter_height, filter_width, in_channels, out_channels = W.shape

W_flat = tf.reshape(W, [-1])
W_indices = tf.stack([tf.stack([tf.to_int64((((i*filter_width) + j)*in_channels) + k), tf.to_int64(l)]) for i in range(filter_height) for j in range(filter_width) for k in range(in_channels) for l in range(out_channels)])
W_shp = tf.stack([tf.to_int64(tf.to_int64(filter_height)*tf.to_int64(filter_width)*in_channels), tf.to_int64(out_channels)])
W_sparse = tf.SparseTensor(indices=W_indices, values=W_flat, dense_shape=W_shp)

with tf.Session(''):
    t1 = time.time()
    x1 = sparse_conv_module.sparse_conv(X, W_flat, W_indices, W_shp, filter_shape, strides=strides, padding=padding).eval()
    print(time.time()-t1)

    t1 = time.time()
    x2 = tf.nn.conv2d(X, W, strides=strides, padding=padding).eval()
    print(time.time()-t1)
print(x1.shape)
print(x2.shape)

diff = np.abs(x2-x1)
# print(diff)
print(np.max(diff))
print(np.sum(diff))

# print(diff[0,:,:,0])
# print(diff.shape)
# print(diff[0,:,:,0].shape)