import matplotlib
matplotlib.use('Agg')
import numpy as np
import lmdb
import sys
import os, os.path
#from contextlib import nested
import scipy.misc

CAFFE_ROOT = '/mnt/ilcompf0d0/user/xliu/code/caffe/python'
sys.path.append(CAFFE_ROOT)
import caffe
from caffe.io import caffe_pb2
import matplotlib.pyplot as plt

network_fn = '/mnt/ilcompf0d0/user/xliu/code/caffe/models/bvlc_alexnet/deploy.prototxt'
model_fn = '/mnt/ilcompf0d0/user/xliu/code/caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel'

caffe.set_mode_gpu()
net = caffe.Net(network_fn, model_fn, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.array([127, 127, 127]))

transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
net.blobs['data'].reshape(50,3,227,227)

img_data = np.load('/mnt/ilcompf0d0/user/xliu/code/caffe/examples/distilling/0.npy')
print img_data.flatten()[:200]
print img_data.flags
print img_data.dtype
print net.params['conv2'][0].data.flatten()[:200]
img_data = np.zeros(img_data.shape)
net.blobs['data'].data[...] = img_data

out = net.forward()
print "conv1"
print net.blobs['conv1'].data[0].flatten()[:200]
print net.blobs['conv1'].data[0].dtype

response =  net.blobs['fc6'].data[0].flatten()
#print response[:100]

img_data = transformer.preprocess('data', caffe.io.load_image('/mnt/ilcompf0d0/user/xliu/code/caffe/examples/distilling/' + '0.png'))
net.blobs['data'].data[...] = img_data
out = net.forward()
response =  net.blobs['fc6'].data[0].flatten()
#print response[:100]
