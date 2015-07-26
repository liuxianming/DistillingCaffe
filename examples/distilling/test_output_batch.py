import performance_test
#from performance_test import *
import pickle
import os, os.path
import pprint

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import sys
sys.path.append('/mnt/ilcompf0d0/user/xliu/code/caffe/python/')
import caffe
from caffe.io import caffe_pb2
task_dir = '/mnt/ilcompf0d0/user/xliu/code/caffe/examples/distilling/Retrain_Conv5/'
os.chdir(task_dir)
pp = pprint.PrettyPrinter(indent=4)

network_fn=os.path.join(task_dir, 'train_val_field.prototxt')
model_fn = os.path.join(task_dir, 'alexnet_field_iter_30000.caffemodel')
solver_fn = os.path.join(task_dir, 'field_solver.prototxt')
caffe.set_mode_cpu()
caffe.set_device(0)
solver = caffe.SGDSolver(solver_fn)
print "[=============Loading Pre-Trained Model===========]"
solver.net.copy_from(model_fn)

print "[=============Print Netowrk Structure============]"
pp.pprint([(k, v.data.shape) for k, v in solver.net.blobs.items()])

solver.step(1)
# show gradient
print "[=============Showing gradient=============]"
gradient_map = solver.net.params['conv1'][0].diff[:, 0]
plt.imshow(gradient_map.reshape(12, 8, 11, 11).transpose(0, 2, 1, 3).reshape(12*11, 8*11), cmap='gray')
plt.savefig('./conv1_gradient.png')
print gradient_map.shape
# net = caffe.Net(network_fn, model_fn, caffe.TEST)

print "[=============Showing Classification=========]"
print "loss = {}".format(solver.net.blobs['loss'].data)
"""
print "Output Probablity"
pp.pprint(solver.net.blobs['prob'].data)
output = solver.net.blobs['prob'].data
label = solver.net.blobs['label'].data
for idx in range(32):
    print "Prediction:"
    pred = np.argmax(output[idx, ...].flatten())
    pp.pprint("{} : {}".format(pred, output[idx, ...].flatten()[pred]))
    print "Labels:"
    pp.pprint("{} : {}".format( np.nonzero(label[idx, ...].flatten()), np.sum(label[idx, ...].flatten()) ))
"""
for idx in range(16):
    response = solver.net.blobs['fc6'].data[idx, ...]
    print np.sum(response)
