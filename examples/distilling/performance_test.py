#Test performance of trained model
import numpy as np
import lmdb
import sys
import os, os.path

CAFFE_ROOT = '/mnt/ilcompf0d0/user/xliu/code/caffe/python'
sys.path.append(CAFFE_ROOT)
import caffe
from caffe.io import caffe_pb2
import matplotlib.pyplot as plt

def get_net(network_fn, model_fn, mode='CPU'):
    if mode == 'CPU':
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
    try:
        net = caffe.Net(os.path.join(CAFFE_ROOT, network_fn),
                        os.path.join(CAFFE_ROOT, model_fn),
                        caffe.TEST)
        return net
    except ():
        return None

def classify_sample(net, sample):
    # classify a data sample using current net
    out = net.forward()
    return out[-1] # return the last layer output

def extract_sample(datum, mean_image):
    # extract numpy array from datum, then substract the mean_image
    return None
