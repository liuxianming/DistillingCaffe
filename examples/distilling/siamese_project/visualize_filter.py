import performance_test
import os, os.path

CAFFE_ROOT_DIR = '/mnt/ilcompf0d0/user/xliu/code/caffe'
MODEL_ROOT_DIR = os.path.join(CAFFE_ROOT_DIR, 'examples/distilling/multiclass_alexnet')
network_fn = 'deploy.prototxt'
model_fn = 'alexnet_field_iter_10000.caffemodel'

os.chdir(MODEL_ROOT_DIR)
performance_test.visualize_network_filter(network_fn, model_fn, base_dir=MODEL_ROOT_DIR, figure_fn='conv2.png', layer='conv2')
