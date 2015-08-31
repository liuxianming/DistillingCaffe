import performance_test
import os, os.path

CAFFE_ROOT_DIR = '/mnt/ilcompf0d0/user/xliu/code/caffe'
MODEL_ROOT_DIR = os.path.join(CAFFE_ROOT_DIR, 'examples/distilling')
network_fn = os.path.join(MODEL_ROOT_DIR, 'train_field_alexnet.prototxt')
model_fn = os.path.join(MODEL_ROOT_DIR, 'conv_field_finetune_iter_11000.caffemodel')
original_model_fn = os.path.join(MODEL_ROOT_DIR, 'conv_field_finetune_iter_1000.caffemodel')

os.chdir(CAFFE_ROOT_DIR)
performance_test.visualize_network_filter(network_fn, model_fn, './examples/distilling/finetune_filter_conv3.png', 'conv1')
#performance_test.visualize_network_filter(network_fn, original_model_fn, './examples/distilling/finetune_filter_1000.png', 'conv1')
