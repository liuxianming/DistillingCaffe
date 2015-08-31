import sys
import os, os.path
CAFFE_ROOT_DIR = '/mnt/ilcompf0d0/user/xliu/code/caffe/'
sys.path.append(os.path.join(CAFFE_ROOT_DIR, 'python'))

import caffe
from caffe.io import caffe_pb2
import py_data_layers

solver_dir = os.path.join(CAFFE_ROOT_DIR, 'examples/distilling/siamese_project')
os.chdir(solver_dir)
caffe.set_mode_gpu()
caffe.set_device(1)
alexnet_model_fn = os.path.join(CAFFE_ROOT_DIR, 'models/bvlc_alexnet/bvlc_alexnet.caffemodel')

solver = caffe.SGDSolver( os.path.join(solver_dir, 'solver_1.prototxt') )
solver.net.copy_from(alexnet_model_fn)
solver.solve()
