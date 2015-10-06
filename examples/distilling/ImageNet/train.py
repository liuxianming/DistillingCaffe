import sys
import os, os.path
CAFFE_ROOT_DIR = '/mnt/sdb/xliu102/code/DeepLearning/DistillingCaffe'
sys.path.append(os.path.join(CAFFE_ROOT_DIR, 'python'))

import caffe
import py_data_layers

if __name__ == "__main__":
    task = sys.argv[1]
    solver_dir = os.path.join(
        CAFFE_ROOT_DIR, 'examples/distilling/ImageNet/', task)
    os.system("cp -f train_val.prototxt ./{}/".format(task))
    os.system("cp -f solver.prototxt ./{}/".format(task))

    os.chdir(task)

    caffe.set_mode_gpu()
    if len(sys.argv) == 3:
        device_id = int(sys.argv[2])
    else:
        device_id = 0
    caffe.set_device(device_id)

    solver = caffe.SGDSolver('solver.prototxt')
    solver.solve()
