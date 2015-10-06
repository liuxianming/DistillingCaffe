import sys
import os, os.path
CAFFE_ROOT_DIR = '/mnt/sdb/xliu102/code/DeepLearning/DistillingCaffe'
sys.path.append(os.path.join(CAFFE_ROOT_DIR, 'python'))

import caffe
import py_data_layers
import urllib

if __name__ == "__main__":
    task = sys.argv[1]
    solver_dir = os.path.join(
        CAFFE_ROOT_DIR, 'examples/distilling/ImageNet/', task)
    os.system("cp -f train_val.prototxt ./{}/".format(task))
    os.system("cp -f solver.prototxt ./{}/".format(task))

    caffe.set_mode_gpu()
    if len(sys.argv) == 3:
        device_id = int(sys.argv[2])
    else:
        device_id = 0
    caffe.set_device(device_id)

    if len(sys.arg) == 4:
        # finetune
        pretrained_model_fn = sys.argv[3]
        if not os.path.exists(pretrained_model_fn):
            # download the model
            url = "http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel"
            urllib.urlretrieve(url, pretrained_model_fn)

    os.chdir(task)
    solver = caffe.SGDSolver('solver.prototxt')
    if pretrained_model_fn:
        solver.net.copy_from(pretrained_model_fn)
    solver.solve()
