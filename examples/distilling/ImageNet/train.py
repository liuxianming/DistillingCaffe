import sys
import os, os.path
import caffe
import py_data_layers

"""
usage: python train.py solver_file [device_id] [init_model]
"""
if __name__ == "__main__":
    solver_file = sys.argv[1]

    caffe.set_mode_gpu()
    if len(sys.argv) == 3:
        device_id = int(sys.argv[2])
    else:
        device_id = 0
    caffe.set_device(device_id)

    if len(sys.argv) == 4:
        # finetune
        pretrained_model_fn = sys.argv[3]
        if not os.path.exists(pretrained_model_fn):
            # download the model
            import urllib
            url = "http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel"
            urllib.urlretrieve(url, pretrained_model_fn)

    solver = caffe.SGDSolver(solver_file)
    if pretrained_model_fn:
        solver.net.copy_from(pretrained_model_fn)
    solver.solve()

