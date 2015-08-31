"""project_verification.py
Using deploy network definition to do project verification task
The basic work flow is to extract certain amount of samples from the testing set
and then get the output (feat_norm)
Then calculate a distance matrix N * N
Using PR-Curve to evaluate the performance of each network
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import os, os.path
CAFFE_ROOT_DIR = '/mnt/ilcompf0d0/user/xliu/code/caffe/'
sys.path.append(os.path.join(CAFFE_ROOT_DIR, 'python'))
import caffe
from caffe.io import caffe_pb2
import numpy as np
import scipy.spatial
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

class project_verifier:
    def __init__(self, network_fn, model_fn, data,
                 device_id=0, output_name='feat_norm',
                 dist_metric='euclidean'):
        self._network_fn = network_fn
        self._model_fn = model_fn
        self._init_network(device_id)
        self._data = data
        self._img_count = data.shape[0]
        self._output_name = output_name
        self._dist_metric = dist_metric

    def _init_network(self, device_id=0):
        caffe.set_mode_gpu()
        caffe.set_device(device_id)
        self._net = caffe.Net(self._network_fn, self._model_fn,
                        caffe.TEST)
        k, v = self._net.blobs.items()[-1]
        self._output_dim = v.data.size / v.data.shape[0]

    """
    This function is used to get all feature vectors of data samples
    """
    def _data_worker(self):
        self._features = np.zeros((self._img_count, self._output_dim))
        # begin running network
        datum_idx = 0
        for datum in self._data:
            self._net.blobs['data'].data[...] = datum
            self._net.forward()
            self._features[datum_idx,...] = self._net.blobs[self._output_name].data[0,...].flatten()
            datum_idx += 1
            if datum_idx % 100 == 0:
                print "Has processed {} samples".format(datum_idx)

    """
    Calculate similarity matrix
    using pdist, and sim = 1-pdist
    """
    def _sim_calculator(self):
        if self._features is None:
            print "Error: should run get features first"
            raise
        else:
            self._sim_matrix = 1 - squareform(
                pdist(self._features,
                      metric=self._dist_metric))

    def process_data(self):
        self._data_worker()
        self._sim_calculator()

    def get_features(self):
        return self._features

    def get_sim_matrix(self):
        return self._sim_matrix

    """
    Evaluate PR curve using scipy
    """
    def evaluate(self, label_mat):
        precision, recall, _ = precision_recall_curve(label_mat.ravel(), self._sim_matrix.ravel())
        avg_p = average_precision_score(label_mat.ravel(), self._sim_matrix.ravel())
        return precision, recall, avg_p



import lmdb
"""
Prepare data for evaluation
read from backend and source
only read the first img_count samples
"""
def resize_img_data(img_data, resize):
    img_data = img_data.transpose(1, 2, 0)
    img_data = scipy.misc.imresize(img_data, (resize, resize))
    return img_data.transpose(2, 0, 1)


def decode_datum(datum):
    if datum.encoded is True:
        datum = caffe.io.decode_datum(datum)
    img_data = caffe.io.datum_to_array(datum)
    return img_data

def substract_mean(img, image_mean):
    if image_mean.ndim == 1:
        image_mean = image_mean[:, np.newaxis, np.newaxis]
    img -= image_mean
    return img

def read_data(source, img_count, mean_fn=None, resize=-1):
    if mean_fn is not None:
        blob = caffe_pb2.BlobProto()
        blob_str = open(mean_fn, 'rb').read()
        blob.ParseFromString(blob_str)
        image_mean = np.array(caffe.io.blobproto_to_array(blob))[0]
    db = lmdb.open(source)
    cur = db.begin().cursor()
    cur.first()
    data = []
    label = []
    for idx in range(img_count):
        value_str = cur.value()
        datum = caffe_pb2.Datum()
        datum.ParseFromString(value_str)
        label.append(int(datum.label))
        img_data = decode_datum(datum)
        if mean_fn is not None:
            img_data = substract_mean(img_data, image_mean)
        if resize > 0:
            img_data = resize_img_data(img_data, resize)
        data.append(img_data)
        if not cur.next():
            break
    _img_count = len(data)
    data = np.asarray(data)
    label = np.asarray(label)
    # now prepare label_matrix
    label_mat = np.zeros((_img_count, _img_count))
    for idx in range(_img_count):
        label_mat[idx, ...] = (label == label[idx])
    return data, label_mat


"""
Draw pr curves on a single image
"""
def draw_pr_curves(count, pr, avg_p_scores, legends, figure_fn):
    plt.clf()
    for idx in range(count):
        label_str = "%s: average precision = %.3f" % (legends[idx], avg_p_scores[idx])
        plt.plot(pr[idx]['r'], pr[idx]['p'], label=label_str)
    plt.title("PR Curves for different network settings")
    plt.legend(loc="upper right")
    # save to file
    plt.savefig(figure_fn)

"""
Both data and labels are flattened numpy matrix
"""
def draw_distribution(data, label, figure_fn):
    positive_idx = np.where(label==1)
    negative_idx = np.where(label==0)
    positives = data[positive_idx]
    negatives = data[negative_idx]
    plt.clf()
    plt.hist([1 - positives, 1 - negatives], normed=True, label=['positive', 'negative'], bins=100)
    plt.legend()
    plt.title('Positive / Negative Distribution')
    plt.savefig(figure_fn)

def main():
    # main function
    solver_dir = '/mnt/ilcompf0d0/user/xliu/code/caffe/examples/distilling/siamese_project'
    network_fn = [os.path.join(solver_dir, 'deploy_conv.prototxt'),
                  os.path.join(solver_dir, 'deploy_conv_r.prototxt'),
                  os.path.join(solver_dir, 'deploy_alexnet.prototxt'),
                  os.path.join(solver_dir, 'deploy_alexnet_conv5.prototxt')]
    model_fn = [os.path.join(solver_dir, 'siamese_project_3_iter_30000.caffemodel'),
                os.path.join(solver_dir, 'siamese_project_conv_r1_iter_10000.caffemodel'),
                os.path.join(solver_dir, 'bvlc_alexnet.caffemodel'),
                os.path.join(solver_dir, 'bvlc_alexnet.caffemodel')]

    legends = ['Train_FCN',
               'FCN_CONV1_INIT',
               'Alexnet',
               'Alexnet_Conv5']
    count = 1000
    source = '/mnt/ilcompf2d1/data/be/prepared-2015-06-15/LMDB/TESTING/image'
    mean_fn = '../behance.binaryproto'
    data, label_mat = read_data(source, count, mean_fn, resize=227)
    np.save('./project_verification_label_matrix', label_mat)
    pr = []
    average_p_score = []
    for idx in range(len(network_fn)):
        print "------------------Processing {}-----------------".format(legends[idx])
        verifier = project_verifier(network_fn[idx], model_fn[idx], data)
        print "------------------Processign data------------------"
        verifier.process_data()
        print "------------------Evaluation ---------------------"
        rslt = verifier.evaluate(label_mat)
        # save similarity matrix
        sim_matrix = verifier.get_sim_matrix()
        np.save(legends[idx]+'_sim_matrix', sim_matrix)
        pr_dict_ele = {}
        pr_dict_ele['p'] = rslt[0]
        pr_dict_ele['r'] = rslt[1]
        pr.append(pr_dict_ele)
        average_p_score.append(rslt[2])

        draw_distribution(sim_matrix.ravel(), label_mat.ravel(), figure_fn=legends[idx]+'_distribution.png')
    figure_fn = 'project_verification.png'
    draw_pr_curves(len(legends), pr, average_p_score, legends, figure_fn)


if __name__ == '__main__':
    sys.exit(main())
