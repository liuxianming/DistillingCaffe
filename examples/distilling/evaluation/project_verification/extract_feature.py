"""
Extract features
"""
import sys
import os, os.path
import numpy as np
CAFFE_ROOT_DIR = '/mnt/ilcompf0d0/user/xliu/code/caffe/'
sys.path.append(os.path.join(CAFFE_ROOT_DIR, 'python'))
sys.path.append('/mnt/ilcompf0d0/user/xliu/libs/python2.7/dist-packages')
import caffe
from caffe.io import caffe_pb2
import scipy.spatial
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import lmdb
import pickle

try:
    import matplotlib
    #matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except:
    print 'cannot import matplotlib'

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

class FeatureExtractor:
    def __init__(self, network_fn, model_fn,
                 device_id=0, output_name='feat_norm'):
        self._network_fn = network_fn
        self._model_fn = model_fn
        self._output_name = output_name
        self._init_network(device_id)

    def _init_network(self, device_id=0):
        caffe.set_mode_gpu()
        caffe.set_device(device_id)
        self._net = caffe.Net(self._network_fn, self._model_fn,
                        caffe.TEST)
        k, v = self._net.blobs.items()[-1]
        self._output_dim = v.data.size / v.data.shape[0]

    # extract feature for a specific decoded datum
    def sample_worker(self, sample):
        self._net.blobs['data'].data[0,...] = sample
        self._net.forward()
        return self._net.blobs[self._output_name].data[0,...].flatten()

    # extract feature for a datum
    def datum_worker(self, datum, image_mean=None, resize=-1):
        img_data = decode_datum(datum)
        if image_mean is not None:
            img_data = substract_mean(img_data, image_mean)
        if resize > 0:
            img_data = resize_img_data(img_data, resize)
        return self.sample_worker(img_data)

    """
    extract features for a databasse
    parameters:
    source: the source of lmdb
    count: the number of images need to process
    mean_file: the filename of image mean
    resize: if it needs resize, and what is the size. -1 means no resizing needed
    save_fn: the path and name to save features
    img_list_fn: the file where to save the path of each image
    """
    def lmdb_worker(self, source, count=100000,
                    mean_fn=None, resize=-1,
                    save_fn='./feature.npy',
                    label_fn=None,
                    img_list_fn=None,
                    img_root_dir=None, divider=10000):
        if mean_fn is not None:
            blob = caffe_pb2.BlobProto()
            blob_str = open(mean_fn, 'rb').read()
            blob.ParseFromString(blob_str)
            image_mean = np.array(caffe.io.blobproto_to_array(blob))[0]
        else:
            image_mean = None
        labels = []
        features = np.zeros((count, self._output_dim))
        img_path_list = []
        db = lmdb.open(source)
        with db.begin() as txn:
            cur = txn.cursor()
            cur.first()
            for idx in range(count):
                value_str = cur.value()
                datum = caffe_pb2.Datum()
                datum.ParseFromString(value_str)
                label = int(datum.label)
                labels.append(label)
                image_id = int(cur.key())
                features[idx, ...] = self.datum_worker(datum, image_mean, resize)
                img_path = self.parse_image_path(img_root_dir, divider, label, image_id)
                img_path_list.append(img_path)
                if idx % 10000 == 0:
                    print "Processed %d images" % idx

        if img_list_fn is not None:
            with open(img_list_fn, 'w') as f:
                f.write('\n'.join(img_path_list))
        if label_fn is not None:
            with open(label_fn, 'w') as f:
                pickle.dump(labels, f)
        #save feature
        print "Saving features to {}...".format(save_fn)
        np.save(save_fn, features)

    def parse_image_path(self, img_root_dir, divider,
                         project_id, image_id):
        sub_folder = project_id / divider
        img_fn = "{}_{}.jpg".format(project_id, image_id)
        img_path = os.path.join(img_root_dir, "{}".format(sub_folder), img_fn)
        return img_path


def main(argv):
    gpu_id = int(argv[1])
    task_id = int(argv[2])
    dataset = argv[3]

    mean_file = './behance.binaryproto'

    if dataset == 'train':
        source = '/mnt/ilcompf2d1/data/be/prepared-2015-06-15/LMDB/TRAINING/image'
        count = 100000
        savedir = './training_feature'
    elif dataset == 'test':
        source = '/mnt/ilcompf2d1/data/be/prepared-2015-06-15/LMDB/TEST/image'
        count = 50000
        savedir = './testing_feature'

    if not os.path.exists(savedir):
        os.mkdir(savedir)

    network_dir = './networks'
    model_dir = './models'
    network_fn = [os.path.join(network_dir, 'deploy_conv.prototxt'),
                  os.path.join(network_dir, 'deploy_conv.prototxt'),
                  os.path.join(network_dir, 'deploy_alexnet.prototxt'),
                  os.path.join(network_dir, 'deploy_alexnet.prototxt'),
                  os.path.join(network_dir, 'deploy_alexnet_conv5.prototxt'),
                  os.path.join(network_dir, 'deploy_alexnet_conv5.prototxt'),
                  os.path.join(network_dir, 'deploy_alexnet_conv3.prototxt')]
    model_fn = [os.path.join(model_dir, 'siamese_project_conv.caffemodel'),
                os.path.join(model_dir, 'triplet_project.caffemodel'),
                os.path.join(model_dir, 'alexnet_field_retrain_conv5.caffemodel'),
                os.path.join(model_dir, 'bvlc_alexnet.caffemodel'),
                os.path.join(model_dir, 'alexnet_field_retrain_conv5.caffemodel'),
                os.path.join(model_dir, 'bvlc_alexnet.caffemodel'),
                os.path.join(model_dir, 'bvlc_alexnet.caffemodel')]
    task_names = ['Siamese_Conv_Net',
                  'Triplet_Conv_Net',
                  'field_fc8',
                  'AlexNet_fc8',
                  'field_conv5',
                  'AlexNet_conv5',
                  'AlexNet_conv3']

    resize = [227, 227, 227, 227, 227, 227, 227]

    task_count = len(task_names)
    if task_id >= task_count:
        raise "Error: task id should be less than %d" % task_count

    # process task
    print "Processing Task [{}] on GPU={}".format(task_names[task_id], gpu_id)
    save_fn = os.path.join(savedir, "feature_{}.npy".format(task_names[task_id]))
    label_fn = os.path.join(savedir, "labels.npy")
    img_root_dir = '/mnt/ilcompf2d1/data/be/prepared-2015-06-15/images'
    feature_extractor_ = FeatureExtractor(
        network_fn[task_id], model_fn[task_id], gpu_id)
    if task_id == 0:
        img_list_fn = os.path.join(savedir, 'img_path_list.txt')
        feature_extractor_.lmdb_worker(
            source, count, mean_fn=mean_file, resize=resize[task_id],
            save_fn=save_fn, label_fn=label_fn,img_list_fn=img_list_fn,
            img_root_dir=img_root_dir)
    else:
        feature_extractor_.lmdb_worker(
            source, count, mean_fn=mean_file, resize=resize[task_id],
            save_fn=save_fn, label_fn=None, img_root_dir=img_root_dir)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
