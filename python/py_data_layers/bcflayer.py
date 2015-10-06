import caffe
from caffe.io import caffe_pb2
import numpy as np
import yaml
import sys
import scipy.misc
import time
import bcfstore
from cStringIO import StringIO
from PIL import Image


def extract_sample_from_imgstr(imgstr, image_mean=None, resize=-1):
    """Extract sample from image string

    Use StringIO and PIL.Image to implement
    """
    try:
        img_data = decode_imgstr(imgstr)
        if resize > 0:
            img_data = scipy.misc.imresize(img_data, (resize, resize))
        # change channel for caffe:
        img_data = img_data.transpose(2, 0, 1)
        # substract_mean
        if image_mean:
            img_data = substract_mean(img_data, image_mean)
        return img_data
    except:
        print sys.exc_info()[0], sys.exc_info()[1]
        return


def decode_imgstr(imgstr):
    img_data = np.array(Image.open(StringIO(imgstr)))
    return img_data


def substract_mean(img, image_mean):
    """Substract image mean from data sample

    image_mean is a numpy array,
    either 1 * 3 or of the same size as input image
    """
    if image_mean.ndim == 1:
        image_mean = image_mean[:, np.newaxis, np.newaxis]
    img -= image_mean
    return img


class BcfLayer(caffe.Layer):
    """BcfLayer:

    Read images from BCF format
    """
    def setup(self, bottom, top):
        """Setting up bcfdatalayer

        @Parameters in param_str_:
        batch_size:
        resize:
        mean_file: either a file name or a list of numbers
        source: database name
        label_source: database file name for labels
        label_subset: the file name of a subset of all labels,
                      indicates which labels will be used to train
        sample_rate: if were to use subset of labels, and will create
                     a background class, this is the sample rate to
                     assign images from other class to the background class
        background_class_id: the class id for background classes,
                             by default = 1000
        """

        layer_params = yaml.load(self.param_str_)
        print layer_params
        try:
            self._batch_size = int(layer_params['batch_size'])
            if 'resize' in layer_params.keys():
                self._resize = int(layer_params['resize'])
            else:
                self._resize = -1

            print "Setting up python data layer"
            if 'mean_file' in layer_params.keys():
                self._mean_file = layer_params['mean_file']
            else:
                self._mean_file = None
            self._set_mean(self._mean_file)

            """Settting up dbs
            db_name / source: the location of bcf file
            label_list_name / label_source: filename of the whole list
            label_subset_name / label_subset: filename of label subset
            """

            self._db_name = layer_params['source']
            self._label_list_name = layer_params['label_source']
            if 'label_subset' in layer_params.keys():
                self._label_subset_name = layer_params['label_subset']
            else:
                self._label_subset_name = ''

            # sample rate is used to sample background class images
            if 'sample_rate' in layer_params.keys():
                self._sample_rate = float(layer_params['sample_rate'])
            else:
                self._sample_rate = 0

            if 'background_class_id' in layer_params.keys():
                self._background_class_id = int(
                    layer_params['background_class_id'])
            else:
                self._background_class_id = 1000

            # begin to prepare data
            self._preload_all()

            # reshape the top layer
            top[1].reshape(self._batch_size, 1, 1, 1)
            # fetch a datum from self._db to get size of images
            datum = self._get_a_datum()
            img_data = extract_sample_from_imgstr(
                datum, self._mean, self._resize)
            print type(img_data)
            top[0].reshape(self._batch_size, *(img_data.shape))
            self._top_data_shape = top[0].data.shape
            self._top_label_shape = (self._batch_size, 1, 1, 1)
            # then return the cursor to the initial position
        except ():
            print "Network Python Layer Definition Error"
            sys.exit
        # set mean and preload all data into memory

    def _set_mean(self, image_mean):
        if image_mean is None:
            self._mean = None
        elif type(image_mean) is str:
            # read image mean from file
            try:
                # if it is a pickle file
                self._mean = np.load(image_mean)
            except (IOError):
                blob = caffe_pb2.BlobProto()
                blob_str = open(image_mean, 'rb').read()
                blob.ParseFromString(blob_str)
                self._mean = np.array(caffe.io.blobproto_to_array(blob))[0]
            # self.mean = self.mean.transpose(1,2,0)
        else:
            self._mean = image_mean

    def _preload_all(self):
        """preload_project_db:

        This function preload all datum into memory
        Use bcfstore to load data from bcf file
        """
        print('Preloading BCF file from %s ' % self._db_name)

        bcf = bcfstore.bcf_store_file(self._db_name)
        print('Preloading all labels from %s ' % self._label_list_name)
        labels = np.loadtxt(self._label_list_name).astype(int)

        if bcf.size() != len(labels):
            print "Number of samples in data and labels are not equal"
            raise
        else:
            # see if need to filter out some classes
            if self._label_subset_name != '':
                self._label_subset = np.loadtxt(
                    self._label_subset_name).astype(int)
            self._data = []
            self._labels = []
            idx = 0
            for label in labels:
                if self._label_subset_name != '':
                    if label in self._label_subset:
                        self._data.append(bcf.get(idx))
                        self._labels.append(labels[idx])
                    elif self._sample_rate > 0:
                        # sample other images into the background class
                        if np.random.rand() <= self._sample_rate:
                            self._data.append(bcf.get(idx))
                            self._labels.append(self._background_class_id)
                else:
                    self._data.append(bcf.get(idx))
                    self._labels.append(labels[idx])
                idx += 1
            self._n_samples = len(self._data)
            print("Totally {} samples loaded".format(self._n_samples))
            self._cur = 0

    def _get_a_datum(self):
        return self._data[self._cur]

    def _get_next_minibatch(self):
        start = time.time()
        batch = np.zeros(self._top_data_shape)
        label_batch = np.zeros(self._top_label_shape)
        # decode and return a tuple (data_batch, label_batch)
        for idx in range(self._batch_size):
            datum = self._data[self._cur]
            img_data = extract_sample_from_imgstr(
                datum, self._mean, self._resize)
            batch[idx, ...] = img_data
            label_batch[idx, ...] = self._labels[self._cur]

            self._cur = (self._cur + 1) % self._n_samples
        end = time.time()
        print "Get a batch costs [{} seconds]".format(end-start)
        return (batch, label_batch)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        start = time.time()
        blob, label_blob = self._get_next_minibatch()
        # by default, caffe use float instead of double
        top[0].data[...] = blob.astype(np.float32, copy=False)
        top[1].data[...] = label_blob.astype(np.float32, copy=False)
        end = time.time()
        print "One iteration of forward: {} seconds".format(end-start)

    def backward(self, top, propagate_down, bottom):
        pass
