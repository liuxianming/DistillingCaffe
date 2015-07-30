import caffe
from caffe.io import caffe_pb2
import numpy as np
#from multiprocessing import Process, Queue
import lmdb
import yaml
import sys
import scipy.misc
import time

from multiprocessing import Process, Pipe

"""Some util function
"""
"""
Load data from either datum or from original image
Here, I used caffe.io.read_datum_from_image to resolve all format transformation,
instead of the io functions provided by caffe python wrapper
"""
def extract_sample_from_datum(datum, image_mean, resize=-1):
    # extract numpy array from datum, then substract the mean_image
    img_data = decode_datum(datum)
    img_data = substract_mean(img_data, image_mean)
    if resize == -1:
        return img_data
    else:
        # resize image: first transfer back to h * w * c, -> resize -> c * h * w
        resize_img_data(img_data, resize)

def resize_img_data(img_data, resize):
    img_data = img_data.transpose(1,2,0)
    img_data = scipy.misc.imresize(img_data, (resize, resize))
    return img_data.transpose(2,0,1)

def decode_datum(datum):
    if datum.encoded == True:
        datum = caffe.io.decode_datum(datum)
    img_data = caffe.io.datum_to_array(datum)
    return img_data

def extract_sample_from_image(image_fn, image_mean):
    # extract image data from image file directly,
    # return a numpy array, organized in the format of channel * height * width
    datum = caffe.io.read_datum_from_image(image_fn, encoding='')
    img_data = caffe.io.datum_to_array(datum)
    img_data = substract_mean(img_data, image_mean)
    return img_data

"""
Substract image mean from data sample
image_mean is a numpy array, either 1 * 3 or of the same size as input image
"""
def substract_mean(img, image_mean):
    if image_mean.ndim == 1:
        image_mean = image_mean[:, np.newaxis, np.newaxis]
    img -= image_mean
    return img

"""--------------------------------------------------------------------------"""

"""Expand_Data_Layer
Expand compact labels / vectors into flat ones
Example: from [1, 2, 4] ->[0, 1, 1, 0, 1, ...]
"""
class ExpandLayer(caffe.Layer):
    """Expand compact labels for training / testing"""

    def setup(self, bottom, top):
        # add a filed named "param_str" in python_param, in prototxt
        layer_params = yaml.load(self.param_str_)
        self._n_out = int(layer_params['n_out'])
        self._batch_size = int(layer_params['batch_size'])
        self._top_shape = (self._batch_size, 1, 1, self._n_out)

        print "Setting up python data layer"
        if 'source' not in layer_params.keys():
            print "Must assign a db as source: ***"
            sys.exit
        self._db_name = layer_params['source']
        top[0].reshape(self._batch_size, 1, 1, self._n_out)
        self._set_up_db()

    def _set_up_db(self):
        self._db = lmdb.open(self._db_name)
        self._cur = self._db.begin().cursor()
        self._cur.first()
        print "Starting process for {}".format(self._db_name)


    def _get_mini_batch(self):
        batch = []
        for idx in range(self._batch_size):
            value_str = self._cur.value()
            datum = caffe_pb2.Datum()
            datum.ParseFromString(value_str)
            batch.append(caffe.io.datum_to_array(datum))
            # if it is the end of db, go to the first
            if not self._cur.next():
                self._cur.first()
        return batch

    def _get_next_minibatch(self):
        #print 'ExpandLabelFetcher started...'
        batch = np.zeros(self._top_shape)
        #print batch.shape
        mini_batch = self._get_mini_batch()
        if len(mini_batch) != self._top_shape[0]:
            print "Batch size can not match"
            raise
        for i in range(len(mini_batch)):
            label = mini_batch[i].flatten()
            for idx in label:
                batch[i, 0, 0, idx] = 1.0 / label.shape[0]
        return batch

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector"""
        blob = self._get_next_minibatch()
        top[0].reshape(*(blob.shape))
        # by default, caffe use float instead of double
        top[0].data[...] = blob.astype(np.float32, copy=False)
        # save to file for debugging
        # np.save('batch', top[0].data)

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        top_shape = (self._batch_size, 1, 1, self._n_out)
        top[0].reshape(top_shape[0], top_shape[1], top_shape[2], top_shape[3])

"""BClassificationDataLayer:
Extract and prepare training / testing data from data / label lmdb for multi-class classification,
"""
class MultiClassDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        # add a filed named "param_str" in python_param, in prototxt
        layer_params = yaml.load(self.param_str_)
        try:
            self._batch_size = int(layer_params['batch_size'])
            if 'resize' in layer_params.keys():
                self._resize = layer_params['resize']
            else:
                self._resize = -1

            print "Setting up python data layer"
            self._mean_file = layer_params['mean_file']
            self._set_mean(self._mean_file)
            # settting up dbs
            self._db_name = layer_params['source']
            self._label_db_name = layer_params['label_source']
            self._set_up_db()
            # reshape the top layer
            top[1].reshape(self._batch_size, 1, 1, 1)
            # fetch a datum from self._db to get size of images
            datum = self._get_a_datum(self._cur)
            img_data  = extract_sample_from_datum(datum, self._mean, self._resize)
            top[0].reshape(self._batch_size, *(img_data.shape))
            self._top_data_shape = top[0].data.shape
            self._top_label_shape = (self._batch_size, 1, 1, 1)
            # then return the cursor to the initial position
            self._cur.first()
            self._data = []
            self._label = []
        except ():
            print "Network Python Layer Definition Error"
            sys.exit
        # set mean and preload all data into memory
        self._preload_all()

    def _set_mean(self, image_mean):
        if type(image_mean) is str:
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
        print "[Debug Info]: Image Mean Shape = {}".format(self._mean.shape)

    def _get_a_datum(self, cursor):
        value_str = cursor.value()
        datum = caffe_pb2.Datum()
        datum.ParseFromString(value_str)
        return datum

    def _set_up_db(self):
        self._db = lmdb.open(self._db_name)
        self._label_db = lmdb.open(self._label_db_name)
        self._cur = self._db.begin().cursor()
        self._label_cur = self._label_db.begin().cursor()
        self._cur.first()
        self._label_cur.first()
        print "Starting process for {} / {}".format(self._db_name, self._label_db_name)

    """preload_project_db:
    This function preload all datum into memory"""
    def _preload_all(self):
        self._preload_data()
        self._preload_label()
        print len(self._data)
        print len(self._label)
        if len(self._data) != len(self._label):
            print "Number of samples in data and labels are not equal"
            raise
        else:
            self._n_samples = len(self._data)
            self._pos = 0

    def _preload_data(self):
        # load data into memory but don't decode them
        # decode datum online when generating mini-batch
        print "Preloading data from LMDB..."
        start = time.time()
        self._cur.first()
        while True:
            value_str = self._cur.value()
            datum = caffe_pb2.Datum()
            datum.ParseFromString(value_str)
            self._data.append(datum)
            if not self._cur.next():
                break
        end = time.time()
        print "Preloading image data done [{} second]".format(end-start)

    def _preload_label(self):
        # use the first label as output,
        # turn a mutli-label problem into a multi-class problem
        self._label_cur.first()
        while True:
            value_str = self._label_cur.value()
            datum = caffe_pb2.Datum()
            datum.ParseFromString(value_str)
            label_vec = caffe.io.datum_to_array(datum).flatten()
            self._label.append(label_vec[0])

            if not self._label_cur.next():
                break
        print "Preloading labels done"

    def _get_next_minibatch(self):
        start = time.time()
        batch = np.zeros(self._top_data_shape)
        label_batch = np.zeros(self._top_label_shape)
        # decode and return a tuple (data_batch, label_batch)
        for idx in range(self._batch_size):
            datum = self._data[self._pos]
            img_data  = extract_sample_from_datum(datum, self._mean, self._resize)
            batch[idx, ...] = img_data
            label_batch[idx, ...] = self._label[self._pos]

            self._pos =( self._pos + 1) % self._n_samples
        end = time.time()
        print "Get a batch costs [{} seconds]".format(end-start)
        return (batch, label_batch)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        start = time.time()
        blob, label_blob = self._get_next_minibatch()
        #blob.reshape(*(top[0].data.shape))
        #label_blob.reshape(*(top[1].shape))
        # by default, caffe use float instead of double
        top[0].data[...] = blob.astype(np.float32, copy=False)
        top[1].data[...] = label_blob.astype(np.float32, copy=False)
        end = time.time()
        print "One iteration of forward: {} seconds".format(end-start)

    def backward(self, top, propagate_down, bottom):
        pass

class PrefetchMultiClassDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        # add a filed named "param_str" in python_param, in prototxt
        layer_params = yaml.load(self.param_str_)
        try:
            self._batch_size = int(layer_params['batch_size'])
            if 'resize' in layer_params.keys():
                self._resize = layer_params['resize']
            else:
                self._resize = -1

            print "Setting up python data layer"
            self._mean_file = layer_params['mean_file']
            self._db_name = layer_params['source']
            self._label_db_name = layer_params['label_source']
            self._set_up_db()
            # reshape the top layer
            top[1].reshape(self._batch_size, 1, 1, 1)
            # fetch a datum from self._db to get size of images
            datum = self._get_a_datum(self._cur)
            img_data = decode_datum(datum)
            if self._resize > 0:
                img_data = resize_img_data(img_data, self._resize)
            top[0].reshape(self._batch_size, *(img_data.shape))
            self._top_data_shape = top[0].data.shape
            self._top_label_shape = (self._batch_size, 1, 1, 1)
            # then return the cursor to the initial position
            self._cur.first()
            """
            # prepare the prefetech process
            self._blob_queue = Queue(30)
            self._prefetch_process = BatchFetcher(self._blob_queue, self._cur, self._label_cur,
                                                  self._mean_file, self._resize, self._top_data_shape)
            """
            # using pipe() instead of queue to speed up
            self._conn, conn = Pipe()
            self._prefetch_process = BatchFetcher(conn, self._cur, self._label_cur,
                                                  self._mean_file, self._resize, self._top_data_shape)
            print "Start prefeteching process..."
            self._prefetch_process.start()
            def cleanup():
                print 'Terminating BatchFetcher Process working in backend'
                self._prefetch_process.terminate()
                self._prefetch_process.join()
                self._conn.close()
            import atexit
            atexit.register(cleanup)
        except ():
            print "Network Python Layer Definition Error"
            sys.exit

    def _get_a_datum(self, cursor):
        value_str = cursor.value()
        datum = caffe_pb2.Datum()
        datum.ParseFromString(value_str)
        return datum

    def _set_up_db(self):
        self._db = lmdb.open(self._db_name)
        self._label_db = lmdb.open(self._label_db_name)
        self._cur = self._db.begin().cursor()
        self._label_cur = self._label_db.begin().cursor()
        self._cur.first()
        self._label_cur.first()
        print "Starting process for {} / {}".format(self._db_name, self._label_db_name)


    def _get_next_minibatch(self):
        batch = self._conn.recv()
        return batch

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        blob, label_blob = self._get_next_minibatch()
        top[0].data[...] = blob.astype(np.float32, copy=False)
        top[1].data[...] = label_blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        pass


class BatchFetcher(Process):
    def __init__(self, conn, img_db_cursor, label_cursor, image_mean, resize, top_shape):
        super(BatchFetcher, self).__init__()
        self._conn = conn
        self._cur = img_db_cursor
        self._label_cur = label_cursor
        self._top_shape = top_shape
        self._batch_size = top_shape[0]
        self._top_label_shape = (self._batch_size, 1, 1, 1)
        self._set_mean(image_mean)
        self._label = []
        self._data = []
        self._resize = resize
        self._preload_db()

    def _preload_db(self):
        self._preload_data()
        self._preload_label()
        self._n_samples = len(self._data)
        self._pos = 0

    def _set_mean(self, image_mean):
        if type(image_mean) is str:
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
        print "[Debug Info]: Image Mean Shape = {}".format(self._mean.shape)

    def _preload_data(self):
        # load data into memory but don't decode them
        # decode datum online when generating mini-batch
        print "Preloading data from LMDB..."
        start = time.time()
        self._cur.first()
        while True:
            value_str = self._cur.value()
            datum = caffe_pb2.Datum()
            datum.ParseFromString(value_str)
            self._data.append(datum)
            if not self._cur.next():
                break
        end = time.time()
        print "Preloading image data done [{} second]".format(end-start)

    def _preload_label(self):
        # use the first label as output,
        # turn a mutli-label problem into a multi-class problem
        self._label_cur.first()
        while True:
            value_str = self._label_cur.value()
            datum = caffe_pb2.Datum()
            datum.ParseFromString(value_str)
            label_vec = caffe.io.datum_to_array(datum).flatten()
            self._label.append(label_vec[0])

            if not self._label_cur.next():
                break
        print "Preloading labels done"

    def _get_next_minibatch(self):
        batch = np.zeros(self._top_shape)
        label_batch = np.zeros(self._top_label_shape)
        # decode and return a tuple (data_batch, label_batch)
        for idx in range(self._batch_size):
            datum = self._data[self._pos]
            img_data  = extract_sample_from_datum(datum, self._mean, self._resize)
            batch[idx, ...] = img_data
            label_batch[idx, ...] = self._label[self._pos]

            self._pos =( self._pos + 1) % self._n_samples
        return (batch, label_batch)

    def run(self):
        print "BatchFetcher started!"
        while True:
            batch = self._get_next_minibatch()
            self._conn.send(batch)
