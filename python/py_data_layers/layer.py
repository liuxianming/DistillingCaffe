import caffe
from caffe.io import caffe_pb2
import numpy as np
# from multiprocessing import Process, Queue
import lmdb
import yaml
import sys
import scipy.misc
import time
import np.random
import pickle

from multiprocessing import Process, Pipe

"""
Load data from either datum or from original image
Here, I used caffe.io.read_datum_from_image
to resolve all format transformation,
instead of the io functions provided by caffe python wrapper
"""


def extract_sample_from_datum(datum, image_mean, resize=-1):
    # extract numpy array from datum, then substract the mean_image
    img_data = decode_datum(datum)
    img_data = substract_mean(img_data, image_mean)
    if resize == -1:
        return img_data
    else:
        # resize image:
        # first transfer back to h * w * c, -> resize -> c * h * w
        return resize_img_data(img_data, resize)


def resize_img_data(img_data, resize):
    img_data = img_data.transpose(1, 2, 0)
    img_data = scipy.misc.imresize(img_data, (resize, resize))
    return img_data.transpose(2, 0, 1)


def decode_datum(datum):
    if datum.encoded is True:
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
        batch = np.zeros(self._top_shape)
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
Extract and prepare training / testing data
from data / label lmdb for multi-class classification,
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
            img_data = extract_sample_from_datum(datum, self._mean,
                                                 self._resize)
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
        print "Starting process for {} / {}".format(self._db_name,
                                                    self._label_db_name)

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
            img_data = extract_sample_from_datum(datum, self._mean,
                                                 self._resize)
            batch[idx, ...] = img_data
            label_batch[idx, ...] = self._label[self._pos]

            self._pos = (self._pos + 1) % self._n_samples
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


class PrefetchMultiClassDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        # add a field named "param_str" in python_param, in prototxt
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
            # using pipe() instead of queue to speed up
            self._conn, conn = Pipe()
            self._prefetch_process = BatchFetcher(
                conn, self._cur, self._label_cur,
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
        print "Starting process for {} / {}".format(self._db_name,
                                                    self._label_db_name)

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
    def __init__(self, conn,
                 img_db_cursor, label_cursor,
                 image_mean,
                 resize, top_shape):
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
        print "Staring a {} process".format(self._type())

    def _type(self):
        return 'BatchFetcher'

    def _preload_db(self):
        self._preload_data()
        self._preload_label()
        self._n_samples = len(self._data)
        self._pos = 0
        print 'Number of samples pre-loaded: {}'.format(self._n_samples)

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
        else:
            self._mean = image_mean
        # print "[Debug Info]: Image Mean Shape = {}".format(self._mean.shape)

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
        print "[{}]: Preloading image data done [{} second]".format(
            self._type(), end - start)

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
            img_data = extract_sample_from_datum(
                datum, self._mean, self._resize)
            batch[idx, ...] = img_data
            label_batch[idx, ...] = self._label[self._pos]

            self._pos = (self._pos + 1) % self._n_samples
        return (batch, label_batch)

    def run(self):
        print "BatchFetcher started!"
        self._preload_db()
        while True:
            start = time.time()
            batch = self._get_next_minibatch()
            end = time.time()
            print "[{} seconds] Generating a new batch".format(end-start)
            self._conn.send(batch)

"""pairlayer.py
This file implements python data layer for siamese / triplet training
To improve efficiency,
all layers use prefetching since sampling is time consuming
"""


class PrefetchSiameseLayer(caffe.Layer):
    def setup(self, bottom, top):
        # add a filed named "param_str" in python_param, in prototxt
        layer_params = yaml.load(self.param_str_)
        print "Using PrefetchSiameseLayer"
        try:
            self._batch_size = int(layer_params['batch_size'])
            if 'resize' in layer_params.keys():
                self._resize = layer_params['resize']
            else:
                self._resize = -1

            print "Setting up python data layer: [type = PrefetchSiameseLayer]"
            self._mean_file = layer_params['mean_file']
            self._db_name = layer_params['source']
            self._set_up_db()
            # reshape the top layer
            top[1].reshape(self._batch_size, 1, 1, 1)
            # fetch a datum from self._db to get size of images
            datum = self._get_a_datum(self._cur)
            img_data = decode_datum(datum)
            if self._resize > 0:
                img_data = resize_img_data(img_data, self._resize)
            # top[0] should be of size
            # (batch_size, channels * 2, height, width)
            top[0].reshape(self._batch_size, img_data.shape[0]*2,
                           img_data.shape[1], img_data.shape[2])
            self._top_data_shape = top[0].data.shape
            self._top_sim_shape = (self._batch_size, 1, 1, 1)
            # then return the cursor to the initial position
            self._cur.first()
            # using pipe() instead of queue to speed up
            self._conn, conn = Pipe()
            self._prefetch_process = SiameseBatchFetcher(
                conn, self._cur, self._mean_file,
                self._resize, self._top_data_shape)
            print "Start prefeteching process..."
            self._prefetch_process.start()

            def cleanup():
                print 'Terminating prefetching process working in backend'
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
        self._cur = self._db.begin().cursor()
        self._cur.first()

    def _get_next_minibatch(self):
        batch = self._conn.recv()
        return batch

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        blob = self._get_next_minibatch()
        top[0].data[...] = blob[0].astype(np.float32, copy=False)
        top[1].data[...] = blob[1].astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        pass


class SiameseBatchFetcher(Process):
    def __init__(self, conn, img_db_cursor, image_mean, resize, top_shape):
        super(SiameseBatchFetcher, self).__init__()
        self._conn = conn
        self._cur = img_db_cursor
        self._top_shape = top_shape
        self._batch_size = top_shape[0]
        self._top_label_shape = (self._batch_size, 1, 1, 1)
        self._set_mean(image_mean)
        self._data = []
        self._label = []
        self._resize = resize
        print "Staring a {} process".format(self._type())

    def _type(self):
        return 'SiameseBatchFetcher'

    def _preload_db(self):
        self._preload_data()
        self._n_samples = len(self._data)
        print 'Total number of samples pre-loaded: {}'.format(self._n_samples)

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
        else:
            self._mean = image_mean
        # print "[Debug Info]: Image Mean Shape = {}".format(self._mean.shape)

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
            self._label.append(datum.label)
            if not self._cur.next():
                break
        self._label = np.asarray(self._label)
        end = time.time()
        print "[{}]: Preloading image data done [{} second]".format(
            self._type(), end - start)

    def _sampling(self, query_id=-1, positive=True):
        # sampling function: given query sample id
        # To-Do: implement a more complex sampling function
        if query_id == -1:
            rand_id = np.random.randint(self._n_samples)
        else:
            query_label = self._label[query_id]
            if positive:
                candidates = np.where(self._label == query_label)[0]
            else:
                candidates = np.where(self._label != query_label)[0]
            rand_id = np.random.choice(candidates)
            #print "{} candidates found for query id {}".format(candidates.size, query_id)
            #print "{} / {}".format(query_id,  rand_id)
            #if rand_id != query_id:
            #    break
        return rand_id

    def _get_next_minibatch(self):
        batch = np.zeros(self._top_shape)
        label_batch = np.zeros(self._top_label_shape)
        # decode and return a tuple (data_batch, label_batch)
        for idx in range(self._batch_size / 2):
            q_id = self._sampling()
            # generate positive pair and a negative pair for each query sample
            ref_type = [True, False]
            datum = self._data[q_id]
            img_data = extract_sample_from_datum(
                datum, self._mean, self._resize)
            for type_offset in range(2):
                positive = ref_type[type_offset]
                datum_offset = 2 * idx + type_offset
                r_id = self._sampling(q_id, positive)
                r_datum = self._data[r_id]
                r_img_data = extract_sample_from_datum(
                    r_datum, self._mean, self._resize)
                batch[datum_offset, ...] = np.vstack([img_data, r_img_data])
                label_batch[datum_offset, ...] = 1 if positive else 0
        return (batch, label_batch)

    def run(self):
        print "BatchFetcher started!"
        self._preload_db()
        while True:
            batch = self._get_next_minibatch()
            self._conn.send(batch)


class SiameseDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        # add a filed named "param_str" in python_param, in prototxt
        layer_params = yaml.load(self.param_str_)
        print "Using PrefetchSiameseLayer"
        try:
            self._batch_size = int(layer_params['batch_size'])
            if 'resize' in layer_params.keys():
                self._resize = layer_params['resize']
            else:
                self._resize = -1

            # using random_flip to control the prob of randomly flipping the pair label
            if 'random_flip' in layer_params.keys():
                self._random_flip = float(layer_params['random_flip'])
            else:
                self._random_flip = 0.0
            print "Setting up python data layer: [type = {}]".format(self._type())
            self._mean_file = layer_params['mean_file']
            self._db_name = layer_params['source']
            self._set_up_db()
            # reshape the top layer
            top[1].reshape(self._batch_size, 1, 1, 1)
            # fetch a datum from self._db to get size of images
            datum = self._get_a_datum(self._cur)
            img_data = decode_datum(datum)
            if self._resize > 0:
                img_data = resize_img_data(img_data, self._resize)
            # top[0] should be of size
            # (batch_size, channels * 2, height, width)
            top[0].reshape(self._batch_size, img_data.shape[0]*2,
                           img_data.shape[1], img_data.shape[2])
            self._top_shape = top[0].data.shape
            self._top_label_shape = (self._batch_size, 1, 1, 1)
            # then return the cursor to the initial position
            self._cur.first()
            # read image_mean from file and preload all data into memory
            self._set_mean(self._mean_file)
            self._data = []
            self._label = []
            self._preload_db()
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
        self._cur = self._db.begin().cursor()
        self._cur.first()

    def _preload_db(self):
        self._preload_data()
        self._n_samples = len(self._data)
        print 'Total number of samples pre-loaded: {}'.format(self._n_samples)

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
            self._label.append(datum.label)
            if not self._cur.next():
                break
        self._label = np.asarray(self._label)
        end = time.time()
        print "[{}]: Preloading image data done [{} second]".format(
            self._type(), end - start)

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
        else:
            self._mean = image_mean

    def _type(self):
        return "SiameseDataLayer"

    def _sampling(self, query_id=-1, positive=True):
        # sampling function: given query sample id
        # To-Do: implement a more complex sampling function
        if query_id == -1:
            rand_id = np.random.randint(self._n_samples)
        else:
            query_label = self._label[query_id]
            if positive:
                candidates = np.where(self._label == query_label)[0]
            else:
                candidates = np.where(self._label != query_label)[0]
            rand_id = np.random.choice(candidates)
            #print "{} candidates found for query id {}".format(candidates.size, query_id)
            #print "{} / {}".format(query_id,  rand_id)
            #if rand_id != query_id:
            #    break
        return rand_id

    def _get_next_minibatch(self):
        batch = np.zeros(self._top_shape)
        label_batch = np.zeros(self._top_label_shape)
        # decode and return a tuple (data_batch, label_batch)
        for idx in range(self._batch_size / 2):
            q_id = self._sampling()
            # generate positive pair and a negative pair for each query sample
            ref_type = [True, False]
            datum = self._data[q_id]
            img_data = extract_sample_from_datum(
                datum, self._mean, self._resize)
            for type_offset in range(2):
                positive = ref_type[type_offset]
                datum_offset = 2 * idx + type_offset
                r_id = self._sampling(q_id, positive)
                r_datum = self._data[r_id]
                r_img_data = extract_sample_from_datum(
                    r_datum, self._mean, self._resize)
                batch[datum_offset, ...] = np.vstack([img_data, r_img_data])
                pair_label = 1 if positive else 0
                if self._random_flip > 0:
                    prob = np.random.rand(1)[0]
                    if prob < self._random_flip:
                        pair_label = 1 - pair_label
                label_batch[datum_offset, ...] = pair_label
        return (batch, label_batch)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        blob = self._get_next_minibatch()
        top[0].data[...] = blob[0].astype(np.float32, copy=False)
        top[1].data[...] = blob[1].astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        pass


"""
Add triplet data layer
"""
class TripletDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        # add a filed named "param_str" in python_param, in prototxt
        layer_params = yaml.load(self.param_str_)
        print "Using PrefetchSiameseLayer"
        try:
            self._batch_size = int(layer_params['batch_size'])
            if 'resize' in layer_params.keys():
                self._resize = layer_params['resize']
            else:
                self._resize = -1

            print "Setting up python data layer: [type = {}]".format(self._type())
            self._mean_file = layer_params['mean_file']
            self._db_name = layer_params['source']
            self._set_up_db()
            # reshape the top layer
            # top[1].reshape(self._batch_size, 1, 1, 1)
            # fetch a datum from self._db to get size of images
            datum = self._get_a_datum(self._cur)
            img_data = decode_datum(datum)
            if self._resize > 0:
                img_data = resize_img_data(img_data, self._resize)
            # top[0] should be of size
            # (batch_size, channels * 2, height, width)
            top[0].reshape(self._batch_size, img_data.shape[0]*3,
                           img_data.shape[1], img_data.shape[2])
            self._top_shape = top[0].data.shape
            # self._top_label_shape = (self._batch_size, 1, 1, 1)
            # then return the cursor to the initial position
            self._cur.first()
            # read image_mean from file and preload all data into memory
            self._set_mean(self._mean_file)
            self._data = []
            self._label = []
            self._preload_db()
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
        self._cur = self._db.begin().cursor()
        self._cur.first()

    def _preload_db(self):
        self._preload_data()
        self._n_samples = len(self._data)
        print 'Total number of samples pre-loaded: {}'.format(self._n_samples)

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
            self._label.append(datum.label)
            if not self._cur.next():
                break
        self._label = np.asarray(self._label)
        end = time.time()
        print "[{}]: Preloading image data done [{} second]".format(
            self._type(), end - start)

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
        else:
            self._mean = image_mean

    def _type(self):
        return "TripletDataLayer"

    def _sampling(self, query_id=-1, positive=True):
        # sampling function: given query sample id
        # To-Do: implement a more complex sampling function
        if query_id == -1:
            rand_id = np.random.randint(self._n_samples)
        else:
            query_label = self._label[query_id]
            if positive:
                candidates = np.where(self._label == query_label)[0]
            else:
                candidates = np.where(self._label != query_label)[0]
            rand_id = np.random.choice(candidates)
            #print "{} candidates found for query id {}".format(candidates.size, query_id)
            #print "{} / {}".format(query_id,  rand_id)
            #if rand_id != query_id:
            #    break
        return rand_id

    def _get_next_minibatch(self):
        batch = np.zeros(self._top_shape)
        # decode and return a tuple (data_batch, label_batch)
        for idx in range(self._batch_size):
            q_id = self._sampling()
            # generate positive pair and a negative pair for each query sample
            q_datum = self._data[q_id]
            img_data = extract_sample_from_datum(
                q_datum, self._mean, self._resize)
            # sampling positive ref datum
            p_id = self._sampling(q_id, positive=True)
            p_datum = self._data[p_id]
            p_img_data = extract_sample_from_datum(
                p_datum, self._mean, self._resize)

            # sampling negative ref datum
            n_id = self._sampling(q_id, positive=False)
            n_datum = self._data[n_id]
            n_img_data = extract_sample_from_datum(
                n_datum, self._mean, self._resize)
            batch[idx, ...] = np.vstack([img_data, p_img_data, n_img_data])
        return batch

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        blob = self._get_next_minibatch()
        top[0].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        pass


"""
Define a triplet sampling layer, given an sampling list / hash table
in param_str, there is a term to indicate where the list is located
More details:
What is the format of sampling table?
dict{key: list of candidate projects}
for each project, it is defined as dict{project_id: relevance_score}
to omit the relevance_score, just set all relevance_score as 1
"""
class TripletSamplingDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        # add a filed named "param_str" in python_param, in prototxt
        layer_params = yaml.load(self.param_str_)
        print "Using TripletSamplingDataLayer"
        try:
            self._batch_size = int(layer_params['batch_size'])
            if 'resize' in layer_params.keys():
                self._resize = layer_params['resize']
            else:
                self._resize = -1

            print "Setting up python data layer: [type = {}]".format(self._type())
            self._mean_file = layer_params['mean_file']
            self._db_name = layer_params['source']
            self._set_up_db()
            if 'sampling_table' in layer_params.keys():
                self._sample_table_source = layer_params['sampling_table']
                # ToDo: add a function to parse sampling_table
                self._parse_sampling_table()
                if 'hard_sampling' in layer_params.keys():
                    self._hard_sampling = True
                    self._hard_sampling_stage = int(layer_params['hard_sampling'])
                else:
                    self._hard_sampling = False
            else:
                self._sample_table_source = None
            # reshape the top layer
            # top[1].reshape(self._batch_size, 1, 1, 1)
            # fetch a datum from self._db to get size of images
            datum = self._get_a_datum(self._cur)
            img_data = decode_datum(datum)
            if self._resize > 0:
                img_data = resize_img_data(img_data, self._resize)
            # top[0] should be of size
            # (batch_size, channels * 2, height, width)
            top[0].reshape(self._batch_size, img_data.shape[0]*3,
                           img_data.shape[1], img_data.shape[2])
            self._top_shape = top[0].data.shape
            # self._top_label_shape = (self._batch_size, 1, 1, 1)
            # then return the cursor to the initial position
            self._cur.first()
            # read image_mean from file and preload all data into memory
            self._set_mean(self._mean_file)
            self._data = []
            self._label = []
            self._preload_db()
            # iter is used as a counter, to see when to do hard sampling
            self._iter = 0
        except ():
            print "Network Python Layer Definition Error"
            sys.exit

    # ToDo: 1. implement this function:
    # Guided sampling table is read into self._sampling_table
    # ToDo: 2. rewrite sampling function according to this sampling_table
    def _parse_sampling_table(self):
        with open(self._sample_table_source, 'rb') as f:
            sampling_table_ = pickle.load(f)
            self._sampling_table = list(sampling_table_.values())

    def _get_a_datum(self, cursor):
        value_str = cursor.value()
        datum = caffe_pb2.Datum()
        datum.ParseFromString(value_str)
        return datum

    def _set_up_db(self):
        self._db = lmdb.open(self._db_name)
        self._cur = self._db.begin().cursor()
        self._cur.first()

    def _preload_db(self):
        self._preload_data()
        self._n_samples = len(self._data)
        print 'Total number of samples pre-loaded: {}'.format(self._n_samples)

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
            self._label.append(datum.label)
            if not self._cur.next():
                break
        self._label = np.asarray(self._label)
        end = time.time()
        print "[{}]: Preloading image data done [{} second]".format(
            self._type(), end - start)

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
        else:
            self._mean = image_mean

    def _type(self):
        return "SamplingTripletDataLayer"

    # sample a project for max_trail times until a project is sampled
    # if nothing chosen, return -1
    def _sample_label(
            self, label_list,
            query_label=-1, max_trial=5):
        trial = 0
        while trial < max_trial:
            label = np.random.choice(label_list)
            trial += 1
            if query_label == -1 or label != query_label:
                return label
        return -1

    def _sample_image_id(self, label):
        candidates_list = np.where(self._label == label)[0]
        return np.random.choice(candidates_list)

    def _sampling(self, query_id=-1, positive=True):
        # sampling function: given query sample id
        # To-Do: implement a more complex sampling function
        if query_id == -1:
            rand_id = np.random.randint(self._n_samples)
        else:
            query_label = self._label[query_id]
            if positive:
                candidates = np.where(self._label == query_label)[0]
            else:
                candidates = np.where(self._label != query_label)[0]
            rand_id = np.random.choice(candidates)
            #print "{} candidates found for query id {}".format(candidates.size, query_id)
            #print "{} / {}".format(query_id,  rand_id)
            #if rand_id != query_id:
            #    break
        return rand_id

    # guided sampling function is used to sample a data sample
    # using self._sampling_table
    def _guided_sampling(self):
        """
        workflow:
         1. sampling an anchor project, and then choose an anchor image from the chosen project
         2. Choose positive: from the same project
         3. Sampling negative:
            3.1 if hard_sampling, then choose from a project from the same super-class
            3.2 if not hard_sampling, then choose from a project from different super-classes
        """
        # choose an anchor super-class
        anchor_super_class_ = np.random.randint(len(self._sampling_table))
        # then find an anchor project
        anchor_label_list = np.array(self._sampling_table[anchor_super_class_])
        anchor_label = self._sample_label(anchor_label_list)
        anchor_image_id = self._sample_image_id(anchor_label)
        positive_image_id = self._sample_image_id(anchor_label)

        # sample negative image
        if self._hard_sampling and self._iter > self._hard_sampling_stage:
            # sample hard negative: find project of the same super class, eg. field
            negative_label = self._sample_label(
                anchor_label_list, query_label=anchor_label)
        else:
            # sample simple negative: from a different super class
            negative_super_class_id = np.random.randint(
                len(self._sampling_table))
            negative_label_list = np.array(self._sampling_table[negative_super_class_id])
            negative_label = self._sample_label(
                negative_label_list, query_label=anchor_label)
        # if nothing sampled, return an empty triplet
        if negative_label == -1:
                return None
        else:
            # sample a negative image
            negative_image_id = self._sampele_image_id(negative_label)
        # extract image content from datum and save as an sample
        return self._get_triplet(
            anchor_image_id, positive_image_id, negative_image_id)

    def _get_triplet(self, anchor_image_id, positive_image_id, negative_image_id):
        a_datum = self._data[anchor_image_id]
        p_datum = self._data[positive_image_id]
        n_datum = self._data[negative_image_id]
        a_img_data = extract_sample_from_datum(
            a_datum, self._mean, self._resize)
        p_img_data = extract_sample_from_datum(
            p_datum, self._mean, self._resize)
        n_img_data = extract_sample_from_datum(
            n_datum, self._mean, self._resize)
        return np.vstack([a_img_data, p_img_data, n_img_data])

    def _get_next_minibatch(self):
        if self._sample_table_source is None:
            batch = np.zeros(self._top_shape)
            # decode and return a tuple (data_batch, label_batch)
            for idx in range(self._batch_size):
                q_id = self._sampling()
                # generate positive pair and a negative pair for each query sample
                p_id = self._sampling(q_id, positive=True)
                # sampling negative ref datum
                n_id = self._sampling(q_id, positive=False)
                batch[idx, ...] = self._get_triplet(q_id, p_id, n_id)
        else:
            # sampling according to the sampling_table
            # sampling function is writen in _guided_sampling()
            batch = np.zeros(self._top_shape)
            for idx in range(self._batch_size):
                # sample until find a non-empty triplet
                sample = self._guided_sampling()
                while sample is None:
                    sample = self._guide_sampling()
                batch[idx, ...] = sample
        # counter changes
        self._iter += 1
        return batch

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        blob = self._get_next_minibatch()
        top[0].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        pass
