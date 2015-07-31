import caffe
from caffe.io import caffe_pb2
import numpy as np
#from multiprocessing import Process, Queue
import lmdb
import yaml
import sys
import time
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

"""pairlayer.py
This file implements python data layer for siamese / triplet training
To improve efficiency, all layers use prefetching since sampling is time consuming
"""

class PrefetchSiameseLayer(caffe.Layer):
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
            self._set_up_db()
            # reshape the top layer
            top[1].reshape(self._batch_size, 1, 1, 1)
            # fetch a datum from self._db to get size of images
            datum = self._get_a_datum(self._cur)
            img_data = decode_datum(datum)
            if self._resize > 0:
                img_data = resize_img_data(img_data, self._resize)
            # top[0] should be of size (batch_size, channels * 2, height, width)
            top[0].reshape(self._batch_size, img_data.shape[0]*2, img_data.shape[1], img_data.shape[2])
            self._top_data_shape = top[0].data.shape
            self._top_sim_shape = (self._batch_size, 1, 1, 1)
            # then return the cursor to the initial position
            self._cur.first()
            # using pipe() instead of queue to speed up
            self._conn, conn = Pipe()
            self._prefetch_process = SiameseBatchFetcher(conn, self._cur, self._mean_file,
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
        start = time.time()
        batch = self._conn.recv()
        end = time.time()
        print "Retrieve a prefetech batch costs {} seconds".format(end-start)
        return batch

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        blob, label_blob = self._get_next_minibatch()
        top[0].data[...] = blob.astype(np.float32, copy=False)
        top[1].data[...] = label_blob.astype(np.float32, copy=False)

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
        self._preload_db()

    def _preload_db(self):
        self._preload_data()
        self._n_samples = len(self._data)

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
            self._label.append(datum.label)
            if not self._cur.next():
                break
        self._label = np.asarray(self._label)
        end = time.time()
        print "Preloading image data done [{} second]".format(end-start)

    def _sampling(self, query_id=-1, positive=True):
        # sampling function: given query sample id
        # To-Do: implement a more complex sampling function
        if query_id == -1:
            rand_id = np.random.randint(self._n_samples)
        else:
            while True:
                query_label = self._label[query_id]
                if positive:
                    candidates = np.where(self._label == query_label)[0]
                else:
                    candidates = np.where(self._label != query_label)[0]
                rand_id = np.random.choice(candidates)
                if rand_id != query_id:
                    break
        return rand_id

    def _get_next_minibatch(self):
        batch = np.zeros(self._top_shape)
        label_batch = np.zeros(self._top_label_shape)
        # decode and return a tuple (data_batch, label_batch)
        for idx in range(self._batch_size / 2):
            q_id = self._sampling()
            print q_id
            # generate positive pair and a negative pair for each query sample
            ref_type = [True, False]
            datum = self._data[q_id]
            img_data  = extract_sample_from_datum(datum, self._mean, self._resize)
            for positive in ref_type:
                r_id = self._sampling(q_id, positive)
                r_datum = self._data[r_id]
                r_img_data = extract_sample_from_datum(r_datum, self._mean, self._resize)
                batch[idx, ...] = np.vstack([img_data, r_img_data])
                label_batch[idx, ...] = 1 if positive else 0
        return (batch, label_batch)

    def run(self):
        print "BatchFetcher started!"
        while True:
            start = time.time()
            print "Prefetch a new batch..."
            batch = self._get_next_minibatch()
            end = time.time()
            print "Prefetch a batch costs {} seconds".format(end - start)
            self._conn.send(batch)
