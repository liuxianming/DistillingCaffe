#Test performance of trained model
import matplotlib
matplotlib.use('Agg')
import numpy as np
import lmdb
import sys
import os, os.path
#from contextlib import nested
import scipy.misc

CAFFE_ROOT = '/mnt/ilcompf0d0/user/xliu/code/caffe/python'
sys.path.append(CAFFE_ROOT)
import caffe
from caffe.io import caffe_pb2
import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics

def get_net(network_fn, model_fn, mode='CPU'):
    if mode == 'CPU':
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
    net = caffe.Net(network_fn, model_fn,
                    caffe.TEST)
    return net

def classify_sample(net, sample, output_name='output'):
    # classify a data sample using current net
    net.blobs['data'].data[0,...] = sample.astype(np.float32, copy=False)
    out = net.forward()
    #out = net.blobs[output_name].data.flatten()
    out = out[output_name].flatten()
    out /= np.amax(out)
    print out
    return out

"""
Load data from either datum or from original image
Here, I used caffe.io.read_datum_from_image to resolve all format transformation,
instead of the io functions provided by caffe python wrapper
"""
def extract_sample_from_datum(datum, image_mean, resize=-1):
    # extract numpy array from datum, then substract the mean_image
    if datum.encoded == True:
        # by default, the decoding include color space transform, to RGB
        datum = caffe.io.decode_datum(datum)
    img_data = caffe.io.datum_to_array(datum)
    img_data = substract_mean(img_data, image_mean)
    if resize == -1:
        return img_data
    else:
        # resize image: first transfer back to h * w * c, -> resize -> c * h * w
        img_data = img_data.transpose(1,2,0)
        img_data = scipy.misc.imresize(img_data, (resize, resize))
        return img_data.transpose(2,0,1)

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

# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, figure_fn=None, padsize=1, padval=0, color_map='gray'):
    data -= data.min()
    data /= data.max()
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    imgplot = plt.imshow(data)
    # set colormap, only for single channel images
    if data.ndim == 2:
        imgplot.set_cmap(color_map)
    if figure_fn is None:
        plt.show()
    else:
        plt.savefig(figure_fn)

def visualize_network_filter(network_fn, model_fn, base_dir='./', figure_fn=None, layer='conv1'):
    os.chdir(base_dir)
    print 'Chaning to dir = {}'.format(base_dir)
    net = get_net(os.path.join(base_dir, network_fn), os.path.join(base_dir, model_fn))
    plt.clf()
    if layer not in net.params.keys():
        print "Wrong Layer Name: {}".format(layer)
        return
    filters = net.params[layer][0].data
    #vis_square(filters.transpose(0,2,3,1), figure_fn)
    if filters.shape[1] == 3:
        vis_square(filters.transpose(0,2,3,1), figure_fn)
    else:
        filters = filters[:, 0:3, :, :]
        # filters = filters.reshape((filters.shape[0] * filters.shape[1], 1, filters.shape[2], filters.shape[3]))
        vis_square(filters.transpose(0,2,3,1), figure_fn)

class Data_CaffeNet_Classifier:
    """
    classifiy all data samples using caffe netowrk
    data samples are stored in self.data, while labels are stored in self.label
    """
    def __init__(self, data_lmdb=None, label_lmdb=None, mean=None, resize=-1):
        self.data_lmdb = data_lmdb
        self.label_lmdb = label_lmdb
        self.mean = mean
        self.resize = resize
        self.mean = None
        self.net = None
        self.data = None
        self.label = None
        self.classifications = None
        self.expanded_label_array = None
        self.compressed_label = True
        self.transformer = None

    def set_mean(self, image_mean):
        if type(image_mean) is str:
            # read image mean from file
            print "Reading Image Mean from file = {}".format(image_mean)
            try:
                # if it is a pickle file
                self.mean = np.load(image_mean)
            except (IOError):
                blob = caffe_pb2.BlobProto()
                blob_str = open(image_mean, 'rb').read()
                blob.ParseFromString(blob_str)
                self.mean = np.array(caffe.io.blobproto_to_array(blob))[0]
            # self.mean = self.mean.transpose(1,2,0)
        else:
            self.mean = image_mean
        print "[Debug Info]: Image Mean Shape = {}".format(self.mean.shape)

    def set_network(self, network_fn, model_fn, mode='GPU', input_size=(3,256,256)):
        self.network_fn = network_fn
        self.model_fn = model_fn
        print "Network Filename: " + self.network_fn
        print "Model Filename: " + self.model_fn
        #self.net = get_net(network_fn, model_fn, mode)
        if mode == 'GPU':
            caffe.set_mode_gpu()
        elif mode == 'CPU':
            caffe.set_mode_cpu()
        self.net = caffe.Net(self.network_fn, self.model_fn, caffe.TEST)
        self.net.blobs['data'].reshape(1, *(input_size))

    def visualize_filter(self, figure_fn=None, layer='conv1'):
        plt.clf()
        if self.net is None:
            print "Must setup network before visualization"
            return
        if layer not in self.net.params.keys():
            print "Wrong Layer Name: {}".format(layer)
            return
        filters = self.net.params[layer][0].data
        if filters.shape[1] == 3:
            vis_square(filters.transpose(0,2,3,1), figure_fn)
        else:
            filters = filters[:, 0:3, :, :]
            # filters = filters.reshape((filters.shape[0] * filters.shape[1], 1, filters.shape[2], filters.shape[3]))
            vis_square(filters.transpose(0,2,3,1), figure_fn)

    def visualize_featuremap(self, layer_name, sample_id=0):
        sample = self.data[sample_id]
        self.net.blobs['data'].data[...] = np.ascontiguousarray(sample)
        # print self.net.blobs['data'].data.flags
        self.net.forward()
        feature_map = self.net.blobs[layer_name].data[0]
        vis_square(feature_map, figure_fn = "{}_{}.jpg".format(layer_name, sample_id))

    def _classify_sample(self, sample, output_name='output', normalize=True):
        self.net.blobs['data'].data[...] = sample
        self.net.forward()
        rslt = self.net.blobs[output_name].data[0].flatten()
        if normalize:
            rslt = rslt / np.amax(rslt)
        return rslt

    def classify_dataset(self, normalize=True, output_fn=None, output_name='output'):
        classifications = []
        for idx in range(len(self.data)):
            img_data = self.data[idx]
            img_data = np.ascontiguousarray(img_data)
            #np.save('{}'.format(idx), img_data)
            rslt = self._classify_sample(img_data, output_name=output_name, normalize=normalize)
            classifications.append(rslt)
            # monitor the progress of classification
            if idx % 1000 == 0:
                print "{} samples classified".format(idx)
        # turn into numpy array
        self.classifications = np.asarray(classifications)
        if output_fn is not None:
            # save the classification results to outside file
            np.save(output_fn, self.classifications)
            print "Classification Results are Saved to {}".format(output_fn)

    def expand_label_array(self, output_fn=None):
        if (self.compressed_label == True) and (self.expanded_label_array is None):
            if self.classifications is not None:
                self.expanded_label_array = np.zeros(self.classifications.shape)
                for idx in range(len(self.label)):
                    label = self.label[idx]
                    for k in label:
                        self.expanded_label_array[idx, k] = 1
                if output_fn is not None:
                    # save expanded label array to output file
                    np.save(output_fn, self.expanded_label_array)
            else:
                return
        else:
            self.expanded_label_array = self.label
            self.expanded_label_array = np.array(self.expanded_label_array)

    """
    Functions to calculate performances, including P-R curve,
    """
    def get_prs(self, expanded_label_fn=None):
        self.p = dict()
        self.r = dict()
        self.average_precision = dict()
        self.auc = dict()

        # if labels are compressed, expand first
        print "Expand Compressed Labels..."
        Data_CaffeNet_Classifier.expand_label_array(self, expanded_label_fn)

        n_class = self.classifications.shape[1]
        for i in range(n_class):
            y_pred = self.classifications[:, i]
            y = self.expanded_label_array[:,i]
            self.p[i], self.r[i], _ = sklearn.metrics.precision_recall_curve(y, y_pred)
            self.average_precision[i] = sklearn.metrics.average_precision_score(y, y_pred)
            self.auc[i] = sklearn.metrics.auc(self.r[i], self.p[i], reorder=True)
        # compute the overall measurements
        self.p['micro'], self.r['micro'], _ = \
                                              sklearn.metrics.precision_recall_curve(
                                                  self.expanded_label_array.ravel(),
                                                  self.classifications.ravel())
        self.average_precision['overall'] = sklearn.metrics.average_precision_score(
            self.expanded_label_array, self.classifications, average='micro')
        self.auc['micro'] = sklearn.metrics.auc(self.r['micro'], self.p['micro'], reorder=True)

        print "===============[Evaluation Results]==============="
        print "[Model Micro AUC] = {}".format(self.auc['micro'])

    def plot_curve(self, x, y, label, title, figure_fn=None):
        plt.clf()
        plt.plot(x, y, label=label)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(title)
        plt.legend(loc='upper right')
        if figure_fn is None:
            plt.show()
        else:
            plt.savefig(figure_fn)


    def set_data(self, data_lmdb, label_lmdb=None):
        self.data_lmdb = data_lmdb
        self.label_lmdb = label_lmdb


    def load_data(self, data_lmdb, image_mean, label_lmdb=None, compressed_label=True, max_count=-1):
        self.set_data(data_lmdb, label_lmdb)
        Data_CaffeNet_Classifier.set_mean(self, image_mean)
        self.data = []
        self.label = []

        print "Reading all data into memory..."
        print "****************INFO*********************"
        print "\t Reading data from {}".format(self.data_lmdb)
        print "\t Substracting image mean from {}".format(image_mean)
        img_data = self._load_from_lmdb(self.data_lmdb, max_count)

        # Decode img_data
        for datum in img_data:
            img = extract_sample_from_datum(datum, self.mean, self.resize)
            # img = extract_sample_from_datum(datum)
            self.data.append(img)
        if self.label_lmdb is not None:
            self.load_label(self.label_lmdb, compressed_label, max_count)
        else:
            print "\t Reading label from {}".format(self.data_lmdb)
            self.compressed_label = True
            for datum in img_data:
                self.label.append(datum.label)
        print "**************Data Reading Done****************"

    """
    Load only labels from LMDB
    """
    def load_label(self, label_lmdb, compressed_label=True, max_count=-1):
        self.label = []
        self.label_lmdb = label_lmdb
        self.compressed_label = compressed_label

        print "***************Reading Label******************"
        print "\t Reading labels from {}".format(self.label_lmdb)
        print "\t Label Compressed = {}".format(compressed_label)

        data = self._load_from_lmdb(self.label_lmdb, max_count)
        for datum in data:
            # decode datum
            label=caffe.io.datum_to_array(datum)
            self.label.append(label.flatten())
        self.expanded_label_array = None

    # Load data from LMDB backend
    def _load_from_lmdb(self, lmdb_fn, max_count=-1):
        data = []
        with lmdb.open(lmdb_fn, readonly=True) as db:
            with db.begin() as txn:
                cursor = txn.cursor()
                for key, value in cursor:
                    datum = caffe_pb2.Datum()
                    datum.ParseFromString(value)
                    data.append(datum)
                    # This is for test, only read a limited number of samples
                    if max_count > 0 and len(data)==max_count:
                        break
        return data
