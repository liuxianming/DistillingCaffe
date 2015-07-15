#Test performance of trained model
import numpy as np
import lmdb
import sys
import os, os.path
from contextlib import nested

CAFFE_ROOT = '/mnt/ilcompf0d0/user/xliu/code/caffe/python'
sys.path.append(CAFFE_ROOT)
import caffe
from caffe.io import caffe_pb2
import matplotlib.pyplot as plt
import sklearn
import sklearn.metric

def get_net(network_fn, model_fn, mode='CPU'):
    if mode == 'CPU':
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
    try:
        net = caffe.Net(os.path.join(CAFFE_ROOT, network_fn),
                        os.path.join(CAFFE_ROOT, model_fn),
                        caffe.TEST)
        return net
    except ():
        return None

def classify_sample(net, sample):
    # classify a data sample using current net
    out = net.forward()
    return out[-1].flatten() # return the last layer output, as a 1-dimensional np.array

"""
Load data from either datum or from original image
Here, I used caffe.io.read_datum_from_image to resolve all format transformation,
instead of the io functions provided by caffe python wrapper
"""
def extract_sample_from_datum(datum, image_mean):
    # extract numpy array from datum, then substract the mean_image
    if datum.encoded == True:
        datum = caffe.io.decode_datum(datum)
    img_data = caffe.io.datum_to_array(datum)
    img_data = substract_mean(img_data, image_mean)
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
    return image_mean

class Data_CaffeNet_Classifier:
    """
    classifiy all data samples using caffe netowrk
    data samples are stored in self.data, while labels are stored in self.label
    """
    def __init__(self):
        self.mean = None
        self.net = None
        self.data = None
        self.label = None
        self.classifications = None
        self.expanded_label_array = None

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
        else:
            self.mean = image_mean
        print "[Debug Info]: Image Mean Shape = {}".format(self.mean.shape)

    def set_network(self, network_fn, model_fn, mode='GPU'):
        self.network_fn = network_fn
        self.model_fn = model_fn
        self.net = get_net(network_fn, model_fn, mode)

    def classify_dataset(self, output_fn=None):
        self.classifications = []
        for img_data in self.data:
            self.classifications.append(classify_sample(self.net, img_data))
        # turn into numpy array
        self.classifications = np.array(self.classifications)
        if output_fn is not None:
            # save the classification results to outside file
            np.save(output_fn, self.classifications)
            print "Classification Results are Saved to {}".format(output_fn)

    def expand_label_array(self, output_fn=None):
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

    """
    Functions to calculate performances, including P-R curve,
    """

    """
    # get p-r curve, AUC and plot p-r curve
    # figure_fn is not None: draw figure and output to file. name is the title
    def get_pr(self, interval_count=20):
        thresholds = np.linspace(0, 1.0, interval_count)
        self.p = []
        self.r = []
        for t in thresholds:
            # calculate precision and recall given threshold
            binaryizedClassification = np.zeors(self.classifications.shape)
            binaryizedClassification[self.classifications > t] = 1.0
            # precision: correctly_classified / all classified
            # recall: correctly_classifice / all_groundtruth
            all_classified = np.sum(binaryizedClassification)
            all_groundtruth = np.sum(self.expanded_label_array)
            correct_classification = binaryizedClassification * self.expanded_label_array
            all_correct = np.sum(correct_classification)
            self.p.append(all_correct / all_classified)
            self.r.append(all_correct / all_groundtruth)
        # append (1, 0) to make sure the curve starts from point (0,1) when plotting
        self.p.append(1.0)
        self.r.append(0.0)
        self.p = np.array(p)
        self.r = np.array(r)
        return self.p, self.r, Data_CaffeNet_Classifier.get_auc(self, self.r,self.p)
    """
    def get_prs(self):
        self.p = dict()
        self.r = dict()
        self.average_precision = dict()
        self.auc = dict()

        n_class = self.classifications.shape[1]
        for i in range(n_class):
            y_pred = self.classifications[:, i]
            y = self.expanded_label_array[:,i]
            self.p[i], self.r[i], _ = sklearn.metric.precision_recall_curve(y, y_pred)
            self.average_precision[i] = sklearn.metric.average_precision_score(y, y_pred)
            self.auc[i] = sklearn.metric.auc(self.r[i], self.p[i], reorder=True)
        # compute the overall measurements
        self.p['micro'], self.r['micro'], _ = \
                                              sklearn.metric.precision_recall_curve(
                                                  self.expanded_label_array.ravel(),
                                                  self.classifications.ravel())
        self.average_precision['overall'] = sklearn.metric.average_precision_score(
            self.expanded_label_array, self.classifications, average='micro')
        self.auc['micro'] = sklearn.metric.auc(self.r['micro'], self.p['micro'], reorder=True)


    def plot_curve(x, y, label, title, figure_fn=None):
        plt.clf()
        plt.plot(x, y, label=label)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(title)
        plt.legend(loc='lower left')
        if figure_fn is None:
            plt.show()
        else:
            plt.savefig(figure_fn)


class LMDB_CaffeNet_Classifier(Data_CaffeNet_Classifier):
    """
    Load data from LMDB, and classify all sampels using given network model
    Output various kinds of statistics, including class wise accuracy, precision, recall, etc
    """
    def __init__(self):
        self.data_lmdb = None
        self.label_lmdb = None
        self.mean = None

    """
    Load data from lmdb and transform to numpy array, stored in self.data
    """
    def set_data(self, data_lmdb, label_lmdb=None):
        self.data_lmdb = data_lmdb
        self.label_lmdb = label_lmdb

    """
    load data from LMDB backend
    compressed_label is a flag to indicate whether or not the label are compact or expanded
    e.g., [1, 4, 5] vs [0 1 0 0 1 1...]
    """
    def load_data(self, data_lmdb, image_mean, label_lmdb=None, compressed_label=True):
        LMDB_CaffeNet_Classifier.set_data(self, data_lmdb, label_lmdb)
        Data_CaffeNet_Classifier.set_mean(self, image_mean)
        self.data = []
        self.label = []
        print "Reading all data into memory..."
        print "****************INFO*********************"
        if self.label_lmdb is not None:
            # read data and label from two lmdbs
            print "\t Reading data from {}".format(self.data_lmdb)
            print "\t Reading label from {}".format(self.label_lmdb)
            print "\t Substracting image mean from {}".format(image_mean)

            with nested(lmdb.open(self.data_lmdb, readonly=True),
                        lmdb.open(self.label_lmdb, readonly=True)) as \
                (data_lmdb, label_lmdb):
                with nested(data_lmdb.begin(), label_lmdb.begin()) \
                     as (data_txn, label_txn):
                    data_cursor = data_txn.cursor()
                    label_cursor = label_txn.cursor()
                    for key, value in data_cursor:
                        datum = caffe_pb2.Datum()
                        label_datum = caffe_pb2.Datum()
                        datum.ParseFromString(value)
                        img_data = extract_sample_from_datum(datum, self.mean)
                        self.data.append(img_data)
                        # get label
                        label_value = label_cursor.get(key)
                        label_datum.ParseFromString(label_value)
                        label = caffe.io.datum_to_array(label_datum)
                        self.label.append(label)
        else:
            print "\t Reading data and label from {}".format(self.data_lmdb)
            print "\t Substracting image mean from {}".format(image_mean)

            # in this case, label are not expanded
            compressed_label = True
            # label and data are in the same datum
            with lmdb.open(self.data_lmdb) as data_lmdb:
                with data_lmdb.begin() as data_txn:
                    data_cursor = data_txn.cursor()
                    for key, value in data_cursor:
                        datum = caffe_pb2.Datum()
                        datum.ParseFromString(value)
                        img_data = extract_sample_from_datum(datum, self.mean)
                        self.data.append(datum)
                        # get label
                        self.label.append(datum.label)
        print "**************Data Reading Done****************"

        if not compressed_label:
            print "Label loaded are already expanded. Transform to numpy array format"
            self.expanded_label_array = np.array(self.label)
