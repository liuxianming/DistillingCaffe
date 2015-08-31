"""project_verification.py
Using deploy network definition to do project verification task
The basic work flow is to extract certain amount of samples from the testing set
and then get the output (feat_norm)
Then calculate a distance matrix N * N
Using PR-Curve to evaluate the performance of each network
"""

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except:
    print 'cannot import matplotlib'

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
    print "loading data from {}, size={}".format(source, img_count)
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
        #print img_data.shape
        #with open('./images/img-{:05d}.jpg'.format(idx), 'wb') as f:
        #    f.write(datum.data)
        #f.close()
        if mean_fn is not None:
            img_data = substract_mean(img_data, image_mean)
        if resize > 0:
            img_data = resize_img_data(img_data, resize)
        data.append(img_data)
        if idx % 1000 == 0:
            print "{} images loaded...".format(idx)
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


def evaluate_pr(sim_mat, label_mat):
        precision, recall, _ = precision_recall_curve(label_mat.ravel(), sim_mat.ravel())
        avg_p = average_precision_score(label_mat.ravel(), sim_mat.ravel())
        return precision, recall, avg_p

"""
Draw pr curves on a single image
"""
def draw_pr_curves(count, pr, avg_p_scores, legends, figure_fn):
    plt.clf()
    for idx in range(count):
        label_str = "%s: AP=%.3f" % (legends[idx], avg_p_scores[idx])
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

def main(device_id):
    # main function
    network_dir = './networks'
    model_dir = './models'
    network_fn = [os.path.join(network_dir, 'deploy_conv.prototxt'),
                  os.path.join(network_dir, 'deploy_conv.prototxt'),
                  os.path.join(network_dir, 'deploy_conv.prototxt'),
                  #os.path.join(network_dir, 'deploy_alexnet.prototxt'),
                  os.path.join(network_dir, 'deploy_alexnet.prototxt'),
                  os.path.join(network_dir, 'deploy_alexnet_conv5.prototxt'),
                  #os.path.join(network_dir, 'deploy_alexnet_conv5.prototxt'),
                  #os.path.join(network_dir, 'deploy_alexnet_conv3.prototxt')
    ]
    model_fn = [os.path.join(model_dir, 'siamese_project_conv.caffemodel'),
                os.path.join(model_dir, 'triplet_project.caffemodel'),
                #os.path.join(model_dir, 'triplet_project_3.caffemodel'),
                os.path.join(model_dir, 'alexnet_field_retrain_conv5.caffemodel'),
                os.path.join(model_dir, 'bvlc_alexnet.caffemodel'),
                os.path.join(model_dir, 'alexnet_field_retrain_conv5.caffemodel'),
                #os.path.join(model_dir, 'bvlc_alexnet.caffemodel'),
                #os.path.join(model_dir, 'bvlc_alexnet.caffemodel')
    ]
    task_names = ['Siamese_Conv_Net',
                  'Triplet_Conv_Net',
                  #'Triplet_Conv_Net_lr0.02',
                  'field_fc8',
                  'AlexNet_fc8',
                  'field_conv5',
                  #'AlexNet_conv5',
                  #'AlexNet_conv3'
    ]

    img_size = 227
    count = 1000
    source = '/mnt/ilcompf2d1/data/be/prepared-2015-06-15/LMDB/TESTING/image'
    mean_fn = './behance.binaryproto'
    data, label_mat = read_data(source, count, mean_fn, resize=img_size)
    print data.shape
    print label_mat.shape

    np.save('./evaluation/project_verification_label_matrix', label_mat)
    pr = []
    average_p_score = []
    for idx in range(len(network_fn)):
        print "------------------Processing {}-----------------".format(task_names[idx])
        verifier = project_verifier(network_fn[idx], model_fn[idx], data)
        print "------------------Processign data------------------"
        verifier.process_data()
        print "------------------Evaluation ---------------------"
        rslt = verifier.evaluate(label_mat)
        # save similarity matrix
        sim_matrix = verifier.get_sim_matrix()
        print sim_matrix.shape
        np.save('./evaluation/'+task_names[idx]+'_sim_matrix', sim_matrix)
        pr_dict_ele = {}
        pr_dict_ele['p'] = rslt[0]
        pr_dict_ele['r'] = rslt[1]
        pr.append(pr_dict_ele)
        average_p_score.append(rslt[2])
        draw_distribution(sim_matrix.ravel(), label_mat.ravel(),
                          figure_fn='./evaluation/'+task_names[idx]+'_distribution.png')

    figure_fn = './evaluation/project_verification.png'
    draw_pr_curves(len(task_names), pr, average_p_score, task_names, figure_fn)
    #print pr
    print average_p_score

def maineval():
    BASE_DIR = './evaluation/1000'
    LABEL_MAT = os.path.join(BASE_DIR, 'project_verification_label_matrix.npy')
    task_names = ['Siamese_Conv_Net',
                  'Triplet_Conv_Net',
                  #'Triplet_Conv_Net_lr0.02',
                  'field_fc8',
                  'AlexNet_fc8',
                  'field_conv5',
                  'AlexNet_conv5'
                  # 'AlexNet_conv3'
    ]
    SIM_MATS = [os.path.join(BASE_DIR, "{}_sim_matrix.npy".format(task)) for task in task_names]

    label_mat = np.load(LABEL_MAT)
    sim_mats = [np.load(s) for s in SIM_MATS]
    print label_mat.shape

    # evaluate each individual model
    for i,f in enumerate(SIM_MATS):
        _, _, avg_p = evaluate_pr(sim_mats[i], label_mat)
        print i, f+':', avg_p

    # evaluate combinations of models: order = 2
    for i in range(len(SIM_MATS) - 1):
        for j in range(i, len(SIM_MATS)):
            combination_name = "{}+{}".format(task_names[i], task_names[j])
            sim_matrix = sim_mats[i] + sim_mats[j]
            avg_p = average_precision_score(
                label_mat.ravel(), sim_matrix.ravel())
            print 'ensemble {}'.format(combination_name), avg_p
            draw_distribution(sim_matrix.ravel(), label_mat.ravel(),
                          figure_fn='./evaluation/'+combination_name+'.png')
    # evaluate the combination of all models
    combination_name = "Siamese_Conv+Triplet_Conv+Field_Conv5+AlexNet_Conv5"
    sim_matrix = sim_mats[0]+sim_mats[1]+sim_mats[4]+sim_mats[5]
    precision, recall, avg_p = evaluate_pr(sim_matrix.ravel(), label_mat)
    print 'ensemble {}'.format(combination_name), avg_p
    # save plot
    draw_distribution(sim_matrix.ravel(), label_mat.ravel(),
                          figure_fn='./evaluation/combination_distribution.png')
    """
    for i in range(6):
        for j in range(6):
            if i>=j-100:
                continue
            avg_p = average_precision_score(label_mat.ravel(), (sim_mats[i]+sim_mats[j]).ravel())
            print i, j, 'ensemble 2', avg_p
    """

def mainplot():
    LABEL_MAT = './result/project_verification_label_matrix.npy'
    SIM_MATS = ['./result/Train_FCN_sim_matrix.npy', './result/FCN_CONV1_INIT_sim_matrix.npy', './result/Train_Field_sim_matrix.npy', \
                './result/Alexnet_Conv4_sim_matrix.npy', './result/Alexnet_Conv5_sim_matrix.npy', './result/Alexnet_sim_matrix.npy']

    label_mat = np.load(LABEL_MAT)
    print label_mat.shape
    sim_mats = []
    for f in SIM_MATS:
        m = np.load(f)
        sim_mats += [m]

    pr = []
    average_p_score = []
    MDL_IDS = [[5], [2], [0], [0, 2], [0, 2, 4]]
    LEGENDS = ['AlexFC', 'FC', 'FCN', 'FCN+FC', 'FCN+FC+AlexFC']
    for i in range(len(MDL_IDS)):
        sim_mat = sim_mats[0]*0
        for j in MDL_IDS[i]:
            sim_mat = sim_mat+sim_mats[j]
        sim_mat = sim_mat*1.0/len(MDL_IDS[i])

        precision, recall, _ = precision_recall_curve(label_mat.ravel(), sim_mat.ravel())
        avg_p = average_precision_score(label_mat.ravel(), sim_mat.ravel())
        rslt = [precision, recall, avg_p]

        pr_dict_ele = {}
        pr_dict_ele['p'] = rslt[0]
        pr_dict_ele['r'] = rslt[1]
        pr.append(pr_dict_ele)
        average_p_score.append(rslt[2])
        #draw_distribution(sim_mat.ravel(), label_mat.ravel(), figure_fn='./result/{}_distribution.png'.format(LEGENDS[i]))

    figure_fn = './result/project_verification.png'
    draw_pr_curves(len(LEGENDS), pr, average_p_score, LEGENDS, figure_fn)

if __name__ == '__main__':
    #sys.exit(main(2))
    sys.exit(maineval())
    #sys.exit(mainplot())

