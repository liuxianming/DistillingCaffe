"""
This is used to generate network structure, using pynet_spec functions
"""

import prepare_net
import sys
sys.path.append('../../python')
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import os, os.path

def all_conv_net(data_lmdb, mean_value, mean_file, label_lmdb,
                 nout, lossfunction='CrossEntropy'):
    # prepare data and label layer
    data, label = prepare_net.data_layer(data_lmdb, mean_value, mean_file, label_lmdb)
    conv1, relu1 = prepare_net.conv_relu(data, 11, 96, stride=4)
    pool1 = prepare_net.max_pool(relu1, 3, stride=2)
    conv2, relu2 = prepare_net.conv_relu(pool1, 5, 256, pad=2, group=2)
    pool2 = prepare_net.max_pool(relu2, 3, stride=2)
    conv3, relu3 = prepare_net.conv_relu(pool2, 3, 384, pad=1)
    conv4, relu4 = prepare_net.conv_relu(relu3, 3, 384, pad=1, group=2)
    conv5, relu5 = prepare_net.conv_relu(relu4, 3, 384, pad=1, group=2)
    pool5 = prepare_net.max_pool(relu5, 3, stride=2)
    # Adding convolutional layers instead of fully connected layers
    conv6, relu6 = prepare_net.conv_relu(pool5, 3, 256, pad=1)
    conv7, relu7 = prepare_net.conv_relu(relu6, 3, 256, pad=1, group=2)
    conv8, relu8 = prepare_net.conv_relu(relu7, 3, 256, pad=1, group=2)
    # cccp layer
    cccp9, relu9 = prepare_net.cccp(relu8, nout)
    # global average pooling
    g_aver_pooling = prepare_net.aver_pool(relu9)
    # adding loss layer
    sigmoid_label = L.Sigmoid(label)
    if lossfunction == 'CrossEntropy':
        loss = L.SigmoidCrossEntropyLoss(g_aver_pooling, sigmoid_label)
    elif lossfunction == 'SoftmaxLoss':
        loss = L.SoftmaxWithLoss(g_aver_pooling, sigmoid_label)
    return to_proto(loss)

def gen_field_network():
    train_field_net_fn = 'train_field.prototxt'
    test_field_net_fn = 'test_field.prototxt'
    with open(train_field_net_fn, 'w') as f:
        print >>f, all_conv_net('/mnt/ilcompf2d1/data/be/prepared-2015-06-15/LMDB/TRAINING/image',
                                #None, './examples/distilling/behance.binaryproto',
                                [104, 117, 123], None,
                                '/mnt/ilcompf2d1/data/be/prepared-2015-06-15/LMDB/TRAINING/field',
                                nout=67)

    with open(test_field_net_fn , 'w') as f:
        print >>f, all_conv_net('/mnt/ilcompf2d1/data/be/prepared-2015-06-15/LMDB/TESTING/image',
                                #None, './examples/distilling/behance.binaryproto',
                                [104, 117, 123], None,
                                '/mnt/ilcompf2d1/data/be/prepared-2015-06-15/LMDB/TESTING/field', nout=67)

    # for this dataset contains about 55,000 testing iamges, test_iter = 250 (batchsize = 256)
    # test_interval = 2000, stepsize = 10,000 (about 2 epchos), max_iter = 100,000 (20 epchos)
    # snapshot = 5000 (1epcho)
    solver_fn = './conv_field_solver.prototxt'
    fn_prefix = 'examples/distilling'
    solver_param = prepare_net.get_solver(os.path.join(fn_prefix, train_field_net_fn),
                                          os.path.join(fn_prefix, test_field_net_fn),
                                          250, 2000, False,
                                          40,
                                          10000, 100000,
                                          5000, 'conv_field',
                                          solver_mode='GPU', filename=solver_fn)
    return solver_param

def gen_site_network():
    train_site_net_fn = 'train_site.prototxt'
    test_site_net_fn = 'test_site.prototxt'
    with open(train_site_net_fn, 'w') as f:
        print >>f, all_conv_net('/mnt/ilcompf2d1/data/be/prepared-2015-06-15/LMDB/TRAINING/image',
                                #None, './examples/distilling/behance.binaryproto',
                                [104, 117, 123], None,
                                '/mnt/ilcompf2d1/data/be/prepared-2015-06-15/LMDB/TRAINING/site',
                                nout=44)

    with open(test_site_net_fn , 'w') as f:
        print >>f, all_conv_net('/mnt/ilcompf2d1/data/be/prepared-2015-06-15/LMDB/TESTING/image',
                                #None, './examples/distilling/behance.binaryproto',
                                [104, 117, 123], None,
                                '/mnt/ilcompf2d1/data/be/prepared-2015-06-15/LMDB/TESTING/site',
                                nout=44)

    # for this dataset contains about 55,000 testing iamges, test_iter = 250 (batchsize = 256)
    # test_interval = 2000, stepsize = 10,000 (about 2 epchos), max_iter = 100,000 (20 epchos)
    # snapshot = 5000 (1epcho)
    solver_fn = './conv_site_solver.prototxt'
    fn_prefix = 'examples/distilling'
    solver_param = prepare_net.get_solver(os.path.join(fn_prefix, train_site_net_fn),
                                          os.path.join(fn_prefix, test_site_net_fn),
                                          250, 2000, False,
                                          40,
                                          10000, 100000,
                                          5000, 'conv_site',
                                          solver_mode='GPU', filename=solver_fn)
    return solver_param

if __name__ == '__main__':
    gen_field_network()
    gen_site_network()
