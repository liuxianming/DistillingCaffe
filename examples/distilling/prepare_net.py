import sys

sys.path.append('../../python')
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import google.protobuf

# Generate data layer and label
def data_layer(data_lmdb, mean_value=None, mean_file=None, label_lmdb=None, batch_size=256, crop_size=0, mirror=False):
    transform_param = {}
    if not mean_value == None:
        transform_param['mean_value'] = mean_value
    elif not mean_file == None:
        transform_param['mean_file'] = mean_file
    if crop_size > 0:
        transform_param['crop_size']=crop_size
    if mirror == True:
        transform_param['mirror'] = mirror

    if label_lmdb == None:
        data, label = L.Data(source=data_lmdb, backend=P.Data.LMDB, batch_size=batch_size,
                             ntop=2, transform_param=transform_param)
    else:
        data = L.Data(source=data_lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntop=1,
                      transform_param=transform_param,
                      name='data')
        label = L.Data(source=label_lmdb, backend=P.Data.LMDB, batch_size=batch_size,
                       ntop=1, name='label')
    return data, label

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         weight_filler=dict(type='xavier'))
    return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout):
    fc = L.InnerProduct(bottom, num_output=nout)
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

# define average pooling layer: if ks=0, it is global average_pooling
def aver_pool(bottom, ks=0, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.AVE, stride=stride, global_pooling=True)

def cccp(bottom, nout):
    cccp_layer = L.Convolution(bottom, kernel_size=1, num_output=nout,
                               weight_filler=dict(type='xavier'))
    return cccp_layer, L.ReLU(cccp_layer, in_place=True)

def get_solver(train_net, test_net,
               test_iter, test_interval, test_initialization,
               display,
               stepsize, max_iter,
               snapshot, snapshot_prefix,
               solver_mode='GPU',
               filename=''):
    solver_param = caffe_pb2.SolverParameter()
    solver_param.test_iter.append(test_iter)
    solver_param.test_interval = test_interval
    solver_param.test_initialization = test_initialization
    solver_param.display = display
    solver_param.average_loss = display
    solver_param.base_lr = 0.001
    solver_param.lr_policy = 'step'
    solver_param.stepsize = stepsize
    solver_param.gamma = 0.96
    solver_param.max_iter = max_iter
    solver_param.momentum = 0.9
    solver_param.weight_decay = 0.0002
    solver_param.snapshot = snapshot
    solver_param.snapshot_prefix = snapshot_prefix
    if solver_mode == 'GPU':
        solver_param.solver_mode = 1
    else:
        solver_param.Solver_mode = 0
    if test_net == None:
        solver_param.net = train_net
    else:
        solver_param.train_net = train_net
        solver_param.test_net.append(test_net)

    if not filename=='':
        # save to file
        with open(filename, 'w') as f:
            f.write(google.protobuf.text_format.MessageToString(solver_param))
        f.close()
    return solver_param
