import sys
import os, os.path
CAFFE_ROOT_DIR = '/mnt/ilcompf0d0/user/xliu/code/caffe/'
sys.path.append(os.path.join(CAFFE_ROOT_DIR, 'python'))
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.io import caffe_pb2
import py_data_layers
import yaml

DISTILLING_ROOT_DIR = os.path.join(CAFFE_ROOT_DIR, 'examples/distilling')
BASE_NET_DIR = os.path.join(DISTILLING_ROOT_DIR, 'base_alexnet')
base_model_fn = os.path.join(CAFFE_ROOT_DIR, 'models/bvlc_alexnet/bvlc_alexnet.caffemodel')

def load_base_net(base_net_fn):
    network_param = caffe.io.read_net_param(base_net_fn)
    return network_param

def load_base_solver(base_solver_fn):
    solver_param = caffe.io.read_solver_param(base_solver_fn)
    return solver_param

def change_solver(solver_param, arg_str):
    args = yaml.load(arg_str)
    if 'net' in args.keys():
        solver_param.net = args['net']
    if 'test_iter' in args.keys():
        solver_param.test_iter = int(args['test_iter'])
    if 'test_interval' in args.keys():
        solver_param.test_interval = int(args['test_interval'])
    if 'base_lr' in args.keys():
        solver_param.base_lr = float(args['base_lr'])
    if 'display' in args.keys():
        solver_param.display = int(args['display'])
    if 'average_loss' in args.keys():
        solver_param.average_loss = int(args['average_loss'])
    if 'max_iter' in args.keys():
        solver_param.max_iter = int(args['max_iter'])
    if 'lr_policy' in args.keys():
        solver_param.lr_policy = args['lr_policy']
    if 'gamma' in args.keys():
        solver_param.gamma = float(args['gamma'])
    if 'power' in args.keys():
        solver_param.power = float(args['power'])
    if 'momentum' in args.keys():
        solver_param.momentum = float(args['momentum'])
    if 'weight_decay' in args.keys():
        solver_param.weight_decay = float(args['weight_decay'])
    if 'stepsize' in args.keys():
        solver_param.stepsize = int(args['stepsize'])
    if 'snapshot' in args.keys():
        solver_param.snapshot = int(args['snapshot'])
    if 'snapshot_prefix' in args.keys():
        solver_param.snapshot_prefix = args['snapshot_prefix']
    return solver_param

def build_layer_name_map(network_param):
    layer_name_map = dict()
    for layer_param in network_param.layer:
        layer_name_map[layer_param.name] = layer_param
    return layer_name_map

def run_task(task_dir, task_solver_fn, base_model_fn, gpu_id):
    os.chdir(task_dir)
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    solver = caffe.SGDSolver(task_solver_fn)
    solver.net.copy_from(base_model_fn)
    solver.solve()

# run_task, task_type = 'retrain' or 'finetune'
def run_net(gpu_id, retrain_layer_name_list, task_name, task_type='retrain'):
    task_root_dir = os.path.join(DISTILLING_ROOT_DIR, task_name)
    if not os.path.exists(task_root_dir):
        os.mkdir(task_root_dir)
    base_net_param = load_base_net(os.path.join(BASE_NET_DIR, 'train_val_field.prototxt'))
    base_deploy_net_param = load_base_net(os.path.join(BASE_NET_DIR, 'deploy.prototxt'))
    base_solver = load_base_solver(os.path.join(BASE_NET_DIR, 'field_solver.prototxt'))

    task_network_fn = os.path.join(task_root_dir, 'train_val_field.prototxt')
    task_deploy_network_fn = os.path.join(task_root_dir, 'deploy.prototxt')
    task_solver_fn = os.path.join(task_root_dir, 'field_solver.prototxt')
    with open(task_network_fn, 'w') as f:
        print >> f, modify_net(base_net_param, retrain_layer_name_list, task_name, task_type)
    with open(task_deploy_network_fn, 'w') as f:
        print >> f, modify_net(base_deploy_net_param, retrain_layer_name_list, task_name, task_type)
    with open(task_solver_fn, 'w') as f:
        print >> f, base_solver
    run_task(task_root_dir, task_solver_fn, base_model_fn, gpu_id)

def modify_net(base_net_param, retrain_layer_name_list, task_name, task_type='retrain'):
    target_net_param = base_net_param
    target_layer_name_map = build_layer_name_map(target_net_param)
    if task_type == 'retrain':
        for layer_name in retrain_layer_name_list:
            target_layer_name_map[layer_name].name = layer_name + '-retrain'
    elif task_type == 'finetune':
        # set other layers lr_mult = 0
        for layer in target_net_param.layer:
            if layer.name not in retrain_layer_name_list:
                for param in layer.param:
                    param.lr_mult = 0
        for layer_name in retrain_layer_name_list:
            target_layer_name_map[layer_name].name = layer_name + '-finetune'
    return target_net_param

if __name__=="__main__":
    task_id = int(sys.argv[1])
    if task_id == 0:
        task_name = "Retrain_Conv5"
        print "Run task [{}]".format(task_name)
        task_type = 'retrain'
        train_layer_name_list = ['fc8', 'fc7', 'fc6']
        run_net(task_id, train_layer_name_list, task_name, task_type=task_type)
    if task_id == 1:
        task_name = "Retrain_Conv3"
        print "Run task [{}]".format(task_name)
        task_type = 'retrain'
        train_layer_name_list = ['fc8', 'fc7', 'fc6', 'conv5', 'conv4']
        run_net(task_id, train_layer_name_list, task_name, task_type=task_type)
    if task_id == 2:
        task_name = "finetune_fc8"
        print "Run task [{}]".format(task_name)
        task_type = 'finetune'
        train_layer_name_list = ['fc8']
        run_net(task_id, train_layer_name_list, task_name, task_type=task_type)
    if task_id == 3:
        task_name = "finetune_Conv5"
        print "Run task [{}]".format(task_name)
        task_type = 'finetune'
        train_layer_name_list = ['fc8', 'fc7', 'fc6']
        run_net(task_id, train_layer_name_list, task_name, task_type=task_type)
