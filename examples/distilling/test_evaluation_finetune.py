import performance_test
#from performance_test import *
import pickle
import os, os.path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

distilling_base_dir =  '/mnt/ilcompf0d0/user/xliu/code/caffe/examples/distilling'
test_image_lmdb = '/mnt/ilcompf2d1/data/be/prepared-2015-06-15/Encoded_LMDB/TESTING/image'
test_field_lmdb = '/mnt/ilcompf2d1/data/be/prepared-2015-06-15/Encoded_LMDB/TESTING/field'
image_mean = './behance.binaryproto'

def evaluate_task(task_name, model, network='deploy.prototxt',
                  visualize_filter=False, visualize_featuremap=False):
    task_dir = os.path.join(distilling_base_dir, task_name)
    print "Evaluating task {}".format(task_name)
    network_fn=os.path.join(task_dir, network)
    model_fn = os.path.join(task_dir, model)

    tester = performance_test.Data_CaffeNet_Classifier(resize=227)
    print "=================Testing Field Performance=================="
    tester.set_network(network_fn, model_fn, mode='GPU', input_size=(3, 227, 227))

    if visualize_filter:
        tester.visualize_filter(figure_fn=os.path.join(task_dir, 'conv1_filter.png'), layer='conv1')
        tester.visualize_filter(figure_fn=os.path.join(task_dir, 'conv2_filter.png'), layer='conv2')
        tester.visualize_filter(figure_fn=os.path.join(task_dir, 'conv3_filter.png'), layer='conv3')
        tester.visualize_filter(figure_fn=os.path.join(task_dir, 'conv4_filter.png'), layer='conv4')
        tester.visualize_filter(figure_fn=os.path.join(task_dir, 'conv5_filter.png'), layer='conv5')

    tester.load_data(test_image_lmdb, image_mean, label_lmdb=test_field_lmdb, compressed_label=True)
    tester.classify_dataset(output_fn = os.path.join(task_dir, 'classification.p'))

    if visualize_featuremap:
        print "=================Visualizing feature maps of first 10 samples, layer conv_1================="
        for i in range(10):
            tester.visualize_featuremap('conv1', sample_id=i)

    tester.get_prs(expanded_label_fn = os.path.join(task_dir, 'expanded_label') )
    tester.plot_curve(tester.r['micro'], tester.p['micro'],
                      label='Micro, AUC={}'.format(tester.auc['micro']),
                      title='PR Curve of Field Classification',
                      figure_fn=os.path.join(task_dir, 'field_pr.png') )
    # save to files
    with open(os.path.join(task_dir, 'precision.p'), 'wb') as f_p_fp:
        pickle.dump(tester.p, f_p_fp)
    with open(os.path.join(task_dir, 'recall.p'), 'wb') as f_r_fp:
        pickle.dump(tester.r, f_r_fp)
    with open(os.path.join(task_dir, 'auc.p'), 'wb') as f_a_fp:
        pickle.dump(tester.auc, f_a_fp)

"""Plot bars given dicts as input:
dicts is a dictionary of dictionary, dicts.key is the name of task, and dict.value is a dictionary to plot
"""
def plot_bars(dicts, ax):
    colors = "bgrcmykw"
    #fig, ax = plt.subplots()
    #fig.set_size_inches(20, 16)
    rects = []
    # prepare axes
    performance = dicts.values()[0]
    N = len(performance)
    task_num = len(dicts)
    ind = np.arange(N) * task_num
    width = 0.75
    task_idx = 0
    for task_name in dicts:
        performance = dicts[task_name]
        c = colors[task_idx % len(colors)]
        rect = ax.bar(ind + task_idx * width, performance.values(), width, color=c)
        task_idx += 1
        rects.append(rect)
    ax.set_ylabel('Scores')
    ax.set_xticks(ind+width * task_num)
    ax.set_xticklabels(performance.keys())
    ax.legend( rects, dicts.keys() )
    # save to file
    #plt.show()
    #plt.savefig(figure_fn)

def main():
    tasks = ['Retrain_Conv5', 'Retrain_Conv3', 'finetune_fc8']
    """
    model = 'alexnet_field_iter_30000.caffemodel'
    for task in tasks:
        evaluate_task(task, model=model)
    """
    plot_class_specfic_auc(tasks)

def plot_class_specfic_auc(tasks):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,16), sharex=True)
    tasks_dicts = {}
    for task in tasks:
        # load data
        auc_fn = os.path.join(task, 'auc.p')
        with open(auc_fn,'r') as f:
            performance = pickle.load(f)
            tasks_dicts[task] = performance
    plot_bars(tasks_dicts, ax1)

    # now plot training sample distribution
    label_dist = np.load('./training_label_distribution.npy')
    ind = np.arange(label_dist.shape[0]) * len(tasks)
    ax2.bar(ind, label_dist, 0.75 * len(tasks))
    ax2.set_ylabel('Number of Training Samples')
    plt.show()
    plt.savefig('class_specific_performance.png')


if __name__ == "__main__":
    main()
    # to plot the class specific performance,
    # should run get_training_stat.py first to generate the training label distribution
