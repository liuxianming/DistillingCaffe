import performance_test
#from performance_test import *
import pickle

# Test field
tester = performance_test.LMDB_CaffeNet_Classifier()
tester = performance_test.LMDB_CaffeNet_Classifier()

test_image_lmdb = '/mnt/ilcompf2d1/data/be/prepared-2015-06-15/LMDB/TESTING/image'
test_field_lmdb = '/mnt/ilcompf2d1/data/be/prepared-2015-06-15/LMDB/TESTING/field'
test_site_lmdb = '/mnt/ilcompf2d1/data/be/prepared-2015-06-15/LMDB/TESTING/site'
image_mean = './behance.binaryproto'

field_network_fn='/mnt/ilcompf0d0/user/xliu/code/caffe/examples/distilling/deploy_field_alexnet.prototxt'
field_model_fn = '/mnt/ilcompf0d0/user/xliu/code/caffe/examples/distilling/conv_field_iter_30000.caffemodel'

site_network_fn='/mnt/ilcompf0d0/user/xliu/code/caffe/examples/distilling/deploy_site_alexnet.prototxt'
site_model_fn = '/mnt/ilcompf0d0/user/xliu/code/caffe/examples/distilling/conv_site_iter_30000.caffemodel'

print "Testing Field Performance==================================="
tester.set_network(field_network_fn, field_model_fn, mode='GPU')
tester.visualize_filter(figure_fn='field_conv1_filter.png', layer='Convolution1')
tester.load_data(test_image_lmdb, image_mean, label_lmdb=test_field_lmdb, compressed_label=False)
tester.classify_dataset()

# calculate performance
tester.get_prs()
tester.plot_curve(tester.r['micro'], tester.p['micro'],
                  label='Micro, AUC={}'.format(tester.auc['micro']),
                  title='PR Curve of Field Classification',
                  figure_fn='field_pr.png')
# save to files
with open('field_precision.p', 'wb') as f_p_fp:
    pickle.dump(tester.p, f_p_fp)
with open('field_recall.p', 'wb') as f_r_fp:
    pickle.dump(tester.r, f_r_fp)
with open('field_auc.p', 'wb') as f_a_fp:
    pickle.dump(tester.auc, f_a_fp)

# Test site classification
tester.set_network(site_network_fn, site_model_fn, mode='GPU')
tester.visualize_filter(figure_fn='site_conv1_filter.png', layer='Convolution1')
tester.load_label(test_site_lmdb, compressed_label=False)
tester.classify_dataset()

# calculate performance
tester.get_prs()
tester.plot_curve(tester.r['micro'], tester.p['micro'],
                  label='Micro, AUC={}'.format(tester.auc['micro']),
                  title='PR Curve of Site Classification',
                  figure_fn='site_pr.png')
# save to files
with open('site_precision.p', 'wb') as s_p_fp:
    pickle.dump(tester.p, s_p_fp)
with open('site_recall.p', 'wb') as s_r_fp:
    pickle.dump(tester.r, s_r_fp)
with open('site_auc.p', 'wb') as s_a_fp:
    pickle.dump(tester.auc, s_a_fp)
