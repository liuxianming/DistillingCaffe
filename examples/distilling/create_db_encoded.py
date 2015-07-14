"""create_db_encoded.py
Create lmdb using encoded images
"""
import sys
import os, os.path
CAFFE_ROOT_DIR = '/mnt/ilcompf0d0/user/xliu/code/caffe'
sys.path.append(os.path.join(CAFFE_ROOT_DIR, 'python'))
import caffe
from caffe.io import caffe_pb2
import lmdb
import numpy as np
import pickle
from random import shuffle
from contextlib import nested

"""
dump encoded images into LMDB, to save disk usage
The size should be similar to bcf file
"""
def create_db_encoded(db_root_dir, image_list, field_list, sites_list, shuffle_flag=True):
    image_db = lmdb.open(os.path.join(db_root_dir, 'image'), map_size=int(1e12))
    field_db = lmdb.open(os.path.join(db_root_dir, 'field'), map_size=int(1e12))
    site_db = lmdb.open(os.path.join(db_root_dir, 'site'), map_size=int(1e12))

    if shuffle_flag:
        print "shuffling all images..."
        shuffle(image_list)
        print "shuffling done"
    else:
        print "No shuffling..."

    # save shuffled image list to file
    pickle.dump(image_list, open(os.path.join(db_root_dir, 'image_list.p'), 'w'))

    print "Begin generating LMDBs..."

    with nested(image_db.begin(write=True),
                field_db.begin(write=True),
                site_db.begin(write=True)) \
        as (image_txn, field_txn, site_txn):
        # 1. load image data
        # 2. load class information, including field and site
        # Use field_list and sites_list to find corresponding index of each label
        img_count = 0
        for img_info in image_list:
            img_id = img_info['image_id']
            img_id = int(img_id.split('_')[-1])
            # print img_id
            img_fn = img_info['image_fn']
            fields = img_info['field']
            sites = img_info['site_id']
            proj_id = int(img_info['project_id'])
            # begin processing
            try:
                im_dat = caffe.io.read_datum_from_image(img_fn, label=proj_id)
                f_array = [field_list.index(f) for f in fields if (f in field_list)]
                s_array = [sites_list.index(s) for s in sites if (s in sites_list)]
                f_array = np.array(f_array)
                f_array = np.reshape(f_array, (1,1, len(f_array)))
                s_array = np.array(s_array)
                s_array = np.reshape(s_array, (1,1, len(s_array)))
                field_dat = caffe.io.array_to_datum(f_array)
                sites_dat = caffe.io.array_to_datum(s_array)
            except (IndexError, ValueError):
                print "Error with image information [%d]" % img_id
                continue
            # put everything into LMDB
            image_txn.put('{:0>24d}'.format(img_id), im_dat.SerializeToString())
            field_txn.put('{:0>12d}'.format(img_id), field_dat.SerializeToString())
            site_txn.put('{:0>12d}'.format(img_id), sites_dat.SerializeToString())
            img_count += 1
            if img_count % 10000 == 0:
                print img_count

    image_db.close()
    field_db.close()
    site_db.close()


def main(task='train'):
    ROOT = '/mnt/ilcompf2d1/data/be/prepared-2015-06-15'
    site_list_fn = os.path.join(ROOT, './crawler_code/lists/used_site_list')
    field_list_fn = os.path.join(ROOT, './crawler_code/lists/used_field_list')
    with nested(open(site_list_fn, 'r'),
                open(field_list_fn, 'r')) \
         as (site_list_fp, field_list_fp):
        lines = site_list_fp.readlines()
        sites_list = [int(k) for k in lines]
        lines = field_list_fp.readlines()
        field_list = [int(k) for k in lines]

    root_dir = os.path.join(ROOT, './Encoded_LMDB')
    #root_dir = './LMDB'
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    if task == 'train':
        # generate training data
        image_list_fn = os.path.join(ROOT, './crawler_code/lists/training_images.p')
        db_root_dir = os.path.join(root_dir, 'TRAINING')
        print "Creating Training LMDB into %s" % db_root_dir
    elif task == 'test':
        # generate testing data / validation data
        image_list_fn = os.path.join(ROOT, './crawler_code/lists/testing_images.p')
        db_root_dir = os.path.join(root_dir, 'TESTING')
        print "Creating Testing LMDB into %s" % db_root_dir
    else:
        print "Wrong type of task: %s" % task
        sys.exit()
    if not os.path.exists(db_root_dir):
        os.mkdir(db_root_dir)
    with open(image_list_fn, 'rb') as image_list_fp:
        image_list = pickle.load(image_list_fp)
    create_db_encoded(db_root_dir, image_list, field_list, sites_list)
    print "Job Completed!"

if __name__ == "__main__":
    for task in sys.argv[1:]:
        main(task.lower())
