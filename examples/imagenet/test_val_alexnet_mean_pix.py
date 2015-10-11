#! /usr/bin/env python

import sys
import numpy as np
import cPickle as pickle
import caffe
from caffe.io import caffe_pb2

CAFFERT = '/mnt/ilcompf0d0/user/zhawang/git/caffe-distilling'
MODEL_FILE = CAFFERT+'/models/bvlc_alexnet/deploy.prototxt'
PRETRAINED = CAFFERT+'/models/bvlc_alexnet/bvlc_alexnet.caffemodel'
MEAN_FILE = CAFFERT+'/data/ilsvrc12/imagenet_mean.binaryproto'
IMAGE_DIM = 256;
CROPPED_DIM = 227;
IMAGE_FILE = '/mnt/ilcompf0d0/data/imagenet/ilsvrc2012-val-256/ILSVRC2012_val_{:08d}.jpg'
IMAGE_NUM = 50000
CLASS_NUM = 1000
LABEL_FILE = '/mnt/ilcompf0d0/user/zhawang/data/ImgNet/val.label.txt'
LABEL_CVRT = CAFFERT+'/data/ilsvrc12/cf2im.txt'


blob = caffe_pb2.BlobProto()
blob_str = open(MEAN_FILE, 'rb').read()
blob.ParseFromString(blob_str)
mean_img = np.array(caffe.io.blobproto_to_array(blob))[0]
print 'mean image', mean_img.shape
dy = (mean_img.shape[1]-CROPPED_DIM)/2
dx = (mean_img.shape[2]-CROPPED_DIM)/2
mean_img = mean_img[:, dy:dy+CROPPED_DIM, dx:dx+CROPPED_DIM]

caffe.set_mode_gpu()
caffe.set_device(1)
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                        mean=mean_img, channel_swap=(2,1,0), raw_scale=255, image_dims=(IMAGE_DIM,IMAGE_DIM))
print 'Network {} loaded'.format(PRETRAINED)

# predict
prob=np.zeros([IMAGE_NUM, CLASS_NUM])
gt = np.loadtxt(LABEL_FILE).astype(np.int32)-1 #[0, 999]
gt = gt[0:IMAGE_NUM]
cf2im = np.loadtxt(LABEL_CVRT)
for i in range(IMAGE_NUM):
    fn = IMAGE_FILE.format(i+1)
    input_image = caffe.io.load_image(fn)
    dy = (input_image.shape[0]-CROPPED_DIM)/2
    dx = (input_image.shape[1]-CROPPED_DIM)/2
    mean_img = mean_img[dy:dy+CROPPED_DIM, dx:dx+CROPPED_DIM, :]
    probi = net.predict([input_image], oversample=True)
    prob[i] = probi
    #print 'prediction shape:', probi.shape
    #print 'predicted class:', probi.argmax()

    if i%100==0 or i==IMAGE_NUM-1:
        print 'image file:', fn
        # evaluate
        pred = np.argsort(prob[0:i+1], axis=1)
        pred = pred[:, ::-1]
        pred = cf2im[pred[:, 0:5]]-1 #convert to imagenet label [0, 999]
        diff=np.abs(np.reshape(np.repeat(gt[0:i+1], 5), [i+1, 5])-pred)
        acc1=np.mean(diff[:, 0]==0)
        acc5=np.mean(np.min(diff, axis=1)==0)
        print 'val {}: acc={}, {}'.format(i, acc1*100, acc5*100)
        sys.stdout.flush()
        #pickle.dump({'prob': prob, 'pred':pred}, open('./save/val_res_vgg16.p', 'wb'))

