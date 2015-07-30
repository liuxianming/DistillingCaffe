"""Util functions for python layer
"""
import scipy.misc
import numpy as np
import caffe
from caffe.io import caffe_pb2

"""Some util function
"""
"""
Load data from either datum or from original image
Here, I used caffe.io.read_datum_from_image to resolve all format transformation,
instead of the io functions provided by caffe python wrapper
"""
def extract_sample_from_datum(datum, image_mean, resize=-1):
    # extract numpy array from datum, then substract the mean_image
    img_data = decode_datum(datum)
    img_data = substract_mean(img_data, image_mean)
    if resize == -1:
        return img_data
    else:
        # resize image: first transfer back to h * w * c, -> resize -> c * h * w
        resize_img_data(img_data, resize)

def resize_img_data(img_data, resize):
    img_data = img_data.transpose(1,2,0)
    img_data = scipy.misc.imresize(img_data, (resize, resize))
    return img_data.transpose(2,0,1)

def decode_datum(datum):
    if datum.encoded == True:
        datum = caffe.io.decode_datum(datum)
    img_data = caffe.io.datum_to_array(datum)
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
    return img

"""--------------------------------------------------------------------------"""
