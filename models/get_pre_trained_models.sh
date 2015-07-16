#!/bin/bash env
if [ $1 == 'alexnet' ]; then
    wget http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel  
fi

if [ $1 == 'googlenet' ]; then
    wget http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel
fi 
