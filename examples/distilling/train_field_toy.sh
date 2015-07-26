#!/usr/bin/env sh

./build/tools/caffe train -gpu=0\
    --solver=./examples/distilling/conv_field_solver.prototxt
