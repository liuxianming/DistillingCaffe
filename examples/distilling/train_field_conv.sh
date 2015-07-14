./build/tools/caffe train -gpu=0 --solver=examples/distilling/alexnet_field_solver.prototxt 2>&1 | tee train_field_conv.log

