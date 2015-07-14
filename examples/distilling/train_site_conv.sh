./build/tools/caffe train -gpu=1 --solver=examples/distilling/alexnet_site_solver.prototxt 2>&1 | tee train_site_conv.log

