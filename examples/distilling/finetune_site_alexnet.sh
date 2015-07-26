./build/tools/caffe train -gpu=0 --solver=examples/distilling/alexnet_site_finetune_solver.prototxt \
	-weights models/bvlc_alexnet/bvlc_alexnet.caffemodel \
	2>&1 | tee finetune_site_conv.log

