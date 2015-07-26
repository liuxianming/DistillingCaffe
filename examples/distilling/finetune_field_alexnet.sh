./build/tools/caffe train -gpu=0 --solver=examples/distilling/alexnet_field_finetune_solver.prototxt \
	-weights models/bvlc_alexnet/bvlc_alexnet.caffemodel \
	2>&1 | tee finetune_field_conv.log

