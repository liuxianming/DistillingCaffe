#!/bin/bash
# network training on caffe
# arguments from condor job: ${Cluster} $(Process) $(condor_job_size)

clusterid=$1
jobid=$2
njob=$3

host=$(hostname)
slot=$_CONDOR_SLOT
echo on $host:$slot
echo main_train.sh, cluster ${clusterid}, job ${jobid}/${njob}

export LD_LIBRARY_PATH=/mnt/ilcompf0d0/user/zhawang/lib/syslib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/mnt/ilcompf0d0/user/zhawang/lib/mkl/lib/intel64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/mnt/ilcompf0d0/user/zhawang/lib/cuda/cudnn3/lib64:${LD_LIBRARY_PATH}
export PYTHONPATH=/mnt/ilcompf0d0/user/zhawang/utils/local/lib/python2.7/site-packages:$PYTHONPATH
export PYTHONPATH=/mnt/ilcompf0d0/user/zhawang/utils/local/lib/python2.7/dist-packages:$PYTHONPATH
export PYTHONPATH=/mnt/ilcompf0d0/user/zhawang/git/caffe-distilling/python:$PYTHONPATH

gpu_id=$(( ${slot#slot}-1 )) #gpu_id = slot_id-1
if (( gpu_id==-1 ))
then
    echo 'unable to find available gpu! use gpu0 anyway'
    gpu_id=0
    #exit
else
    echo find free gpu ${gpu_id}!
fi

caffe_dir=/mnt/ilcompf0d0/user/zhawang/git/caffe-distilling
base_model=${caffe_dir}/models/bvlc_alexnet/bvlc_alexnet.caffemodel
work_dir=${caffe_dir}/examples/distilling/ImageNet
bin=${work_dir}/train.py
solver=${work_dir}/solver.prototxt
netdef=${work_dir}/train_val.prototxt

train_data=/mnt/ilcompf0d0/data/imagenet/imagenet1k-256.bcf
train_label=/mnt/ilcompf0d0/user/zhawang/data/ImgNet/train.label.txt
test_data=/mnt/ilcompf0d0/data/imagenet/ilsvrc2012-val-256.bcf
test_label=/mnt/ilcompf0d0/user/zhawang/data/ImgNet/val.label.txt
label_map=${caffe_dir}/data/ilsvrc12/im2cf.txt #map from imagenet label to caffe label
mean_file=${caffe_dir}/data/ilsvrc12/imagenet_mean.binaryproto
batch_size=256
resize=227
label_subset=./subset_list.txt
sample_rate=0.05

tasks=(n01503061_bird n01767661_arthropod n02528163_fish n04524313_vehicle n01661091_reptile n02075296_carnivore n03800933_instrument)
task=${tasks[$jobid]}
task_dir=${work_dir}/${task}

# write config files
solver_target=$task_dir/$(basename $solver)
netdef_target=$task_dir/$(basename $netdef)
cp $solver $solver_target
cp $netdef $netdef_target
sed -i -- 's:<NET_HOLDER>:./'$(basename $netdef_target)':g' $solver_target

sed -i -- 's/<BATCH_SIZE_HOLDER>/'${batch_size}'/g' $netdef_target
sed -i -- 's/<RESIZE_HOLDER>/'${resize}'/g' $netdef_target
sed -i -- 's:<TRAIN_SRC_HOLDER>:'${train_data}':g' $netdef_target
sed -i -- 's:<TRAIN_LABEL_SRC_HOLDER>:'${train_label}':g' $netdef_target
sed -i -- 's:<TEST_SRC_HOLDER>:'${test_data}':g' $netdef_target
sed -i -- 's:<TEST_LABEL_SRC_HOLDER>:'${test_label}':g' $netdef_target
sed -i -- 's:<LABEL_MAP_HOLDER>:'${label_map}':g' $netdef_target
sed -i -- 's:<MEAN_FILE_HOLDER>:'${mean_file}':g' $netdef_target
sed -i -- 's:<LABEL_SUBSET_HOLDER>:'${label_subset}':g' $netdef_target
sed -i -- 's/<SAMPLE_RATE_HOLDER>/'${sample_rate}'/g' $netdef_target

# execution
cd ${task_dir}
arg="$bin $solver_target $gpu_id $base_model"
echo $arg
python $arg

