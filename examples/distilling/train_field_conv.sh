#!/usr/bin/env sh

# setting up environment variables
source /mnt/ilcompf0d0/user/xliu/libs/profile

host=$(hostname)
slot=$_CONDOR_SLOT
echo on $host:$slot

basedir=/mnt/ilcompf0d0/user/xliu
codedir=${basedir}/code/caffe

gpu_id=$(( ${slot#slot}-1 )) #gpu_id = slot_id-1
if (( gpu_id==-1 ))
then
    echo 'unable to find available gpu! use gpu0 anyway'
    gpu_id=0
    #exit
else
    echo find free gpu ${gpu_id}!
    echo Training behance_field model
    # running job
    cd ${codedir}
    ./build/tools/caffe train -gpu ${gpu_id} \
        --solver=./examples/distilling/conv_field_solver.prototxt
fi
