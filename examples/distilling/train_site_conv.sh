#!/usr/bin/env sh
export LD_LIBRARY_PATH="/mnt/ilcompf0d0/user/xliu/libs/syslib:$LD_LIBRARY_PATH"
export PATH="/mnt/ilcompf0d0/user/xliu/libs/syslib:$PATH"
# setting up python libs
export PYTHONPATH="$PYTHONPATH:/mnt/ilcompf0d0/user/xliu/libs/python2.7/dist-packages"
export PYTHON_INCLUDE="$PYTHON_INCLUDE:/mnt/ilcompf0d0/user/xliu/libs/include:/mnt/ilcompf0d0/user/xliu/libs/include/python2.7"
export CPATH="$CPATH:/mnt/ilcompf0d0/user/xliu/libs/include/python2.7:/mnt/ilcompf0d0/user/xliu/libs/include"

host=$(hostname)
slot=$_CONDOR_SLOT
echo on $host:$slot

basedir=/mnt/ilcompf0d0/user/xliu
codedir=${basedir}/code/caffe
datadir=/scratch/be/

if [ ! -d "${datadir}" ]; then
    # copy data to datadir
    echo Copying data from ilcompf2 to local scratch folder
    mkdir ${datador}
    cp -r /mnt/ilcompf2d1/data/be/prepared-2015-06-15/LMDB ${datadir}LMDB
fi

gpu_id=$(( ${slot#slot}-1 )) #gpu_id = slot_id-1

if [ "$gpu_id" -eq -1 ]; then
    echo 'unable to find available gpu! use gpu0 anyway'
    gpu_id=0
    #exit
else
    echo find free gpu ${gpu_id}!
    echo Training behance_site model
fi

echo gpu_id = ${gpu_id}
cd ${codedir}
./build/tools/caffe train -gpu=$gpu_id \
    --solver=./examples/distilling/conv_site_solver.prototxt 2>&1
