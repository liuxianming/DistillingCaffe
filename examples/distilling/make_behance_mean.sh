#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=examples/distilling
DATA=/mnt/ilcompf2d1/data/be/prepared-2015-06-15/LMDB/TRAINING/image
TOOLS=build/tools

$TOOLS/compute_image_mean $DATA \
  $EXAMPLE/behance.binaryproto

echo "Done."
