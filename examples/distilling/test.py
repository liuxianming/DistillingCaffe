import sys
import os
import subprocess

command = "./build/tools/caffe train"
solver = "--solver=./examples/cifar10/cifar10_quick_solver.prototxt"

#os.fork(command + " --gpu 0 " + solver)
#os.fork(command + " --gpu 1 " + solver)

p1 = subprocess.Popen("./build/tools/caffe train --gpu 0 --solver=./examples/cifar10/cifar10_quick_solver.prototxt > 0.txt 2>&1", shell=True)
p2 = subprocess.Popen("./build/tools/caffe train --gpu 1 --solver=./examples/cifar10/cifar10_quick_solver.prototxt > 1.txt 2>&1", shell=True)
