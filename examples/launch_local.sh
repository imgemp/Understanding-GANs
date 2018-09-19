#!/bin/bash

export MKL_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export OMP_NUM_THREADS=3

python ugans/run.py $(cat examples/args/circles/simgd/00.txt) &

