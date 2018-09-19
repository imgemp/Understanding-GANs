#!/bin/bash

export MKL_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export OMP_NUM_THREADS=3

set -exu

input=$1

PYTHONPATH=./ python ugans/run.py $(cat $input)
