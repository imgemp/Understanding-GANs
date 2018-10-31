#!/bin/bash

export MKL_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export OMP_NUM_THREADS=3

set -exu

argfile=$1
xtraargs=${2:-""}  # if 2nd input is unset or null, xtraargs is set to ""

PYTHONPATH=./ python ugans/run.py $(cat $argfile) $xtraargs
