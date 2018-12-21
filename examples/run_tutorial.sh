#!/bin/bash

export MKL_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export OMP_NUM_THREADS=3

set -exu

PYTHONPATH=./ python examples/domains/dcgan_faces_tutorial.py
