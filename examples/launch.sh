#!/bin/bash

export MKL_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export OMP_NUM_THREADS=3

set -exu

# configs=( "args/circles/simgd/00.txt" "args/circles/simgd/01.txt" )

# for c in ${configs[@]}
for i in $(seq 0 9); do
	if [ $i -lt 10 ]; then
		sbatch -p m40-long --gres=gpu:1 examples/run.sh "examples/args/circles/simgd/0"$i".txt"
	else
		sbatch -p m40-long --gres=gpu:1 examples/run.sh "examples/args/circles/simgd/"$i".txt"
	fi
done