#!/usr/bin/env bash

ALPHAS=( 0.01 0.03 0.05 0.075 0.1 0.3 0.5 0.75 1 )
THREADS=( 1 2 4 8 12 16 20 24 )

for alpha in ${ALPHAS[@]}; do 
    for num_threads in ${THREADS[@]}; do
        qsub job-run.pbs -v GD_TYPE=${1?},GD_ALPHA=$alpha,GD_NUM_THREADS=$num_threads
    done
done
