#!/usr/bin/env bash

ALPHAS=( 0.001 0.005 0.01 0.05 0.1 0.5 1.0 5.0 10.0 )

for alpha in ${ALPHAS[@]}; do 
    qsub job-run.pbs -v GD_TYPE=${1?},GD_ALPHA=$alpha; 
done
