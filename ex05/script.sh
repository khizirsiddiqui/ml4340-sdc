#!/bin/bash

NO_DISPLAY='--no_display'
start_time=$(date +%s)
t0=$(date +%s)

if [ -n "$SLURM_JOB_ID" ]; then
FORCE_TRAINING=''
else
echo ">> Evaluating model"
python3 modular_pipeline.py --score $NO_DISPLAY
t1=$(date +%s)
elapsed=$(( t1 - t0 ))
echo "== Evaluation completed in $elapsed seconds."
echo
fi

t3=$(date +%s)
total_elapsed=$(( t3 - start_time ))
echo "== Total time: $total_elapsed seconds."
