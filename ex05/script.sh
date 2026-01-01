#!/bin/bash

start_time=$(date +%s)
t0=$(date +%s)

if [ -n "$SLURM_JOB_ID" ]; then
NO_DISPLAY='--no_display'
SCORE='--score'
SCRIPT_PY="modular_pipeline.py"
else
SCORE='--score'
SCRIPT_PY="local_test.py"
NO_DISPLAY=''
fi

echo ">> Evaluating model"
t1=$(date +%s)
python3 $SCRIPT_PY $SCORE $NO_DISPLAY
elapsed=$(( t1 - t0 ))
echo "== Evaluation completed in $elapsed seconds."
echo

t3=$(date +%s)
total_elapsed=$(( t3 - start_time ))
echo "== Total time: $total_elapsed seconds."
