#!/bin/bash

TOTAL_TIMESTEPS=200000
ACTION_REPEATS=4
GAMMA=0.99
BATCH_SIZE=256
ACTION_FILENAME='actions.txt'
USE_DOUBLEQLEARNING=''
NO_DISPLAY='--no_display'

start_time=$(date +%s)
echo ">> Training"
python train_racing.py --total_timesteps $TOTAL_TIMESTEPS --action_repeats $ACTION_REPEATS --gamma $GAMMA --batch_size $BATCH_SIZE --action_filename $ACTION_FILENAME $USE_DOUBLEQLEARNING $NO_DISPLAY
t0=$(date +%s)
elapsed=$(( t0 - start_time ))
echo "== Training completed in $elapsed seconds."

echo ">> Evaluating model from $MODEL_PATH"
python local_eval_race.py $NO_DISPLAY --action_filename $ACTION_FILENAME --agent_name "latest"
t1=$(date +%s)
elapsed=$(( t1 - t0 ))
echo "== Evaluation completed in $elapsed seconds."
echo

t2=$(date +%s)
total_elapsed=$(( t2 - start_time ))
echo "== Total time: $total_elapsed seconds."

