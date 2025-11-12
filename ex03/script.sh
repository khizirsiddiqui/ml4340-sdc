#!/bin/bash

start_time=$(date +%s)
TOTAL_TIMESTEPS=100000
ACTION_REPEATS=4
GAMMA=0.99
ACTION_FILENAME='default_actions.txt'
USE_DOUBLEQLEARNING=''
NO_DISPLAY='--no_display'

NUM_EPOCHS=350
echo ">> Training"
python train_racing.py --total_timesteps $TOTAL_TIMESTEPS --action_repeats $ACTION_REPEATS --gamma $GAMMA --action_filename $ACTION_FILENAME $USE_DOUBLEQLEARNING $NO_DISPLAY
t0=$(date +%s)
elapsed=$(( t0 - start_time ))
echo "== Training completed in $elapsed seconds."

echo ">> Evaluating model from $MODEL_PATH"
python evaluate_racing.py $NO_DISPLAY --action_filename $ACTION_FILENAME --agent_name "latest"
t2=$(date +%s)
elapsed=$(( t2 - t1 ))
echo "== Evaluation completed in $elapsed seconds."
echo

t3=$(date +%s)
total_elapsed=$(( t3 - start_time ))
echo "== Total time: $total_elapsed seconds."

