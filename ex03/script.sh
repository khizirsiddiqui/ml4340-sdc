#!/bin/bash

ACTION_REPEATS=4
GAMMA=0.99
ACTION_FILENAME='actions.txt'
USE_DOUBLEQLEARNING=''
NO_DISPLAY='--no_display'

if [ -n "$SLURM_JOB_ID" ]; then
TOTAL_TIMESTEPS=200000
BATCH_SIZE=256
NO_DISPLAY='--no_display'
EVALUATION_OVER_N_SEEDS=200
FORCE_TRAINING=''
else
TOTAL_TIMESTEPS=500
BATCH_SIZE=16
EVALUATION_OVER_N_SEEDS=10
FORCE_TRAINING='--force'
fi

start_time=$(date +%s)
echo ">> Training"
python train_racing.py --total_timesteps $TOTAL_TIMESTEPS --action_repeats $ACTION_REPEATS --gamma $GAMMA --batch_size $BATCH_SIZE --action_filename $ACTION_FILENAME $USE_DOUBLEQLEARNING $NO_DISPLAY $FORCE_TRAINING
t0=$(date +%s)
elapsed=$(( t0 - start_time ))
echo "== Training completed in $elapsed seconds."

echo ">> Evaluating model from $MODEL_PATH"
python local_eval_race.py $NO_DISPLAY --action_filename $ACTION_FILENAME --agent_name "latest" --N $EVALUATION_OVER_N_SEEDS
t1=$(date +%s)
elapsed=$(( t1 - t0 ))
echo "== Evaluation completed in $elapsed seconds."
echo

# Evaluation used for the final submission
# echo ">> Evaluating model from $MODEL_PATH for submission"
# python evaluate_racing.py $NO_DISPLAY --action_filename $ACTION_FILENAME --agent_name "submission"
# t2=$(date +%s)
# elapsed=$(( t2 - t1 ))
# echo "== Evaluation completed in $elapsed seconds."
# echo

t3=$(date +%s)
total_elapsed=$(( t3 - start_time ))
echo "== Total time: $total_elapsed seconds."

