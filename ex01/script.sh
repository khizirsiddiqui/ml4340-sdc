start_time=$(date +%s)

NUM_EPOCHS=50
MODEL_PATH=train8.pth
echo ">> Training model and saving to $MODEL_PATH"
python main.py --train --nr_epochs $NUM_EPOCHS --agent_save_path $MODEL_PATH
t0=$(date +%s)
elapsed=$(( t0 - start_time ))
echo "== Training completed in $elapsed seconds."


echo ">> Testing model from $MODEL_PATH"
python main.py --test --agent_load_path $MODEL_PATH --no_display
t1=$(date +%s)
elapsed=$(( t1 - t0 ))
echo "== Testing completed in $elapsed seconds."

echo ">> Evaluating model from $MODEL_PATH"
OUTPUT=$(python main.py --no_display --score --agent_load_path $MODEL_PATH)
t2=$(date +%s)
elapsed=$(( t2 - t1 ))
echo "== Evaluation completed in $elapsed seconds."
echo $OUTPUT
echo

t3=$(date +%s)
total_elapsed=$(( t3 - start_time ))
echo "== Total time: $total_elapsed seconds."

# send telegram notification with SLURM JOB_ID and total time
job_id=${SLURM_JOB_ID:-"N/A"}
job_name=${SLURM_JOB_NAME:-"N/A"}
MESSAGE="Job ID: $job_id, Job Name: $job_name, Total Time: $total_elapsed seconds."
cd ~/util-scripts
bash send_tg.sh "$MESSAGE"
echo

