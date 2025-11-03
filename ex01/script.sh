start_time=$(date +%s)

MODEL_PATH=train2.pth
echo ">> Training model and saving to $MODEL_PATH"
python main.py --train --nr_epochs 1000 --agent_save_path $MODEL_PATH
t0=$(date +%s)
elapsed=$(( t0 - start_time ))
echo "== Training completed in $elapsed seconds."


echo ">> Testing model from $MODEL_PATH"
python main.py --test --agent_load_path $MODEL_PATH
t1=$(date +%s)
elapsed=$(( t1 - t0 ))
echo "== Testing completed in $elapsed seconds."

echo ">> Evaluating model from $MODEL_PATH"
OUTPUT=$(python main.py --score --agent_load_path $MODEL_PATH)
t2=$(date +%s)
elapsed=$(( t2 - t1 ))
echo "== Evaluation completed in $elapsed seconds."
echo $OUTPUT

t3=$(date +%s)
total_elapsed=$(( t3 - start_time ))
echo "== Total time: $total_elapsed seconds."
echo ">> Sending notification to tg.."
cd ~/util-scripts
bash send_tg.sh "Training and evaluation completed in $total_elapsed seconds. Score: $OUTPUT"
echo "== Notification sent."
