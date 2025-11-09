start_time=$(date +%s)

NUM_EPOCHS=350
MODEL_PATH=train18.pth
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
python main.py --no_display --score --agent_load_path $MODEL_PATH
t2=$(date +%s)
elapsed=$(( t2 - t1 ))
echo "== Evaluation completed in $elapsed seconds."
echo

t3=$(date +%s)
total_elapsed=$(( t3 - start_time ))
echo "== Total time: $total_elapsed seconds."

