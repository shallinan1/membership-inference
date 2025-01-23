#!/bin/bash

# ./code/experiments/ours/scripts/run_all_scoring.sh


ACTIVATE_ENV="source activate vllmgen"
SLURM_COMMAND="srun -A cse-ckpt -p ckpt --time=1:00:00 --nodes=1 --cpus-per-task=1 --mem=25G --gres=gpu:rtx6k:0 --pty /bin/bash -c"

# Define dataset and split
DATASET="pile_external"
SPLIT="train"

# Build paths dynamically
TARGET_DIR="outputs/ours/${DATASET}/coverages/${SPLIT}"

# Iterate through each file in the target directory
for TARGET_FILE in "$TARGET_DIR"/*.jsonl; do
    cmd="python3 -m code.experiments.ours.run_scores \
        --outputs_file $TARGET_FILE"
    
    echo "Running command: $cmd"
    echo

    tmux new-session -d "$SLURM_COMMAND \"$ACTIVATE_ENV && $cmd\""
done
