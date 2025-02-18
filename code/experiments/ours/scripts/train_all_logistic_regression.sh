#!/bin/bash

# Define the directories containing the .jsonl files
DIRECTORIES=("experiments/ours/outputs/bookMIA/generations/train/")

# Command to activate the environment
ACTIVATE_ENV="source activate vllmgen"

# SLURM command to request resources (updated to run commands within it)
SLURM_COMMAND="srun -p gpu-rtx6k -A xlab --time=2:00:00 --nodes=1 --cpus-per-task=1 --mem=15G --pty /bin/bash -c"

# N-gram values to iterate over
NGRAM_VALUES=(4 5 6)

# Iterate over each directory
for DIRECTORY in "${DIRECTORIES[@]}"; do
    # Iterate over each .jsonl file in the current directory
    for gen_path in "$DIRECTORY"*.jsonl; do
        for ngram in "${NGRAM_VALUES[@]}"; do
            # Create a new tmux session for each file and run the Python script
            cmd_with_source="python3  -m code.experiments.ours.train_logistic_regression --gen_path $gen_path --min_ngram $ngram"
            
            tmux new-session -d "$SLURM_COMMAND \"$ACTIVATE_ENV && $cmd_with_source\""
            
            echo "Launched tmux session for $gen_path"
        done
    done
done