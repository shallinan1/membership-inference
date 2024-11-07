#!/bin/bash

# Define the directory containing the .jsonl files
DIRECTORY="experiments/ours/bookMIA/coverages/val/"
OUTPUT_DIR="experiments/ours/bookMIA/cis"

# Command to activate the environment
ACTIVATE_ENV="source activate vllminf"

# SLURM command to request resources (updated to run commands within it)
SLURM_COMMAND="srun -p gpu-rtx6k -A xlab --time=8:00:00 --nodes=1 --cpus-per-task=4 --mem=25G --pty /bin/bash -c"

# Iterate over each .jsonl file in the directory
for file_path in "$DIRECTORY"*.jsonl; do
    
    # Create a new tmux session for each file and run the Python script
    cmd_with_source="python3 -m experiments.ours.get_creativity_index --coverage_path $file_path --output_dir $OUTPUT_DIR"
    
    tmux new-session -d "$SLURM_COMMAND \"$ACTIVATE_ENV && $cmd_with_source\""

    echo "Launched tmux session for $file_path"
done
