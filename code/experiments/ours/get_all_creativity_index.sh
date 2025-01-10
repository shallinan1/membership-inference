#!/bin/bash

# Define the directories containing the .jsonl files
DIRECTORIES=("experiments/ours/outputs/bookMIA/coverages/train/" "experiments/ours/outputs/bookMIA/coverages/val/")
OUTPUT_DIR="experiments/ours/outputs/bookMIA/cis"

# Command to activate the environment
ACTIVATE_ENV="source activate vllminf"

# SLURM command to request resources (updated to run commands within it)
SLURM_COMMAND="srun -p gpu-rtx6k -A xlab --time=4:00:00 --nodes=1 --cpus-per-task=4 --mem=25G --pty /bin/bash -c"

# Iterate over each directory
for DIRECTORY in "${DIRECTORIES[@]}"; do
    # Iterate over each .jsonl file in the current directory
    for file_path in "$DIRECTORY"*.jsonl; do
        # # Skip files that do not contain "turbo" in their file path
        # if [[ "$file_path" != *"turbo"* ]]; then
        #     continue
        # fi

        # Create a new tmux session for each file and run the Python script
        cmd_with_source="python3 -m code.experiments.ours.get_creativity_index --coverage_path $file_path --output_dir $OUTPUT_DIR"
        
        tmux new-session -d "$SLURM_COMMAND \"$ACTIVATE_ENV && $cmd_with_source\""
        
        echo "\n$file_path"
    done
done
