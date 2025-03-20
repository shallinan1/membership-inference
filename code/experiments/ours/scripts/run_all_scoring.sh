#!/bin/bash
# ./code/experiments/ours/scripts/run_all_scoring.sh

ACTIVATE_ENV="source activate vllmgen"
SLURM_COMMAND="srun -A cse-ckpt -p ckpt --time=1:00:00 --nodes=1 --cpus-per-task=1 --mem=25G --gres=gpu:rtx6k:0 --pty /bin/bash -c"
SLURM_COMMAND="srun -A xlab -p gpu-rtx6k --time=1:00:00 --nodes=1 --cpus-per-task=1 --mem=25G --gres=gpu:rtx6k:0 --pty /bin/bash -c"

# Define dataset and split
TASK="${1:-tulu_v1}"
SPLIT="${2:-train}"

# Build paths dynamically
TARGET_DIR="outputs/ours/${TASK}/creativities/${SPLIT}"

# Iterate through each file in the target directory
for TARGET_FILE in "$TARGET_DIR"/*.jsonl; do
    # Get the filename without the ".jsonl" extension
    FILE_BASE=$(basename "$TARGET_FILE" .jsonl)
    SCORES_DIR="outputs/ours/${TASK}/scores/${SPLIT}/${FILE_BASE}"
    if [ ! -d "$SCORES_DIR" ]; then
        cmd="python3 -m code.experiments.ours.run_scores \
        --outputs_file $TARGET_FILE" 
        echo "$cmd;"
        echo

        tmux new-session -d "$SLURM_COMMAND \"$ACTIVATE_ENV && $cmd\""
    fi
done
