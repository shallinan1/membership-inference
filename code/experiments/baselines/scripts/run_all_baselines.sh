#!/bin/bash

# ./code/experiments/baselines/scripts/run_all_baselines.sh


ACTIVATE_ENV="source activate vllmgen"
SLURM_COMMAND="srun -p gpu-rtx6k -A xlab --time=1:00:00 --nodes=1 --cpus-per-task=4 --mem=25G --gres=gpu:rtx6k:0 --pty /bin/bash  -c"

# Define dataset and split
DATASET="pile_external"
SPLIT="test"

# Build paths dynamically
TARGET_DIR="/gscratch/xlab/hallisky/membership-inference/outputs/baselines/${DATASET}/${SPLIT}/probs"
REF_MODEL_PROBS="${TARGET_DIR}/stablelm-base-alpha-3b-v2.jsonl"

# Iterate through each file in the target directory
for TARGET_FILE in "$TARGET_DIR"/*.jsonl; do
    cmd="python3 -m code.experiments.baselines.run_baselines \
        --target_model_probs $TARGET_FILE \
        --ref_model_probs $REF_MODEL_PROBS"

    echo "Running command: $cmd"
    echo

    tmux new-session -d "$SLURM_COMMAND \"$ACTIVATE_ENV && $cmd\""
done
