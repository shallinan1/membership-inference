#!/bin/bash
# ./code/experiments/ours/scripts/get_all_creativity_index.sh

TASK="${1:-tulu_v1}"
SPLIT="${2:-train}"
MIN_NGRAM=1
MAX_NGRAM=12

# Define the directories containing the .jsonl files
DIRECTORY="outputs/ours/${TASK}/coverages/${SPLIT}/"
OUTPUT_DIR="outputs/ours/${TASK}/creativities/${SPLIT}/"

ACTIVATE_ENV="source activate vllmgen"
SLURM_COMMAND="srun -A cse-ckpt -p ckpt --time=1:00:00 --nodes=1 --cpus-per-task=1 --mem=25G --gres=gpu:rtx6k:0 --pty /bin/bash -c"
SLURM_COMMAND="srun -A xlab -p gpu-rtx6k --time=1:00:00 --nodes=1 --cpus-per-task=1 --mem=25G --gres=gpu:rtx6k:0 --pty /bin/bash -c"
SLURM_COMMAND="srun -A xlab -p gpu-a100 --time=1:00:00 --nodes=1 --cpus-per-task=1 --mem=25G --gpus 0 --pty /bin/bash -c"

# Iterate over each .jsonl file in the current directory
for file_path in "$DIRECTORY"*.jsonl; do
    filename=$(basename "$file_path" .jsonl)
    output_file="${OUTPUT_DIR}${filename}_CI${MIN_NGRAM}-${MAX_NGRAM}.jsonl"

    if [ ! -f "$output_file" ]; then
        # Create a new tmux session for each file and run the Python script
        cmd_with_source="python3 -m code.experiments.ours.get_creativity_index --coverage_path $file_path --output_dir $OUTPUT_DIR --min_ngram $MIN_NGRAM --max_ngram $MAX_NGRAM"
        
        echo "$cmd_with_source"
        echo
        tmux new-session -d "$SLURM_COMMAND \"$ACTIVATE_ENV && $cmd_with_source\""
    fi
done