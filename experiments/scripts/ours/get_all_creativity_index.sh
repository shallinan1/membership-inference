#!/bin/bash
# ./code/experiments/ours/scripts/get_all_creativity_index.sh

TASK="${1:-tulu_v1}"
SPLIT="${2:-train}"
MIN_NGRAM=1
MAX_NGRAM=12

# Define the directories containing the .jsonl files
DIRECTORY="outputs/ours/${TASK}/coverages/${SPLIT}/"
OUTPUT_DIR="outputs/ours/${TASK}/creativities/${SPLIT}/"

# Iterate over each .jsonl file in the current directory
for file_path in "$DIRECTORY"*.jsonl; do
    filename=$(basename "$file_path" .jsonl)
    output_file="${OUTPUT_DIR}${filename}_CI${MIN_NGRAM}-${MAX_NGRAM}.jsonl"

    if [ ! -f "$output_file" ]; then
        cmd="python3 -m code.experiments.ours.compute_creativity_index \
            --coverage_path $file_path \
            --output_dir $OUTPUT_DIR \
            --min_ngram $MIN_NGRAM \
            --max_ngram $MAX_NGRAM"
        
        echo "$cmd"
        echo

        sbatch -A xlab -p gpu-rtx6k \
               --time=1:00:00 \
               --nodes=1 \
               --cpus-per-task=1 \
               --mem=25G \
               --gpus=0 \
               --job-name="creativity_${filename}" \
               --wrap="source activate vllmgen && $cmd"
    fi
done