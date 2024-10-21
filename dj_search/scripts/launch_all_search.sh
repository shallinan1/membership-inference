#!/bin/bash

# User-specified folder for gen_data files
GEN_DATA_FOLDER="/gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/generations"

# Fixed output directory
OUTPUT_DIR="/gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/coverages/"

# Fixed task
TASK="bookMIA"

# Source docs path
SOURCE_DOCS="swj0419/BookMIA"

# Min ngram values to try
MIN_NGRAM_VALUES=(3 4 5 6)
MIN_NGRAM_VALUES=(4 5)

# Command to activate the environment
ACTIVATE_ENV="source activate vllminf"

# SLURM command to request resources (updated to run commands within it)
SLURM_COMMAND="srun -p gpu-rtx6k -A xlab --time=8:00:00 --nodes=1 --cpus-per-task=4 --mem=25G --pty /bin/bash -c"

# Iterate over gen_data files in the folder
for gen_data_file in "$GEN_DATA_FOLDER"/*.jsonl; do
   # Extract the base filename from the gen_data file (without the .jsonl extension)
   base_filename=$(basename "$gen_data_file" .jsonl)

   # Iterate over min_ngram values
   for min_ngram in "${MIN_NGRAM_VALUES[@]}"; do
      # Output file paths
      output_without_source="${OUTPUT_DIR}/${base_filename}_${min_ngram}_onedoc.jsonl"
      output_with_source="${OUTPUT_DIR}/${base_filename}_${min_ngram}_alldoc.jsonl"

      # Check if the output file without --source_docs exists
      if [ ! -f "$output_without_source" ]; then
         # Command without --source_docs
         cmd_without_source="python3 -m dj_search.dj_search_exact_LLM \
            --task $TASK \
            --output_dir $OUTPUT_DIR \
            --gen_data $gen_data_file \
            --min_ngram $min_ngram \
            --parallel"

         # Print the command without --source_docs
         echo "Running command: $cmd_without_source"

         # Create a new tmux session and run the SLURM command inside
         tmux new-session -d "$SLURM_COMMAND \"$ACTIVATE_ENV && $cmd_without_source\""
      else
         :
         # echo "Output file already exists: $output_without_source, skipping..."
      fi

      # Check if the output file with --source_docs exists
      if [ ! -f "$output_with_source" ]; then
         # Command with --source_docs
         cmd_with_source="python3 -m dj_search.dj_search_exact_LLM \
            --task $TASK \
            --output_dir $OUTPUT_DIR \
            --gen_data $gen_data_file \
            --min_ngram $min_ngram \
            --parallel \
            --source_docs $SOURCE_DOCS"

         # Print the command with --source_docs
         echo "Running command: $cmd_with_source"

         # Create a new tmux session and run the SLURM command inside
         tmux new-session -d "$SLURM_COMMAND \"$ACTIVATE_ENV && $cmd_with_source\""
      else
         :
         # echo "Output file already exists: $output_with_source, skipping..."
      fi
   done
done