#!/bin/bash
# ./code/helper/dj_search/scripts/launch_all_search.sh

# TODO FIX ALL DOCS 

# Fixed task
TASK="pile_external"
SPLIT="test"

# User-specified folder for gen_data files
GEN_DATA_FOLDER="outputs/ours/${TASK}/generations/${SPLIT}"

# Fixed output directory
OUTPUT_DIR="${GEN_DATA_FOLDER//generations/coverages}"

# Source docs path
if [ "$SOURCE_DOCS" == "BookMIA" ]; then
   SOURCE_DOCS="swj0419/BookMIA"
else
   SOURCE_DOCS="empty"
fi

# Min ngram values to try
MIN_NGRAM_VALUES=(3 4 5 6)
MIN_NGRAM_VALUES=(2 3 4)

# Command to activate the environment
ACTIVATE_ENV="source activate vllmgen"

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
         cmd_without_source="python3 -m code.helper.dj_search.dj_search_exact_LLM \
            --task $TASK \
            --output_dir $OUTPUT_DIR \
            --gen_data $gen_data_file \
            --min_ngram $min_ngram \
            --parallel"

         # Print the command without --source_docs
         echo "Running (no source) command: $cmd_without_source"

         # Create a new tmux session and run the SLURM command inside
         tmux new-session -d "$SLURM_COMMAND \"$ACTIVATE_ENV && $cmd_without_source\""
      else
         :
         # echo "Output file already exists: $output_without_source, skipping..."
      fi

      # Check if the output file with --source_docs exists
      if [ ! -f "$output_with_source" ] && [ "$SOURCE_DOCS" != "empty" ]; then
         # Command with --source_docs
         cmd_with_source="python3 -m code.helper.dj_search.dj_search_exact_LLM \
            --task $TASK \
            --output_dir $OUTPUT_DIR \
            --gen_data $gen_data_file \
            --min_ngram $min_ngram \
            --parallel \
            --source_docs $SOURCE_DOCS"

         # Print the command with --source_docs
         echo "Running (source) command: $cmd_with_source"

         # Create a new tmux session and run the SLURM command inside
         tmux new-session -d "$SLURM_COMMAND \"$ACTIVATE_ENV && $cmd_with_source\""
      else
         :
         # echo "Output file already exists: $output_with_source, skipping..."
      fi
   done
done
