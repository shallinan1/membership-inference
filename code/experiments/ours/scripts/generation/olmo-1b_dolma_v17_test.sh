python3 -m code.experiments.ours.generate \
    --model allenai/OLMo-1B-0724-hf \
    --max_tokens 512 \
    --task_prompt_idx 0 \
    --task dolma_v17 \
    --data_split test \
    --num_sequences 20 \
    --temperature 1.0 \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_proportion_from_end 0.25;

python3 -m code.experiments.ours.generate \
    --model allenai/OLMo-1B-0724-hf \
    --max_tokens 512 \
    --task_prompt_idx 0 \
    --task dolma_v17 \
    --data_split test \
    --num_sequences 20 \
    --temperature 1.0 \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_words_from_end 50;

python3 -m code.experiments.ours.generate \
    --model allenai/OLMo-1B-0724-hf \
    --max_tokens 512 \
    --task_prompt_idx 1 \
    --task dolma_v17 \
    --data_split test \
    --num_sequences 20 \
    --temperature 1.0 \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_words_from_end 50;