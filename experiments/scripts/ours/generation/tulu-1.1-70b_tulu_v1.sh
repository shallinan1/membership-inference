python3 -m src.attacks.ngram_coverage_attack.generate \
    --model allenai/tulu-v1-llama2-70b \
    --num_sequences 20 \
    --max_tokens 512 \
    --task_prompt_idx 0 \
    --task tulu_v1 \
    --data_split test;

python3 -m src.attacks.ngram_coverage_attack.generate \
    --model allenai/tulu-v1-llama2-70b \
    --num_sequences 50 \
    --max_tokens 512 \
    --task_prompt_idx 0 \
    --task tulu_v1 \
    --temperature 0.2 \
    --data_split test;

# To max seq length
python3 -m src.attacks.ngram_coverage_attack.generate \
    --model allenai/tulu-v1-llama2-70b \
    --num_sequences 20 \
    --max_tokens 512 \
    --task_prompt_idx 0 \
    --task tulu_v1 \
    --max_length_to_sequence_length \
    --data_split test;
