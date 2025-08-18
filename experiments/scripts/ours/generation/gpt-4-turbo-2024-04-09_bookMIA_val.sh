# gpt-4-turbo
python3 -m src.attacks.ngram_coverage_attack.generate \
    --openai \
    --model gpt-4-turbo-2024-04-09 \
    --start_sentence 1 \
    --num_sentences 5  \
    --num_sequences 20 \
    --max_tokens 512 \
    --task_prompt_idx 5 \
    --task bookMIA \
    --data_split val;