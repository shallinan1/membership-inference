# o1 mini
python3 -m code.experiments.ours.generate \
    --openai \
    --model o1-mini-2024-09-12 \
    --start_sentence 1 \
    --num_sentences 5 \
    --num_sequences 20 \
    --max_tokens 512 \
    --task_prompt_idx 5 \
    --task bookMIA \
    --data_split val;
