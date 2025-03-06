python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-3.5-turbo-0125 \
    --start_sentence 1 \
    --num_sentences 5 \
    --num_sequences 20 \
    --max_tokens 512 \
    --task_prompt_idx 5 4 \
    --task bookMIA \
    --data_split train;