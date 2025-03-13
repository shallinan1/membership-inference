python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-3.5-turbo-0125 \
    --start_sentence 1 \
    --num_sentences 5 \
    --num_sequences 20 \
    --max_tokens 512 \
    --task_prompt_idx 4 5 \
    --task bookMIA \
    --data_split train;

python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-3.5-turbo-0125 \
    --start_sentence 1 \
    --num_sentences 5 \
    --num_sequences 20 \
    --max_tokens 512 \
    --task_prompt_idx 4 5 \
    --temperature 0.2 \
    --task bookMIA \
    --data_split train;

python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-3.5-turbo-0125 \
    --start_sentence 1 \
    --num_sentences 5 \
    --num_sequences 10 \
    --max_tokens 512 \
    --task_prompt_idx 1 2 4 5 \
    --task bookMIA \
    --data_split train;

python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-3.5-turbo-0125 \
    --start_sentence 1 \
    --num_sentences 5 \
    --num_sequences 10 \
    --max_tokens 512 \
    --task_prompt_idx 1 2 4 5 \
    --temperature 0.2 \
    --task bookMIA \
    --data_split train;