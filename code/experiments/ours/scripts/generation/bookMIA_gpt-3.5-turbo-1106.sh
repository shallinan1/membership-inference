python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-3.5-turbo-1106 \
    --start_sentence 1 \
    --num_sentences 5 \
    --num_sequences 20 \
    --max_tokens 512 \
    --task_prompt_idx 4 5 \
    --task bookMIA \
    --data_split train;

python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-3.5-turbo-1106 \
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
    --model gpt-3.5-turbo-1106 \
    --start_sentence 1 \
    --num_sentences 5 \
    --num_sequences 10 \
    --max_tokens 512 \
    --task_prompt_idx 1 2 4 5 \
    --task bookMIA \
    --data_split train;

python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-3.5-turbo-1106 \
    --start_sentence 1 \
    --num_sentences 5 \
    --num_sequences 10 \
    --max_tokens 512 \
    --task_prompt_idx 1 2 4 5 \
    --temperature 0.2 \
    --task bookMIA \
    --data_split train;


# Word level
python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-3.5-turbo-1106 \
    --num_sequences 20 \
    --temperature 1.0 \
    --task_prompt_idx 1 \
    --task bookMIA \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_words_from_end 50 \
    --data_split train;

# Word level
python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-3.5-turbo-1106 \
    --num_sequences 20 \
    --temperature 1.0 \
    --task_prompt_idx 1 \
    --task bookMIA \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_words_from_end 50 \
    --remove_bad_first \
    --data_split train;


python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-3.5-turbo-1106 \
    --num_sequences 20 \
    --temperature 1.0 \
    --task_prompt_idx 2 \
    --task bookMIA \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_words_from_end 50 \
    --remove_bad_first \
    --data_split train;

python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-3.5-turbo-1106 \
    --num_sequences 20 \
    --temperature 1.0 \
    --task_prompt_idx 4 \
    --task bookMIA \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_words_from_end 50 \
    --remove_bad_first \
    --data_split train;

python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-3.5-turbo-1106 \
    --num_sequences 20 \
    --temperature 1.0 \
    --task_prompt_idx 5 \
    --task bookMIA \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_words_from_end 50 \
    --remove_bad_first \
    --data_split train;

python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-3.5-turbo-1106 \
    --num_sequences 20 \
    --temperature 1.0 \
    --task_prompt_idx 5 \
    --task bookMIA \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_words_from_end 100 \
    --remove_bad_first \
    --data_split train;

python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-3.5-turbo-1106 \
    --num_sequences 20 \
    --temperature 1.0 \
    --task_prompt_idx 5 \
    --task bookMIA \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_words_from_end 200 \
    --remove_bad_first \
    --data_split train;

python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-3.5-turbo-1106 \
    --num_sequences 20 \
    --temperature 1.0 \
    --task_prompt_idx 5 \
    --task bookMIA \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_words_from_end 400 \
    --remove_bad_first \
    --data_split train;

python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-3.5-turbo-1106 \
    --num_sequences 20 \
    --temperature 1.0 \
    --task_prompt_idx 5 \
    --task bookMIA \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_words_from_end 600 \
    --remove_bad_first \
    --data_split train;

python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-3.5-turbo-1106 \
    --num_sequences 20 \
    --temperature 0.2 \
    --task_prompt_idx 5 \
    --task bookMIA \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_words_from_end 100 \
    --remove_bad_first \
    --data_split train;

python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-3.5-turbo-1106 \
    --num_sequences 10 \
    --temperature 1.0 \
    --task_prompt_idx 4 5 \
    --task bookMIA \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_words_from_end 100 \
    --remove_bad_first \
    --data_split train;

python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-3.5-turbo-1106 \
    --num_sequences 5 \
    --temperature 1.0 \
    --task_prompt_idx 1 2 4 5 \
    --task bookMIA \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_words_from_end 100 \
    --remove_bad_first \
    --data_split train;

# More iteration on 100
python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-3.5-turbo-1106 \
    --num_sequences 50 \
    --temperature 1.0 \
    --task_prompt_idx 5 \
    --task bookMIA \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_words_from_end 100 \
    --remove_bad_first \
    --data_split train;

# More iteration on 100
python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-3.5-turbo-1106 \
    --num_sequences 50 \
    --temperature 1.5 \
    --task_prompt_idx 5 \
    --task bookMIA \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_words_from_end 100 \
    --remove_bad_first \
    --data_split train;

# More iteration on 100
python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-3.5-turbo-1106 \
    --num_sequences 20 \
    --temperature 1.0 \
    --task_prompt_idx 5 \
    --task bookMIA \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_words_from_end 150 \
    --remove_bad_first \
    --data_split train;

# More iteration on 100 - 100 gens
python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-3.5-turbo-1106 \
    --num_sequences 100 \
    --temperature 1.0 \
    --task_prompt_idx 5 \
    --task bookMIA \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_words_from_end 100 \
    --remove_bad_first \
    --data_split train;