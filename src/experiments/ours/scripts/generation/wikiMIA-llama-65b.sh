python3 -m code.experiments.ours.generate \
    --model huggyllama/llama-65b \
    --start_sentence 0 \
    --num_sentences 1 \
    --num_sequences 20 \
    --temperature 0.2 \
    --max_tokens 100 \
    --task_prompt_idx 0 \
    --task wikiMIA \
    --key_name input \
    --data_split test;

# Use words instead
python3 -m code.experiments.ours.generate \
    --model huggyllama/llama-65b \
    --num_sequences 20 \
    --temperature 1.0 \
    --task_prompt_idx 0 \
    --task wikiMIA \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_words_from_end 10 \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --model huggyllama/llama-65b \
    --num_sequences 20 \
    --temperature 1.0 \
    --task_prompt_idx 1 \
    --task wikiMIA \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_words_from_end 10 \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --model huggyllama/llama-65b \
    --num_sequences 20 \
    --temperature 1.0 \
    --task_prompt_idx 2 \
    --task wikiMIA \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_words_from_end 10 \
    --data_split test;

# More variants
python3 -m code.experiments.ours.generate \
    --model huggyllama/llama-65b \
    --num_sequences 20 \
    --temperature 0.2 \
    --task_prompt_idx 2 \
    --task wikiMIA \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_words_from_end 10 \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --model huggyllama/llama-65b \
    --num_sequences 20 \
    --temperature 1.0 \
    --task_prompt_idx 2 \
    --task wikiMIA \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_tokens 100 \
    --num_words_from_end 10 \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --model huggyllama/llama-65b \
    --num_sequences 50 \
    --temperature 1.0 \
    --task_prompt_idx 2 \
    --task wikiMIA \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_words_from_end 10 \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --model huggyllama/llama-65b \
    --num_sequences 20 \
    --temperature 1.0 \
    --task_prompt_idx 2 \
    --task wikiMIA \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_words_from_end 25 \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --model huggyllama/llama-65b \
    --num_sequences 20 \
    --temperature 1.0 \
    --task_prompt_idx 2 \
    --task wikiMIA \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_words_from_end 50 \
    --data_split test;


python3 -m code.experiments.ours.generate \
    --model huggyllama/llama-65b \
    --num_sequences 50 \
    --temperature 1.0 \
    --task_prompt_idx 2 \
    --task wikiMIA \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_words_from_end 50 \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --model huggyllama/llama-65b \
    --num_sequences 20 \
    --temperature 1.0 \
    --task_prompt_idx 2 \
    --task wikiMIA \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_tokens 100 \
    --num_words_from_end 50 \
    --data_split test;



python3 -m code.experiments.ours.generate \
    --model huggyllama/llama-65b \
    --num_sequences 20 \
    --temperature 1.0 \
    --task_prompt_idx 2 \
    --task wikiMIA \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_proportion_from_end 0.25 \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --model huggyllama/llama-65b \
    --num_sequences 20 \
    --temperature 1.0 \
    --task_prompt_idx 2 \
    --task wikiMIA \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_proportion_from_end 0.5 \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --model huggyllama/llama-65b \
    --num_sequences 50 \
    --temperature 1.0 \
    --task_prompt_idx 2 \
    --task wikiMIA \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_proportion_from_end 0.5 \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --model huggyllama/llama-65b \
    --num_sequences 50 \
    --temperature 1.0 \
    --task_prompt_idx 0 \
    --task wikiMIA \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_proportion_from_end 0.5 \
    --data_split test;


# Stop here

python3 -m code.experiments.ours.generate \
    --model huggyllama/llama-65b \
    --num_sequences 50 \
    --temperature 1.0 \
    --task_prompt_idx 0 \
    --task wikiMIA \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_tokens 50 \
    --num_proportion_from_end 0.5 \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --model huggyllama/llama-65b \
    --num_sequences 50 \
    --temperature 1.0 \
    --task_prompt_idx 0 \
    --task wikiMIA \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_tokens 100 \
    --num_proportion_from_end 0.5 \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --model huggyllama/llama-65b \
    --num_sequences 50 \
    --temperature 1.0 \
    --task_prompt_idx 0 \
    --task wikiMIA \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_tokens 200 \
    --num_proportion_from_end 0.5 \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --model huggyllama/llama-65b \
    --num_sequences 50 \
    --temperature 1.0 \
    --task_prompt_idx 0 \
    --task wikiMIA \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_tokens 400 \
    --num_proportion_from_end 0.5 \
    --data_split test;


python3 -m code.experiments.ours.generate \
    --model huggyllama/llama-65b \
    --num_sequences 50 \
    --temperature 1.0 \
    --task_prompt_idx 2 \
    --task wikiMIA \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_tokens 50 \
    --num_proportion_from_end 0.5 \
    --data_split test;


python3 -m code.experiments.ours.generate \
    --model huggyllama/llama-65b \
    --num_sequences 50 \
    --temperature 1.0 \
    --task_prompt_idx 2 \
    --task wikiMIA \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_tokens 100 \
    --num_proportion_from_end 0.5 \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --model huggyllama/llama-65b \
    --num_sequences 50 \
    --temperature 1.0 \
    --task_prompt_idx 2 \
    --task wikiMIA \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_tokens 200 \
    --num_proportion_from_end 0.5 \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --model huggyllama/llama-65b \
    --num_sequences 50 \
    --temperature 1.0 \
    --task_prompt_idx 2 \
    --task wikiMIA \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_tokens 400 \
    --num_proportion_from_end 0.5 \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --model huggyllama/llama-65b \
    --num_sequences 50 \
    --temperature 1.0 \
    --task_prompt_idx 2 \
    --task wikiMIA \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_tokens 50 \
    --num_proportion_from_end 0.9 \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --model huggyllama/llama-65b \
    --num_sequences 50 \
    --temperature 1.0 \
    --task_prompt_idx 2 \
    --task wikiMIA \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_tokens 200 \
    --num_proportion_from_end 0.1 \
    --data_split test;