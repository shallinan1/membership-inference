
# Less words of context - more generation
python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-4o-mini-2024-07-18 \
    --num_sequences 20 \
    --temperature 1.0 \
    --task_prompt_idx 2 \
    --task wikiMIA_hard \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_tokens 128 \
    --num_words_from_end 50 \
    --data_split test;

# Less words of context - higher tempreature
python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-4o-mini-2024-07-18 \
    --num_sequences 20 \
    --temperature 1.0 \
    --task_prompt_idx 2 \
    --task wikiMIA_hard \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_words_from_end 50 \
    --data_split test;

# Commands to run
python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-4o-mini-2024-07-18 \
    --num_sequences 20 \
    --temperature 1.0 \
    --task_prompt_idx 0 \
    --task wikiMIA_hard \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_proportion_from_end 0.5 \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-4o-mini-2024-07-18 \
    --num_sequences 20 \
    --temperature 1.0 \
    --task_prompt_idx 1 \
    --task wikiMIA_hard \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_proportion_from_end 0.5 \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-4o-mini-2024-07-18 \
    --num_sequences 20 \
    --temperature 1.0 \
    --task_prompt_idx 2 \
    --task wikiMIA_hard \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_proportion_from_end 0.5 \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-4o-mini-2024-07-18 \
    --num_sequences 50 \
    --temperature 1.0 \
    --task_prompt_idx 0 \
    --task wikiMIA_hard \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_proportion_from_end 0.5 \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-4o-mini-2024-07-18 \
    --num_sequences 50 \
    --temperature 1.0 \
    --task_prompt_idx 2 \
    --task wikiMIA_hard \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_proportion_from_end 0.5 \
    --data_split test;

# New from here
python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-4o-mini-2024-07-18 \
    --num_sequences 50 \
    --temperature 1.0 \
    --task_prompt_idx 1 \
    --task wikiMIA_hard \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_proportion_from_end 0.5 \
    --data_split test;


python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-4o-mini-2024-07-18 \
    --num_sequences 50 \
    --temperature 1.0 \
    --task_prompt_idx 3 \
    --task wikiMIA_hard \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_proportion_from_end 0.5 \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-4o-mini-2024-07-18 \
    --num_sequences 50 \
    --temperature 1.0 \
    --task_prompt_idx 2 \
    --task wikiMIA_hard \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_proportion_from_end 0.75 \
    --data_split test;


python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-4o-mini-2024-07-18 \
    --num_sequences 50 \
    --temperature 1.0 \
    --task_prompt_idx 0 \
    --task wikiMIA_hard \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_proportion_from_end 0.75 \
    --data_split test;

