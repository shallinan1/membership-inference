# New from here
python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-3.5-turbo-1106 \
    --num_sequences 50 \
    --temperature 1.0 \
    --task_prompt_idx 0 \
    --task wikiMIA_update \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_proportion_from_end 0.5 \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-3.5-turbo-1106 \
    --num_sequences 50 \
    --temperature 1.0 \
    --task_prompt_idx 0 \
    --task wikiMIA_update \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_proportion_from_end 0.75 \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-3.5-turbo-1106 \
    --num_sequences 50 \
    --temperature 1.0 \
    --task_prompt_idx 1 \
    --task wikiMIA_update \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_proportion_from_end 0.5 \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-3.5-turbo-1106 \
    --num_sequences 50 \
    --temperature 1.0 \
    --task_prompt_idx 1 \
    --task wikiMIA_update \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_proportion_from_end 0.75 \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-3.5-turbo-1106 \
    --num_sequences 50 \
    --temperature 1.0 \
    --task_prompt_idx 2 \
    --task wikiMIA_update \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_proportion_from_end 0.5 \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --openai \
    --model gpt-3.5-turbo-1106 \
    --num_sequences 50 \
    --temperature 1.0 \
    --task_prompt_idx 2 \
    --task wikiMIA_update \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_proportion_from_end 0.75 \
    --data_split test;

