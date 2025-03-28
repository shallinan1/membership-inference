
# To run
python3 -m code.experiments.ours.generate \
    --model huggyllama/llama-7b \
    --num_sequences 50 \
    --temperature 1.0 \
    --task_prompt_idx 2 \
    --task wikiMIA_hard \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_proportion_from_end 0.5 \
    --data_split test;
    
python3 -m code.experiments.ours.generate \
    --model huggyllama/llama-7b \
    --num_sequences 50 \
    --temperature 1.0 \
    --task_prompt_idx 2 \
    --task wikiMIA_hard \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_tokens 100 \
    --num_proportion_from_end 0.5 \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --model huggyllama/llama-7b \
    --num_sequences 50 \
    --temperature 1.0 \
    --task_prompt_idx 2 \
    --task wikiMIA_hard \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_tokens 200 \
    --num_proportion_from_end 0.5 \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --model huggyllama/llama-7b \
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
    --model huggyllama/llama-7b \
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
    --model huggyllama/llama-7b \
    --num_sequences 50 \
    --temperature 1.0 \
    --task_prompt_idx 2 \
    --task wikiMIA_hard \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_tokens 100 \
    --num_proportion_from_end 0.5 \
    --data_split test;