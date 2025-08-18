python3 -m src.attacks.ngram_coverage_attack.generate \
    --openai \
    --model gpt-3.5-turbo-instruct \
    --start_sentence 0 \
    --num_sentences 1 \
    --num_sequences 20 \
    --max_tokens 100 \
    --temperature 0.2 \
    --task_prompt_idx 0 \
    --task wikiMIA \
    --key_name input \
    --data_split train;

python3 -m src.attacks.ngram_coverage_attack.generate \
    --openai \
    --model gpt-3.5-turbo-instruct \
    --start_sentence 0 \
    --num_sentences 1 \
    --num_sequences 20 \
    --max_tokens 100 \
    --temperature 0.2 \
    --task_prompt_idx 0 \
    --task wikiMIA \
    --key_name input \
    --data_split val;
    
python3 -m src.attacks.ngram_coverage_attack.generate \
    --openai \
    --model gpt-3.5-turbo-instruct \
    --start_sentence 0 \
    --num_sentences 1 \
    --num_sequences 20 \
    --temperature 0.2 \
    --max_tokens 100 \
    --task_prompt_idx 0 \
    --task wikiMIA \
    --key_name input \
    --data_split test;

python3 -m src.attacks.ngram_coverage_attack.generate \
    --openai \
    --model gpt-3.5-turbo-instruct \
    --start_sentence 0 \
    --num_sentences 1 \
    --num_sequences 20 \
    --temperature 0.2 \
    --max_tokens 100 \
    --task_prompt_idx 1 \
    --task wikiMIA \
    --key_name input \
    --data_split test;


python3 -m src.attacks.ngram_coverage_attack.generate \
    --openai \
    --model gpt-3.5-turbo-instruct \
    --start_sentence 0 \
    --num_sentences 1 \
    --num_sequences 20 \
    --temperature 0.2 \
    --max_tokens 100 \
    --task_prompt_idx 2 \
    --task wikiMIA \
    --key_name input \
    --data_split test;

# Using context

python3 -m src.attacks.ngram_coverage_attack.generate \
    --openai \
    --model gpt-3.5-turbo-instruct \
    --num_sequences 20 \
    --temperature 0.2 \
    --task_prompt_idx 2 \
    --task wikiMIA \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_words_from_end 10 \
    --data_split test;

# Less words of context
python3 -m src.attacks.ngram_coverage_attack.generate \
    --openai \
    --model gpt-3.5-turbo-instruct \
    --num_sequences 20 \
    --temperature 0.2 \
    --task_prompt_idx 2 \
    --task wikiMIA \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_words_from_end 50 \
    --data_split test;

# Less words of context - more generation
python3 -m src.attacks.ngram_coverage_attack.generate \
    --openai \
    --model gpt-3.5-turbo-instruct \
    --num_sequences 20 \
    --temperature 0.2 \
    --task_prompt_idx 2 \
    --task wikiMIA \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_tokens 128 \
    --num_words_from_end 50 \
    --data_split test;

# Less words of context - higher tempreature
python3 -m src.attacks.ngram_coverage_attack.generate \
    --openai \
    --model gpt-3.5-turbo-instruct \
    --num_sequences 20 \
    --temperature 1.0 \
    --task_prompt_idx 2 \
    --task wikiMIA \
    --key_name input \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_words_from_end 50 \
    --data_split test;