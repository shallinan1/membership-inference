python3 -m src.attacks.ngram_coverage_attack.generate \
    --openai \
    --model gpt-4-0613 \
    --start_sentence 0 \
    --num_sentences 1 \
    --num_sequences 20 \
    --max_tokens 100 \
    --temperature 0.2 \
    --task_prompt_idx 5 \
    --task wikiMIA \
    --key_name input \
    --data_split train;

python3 -m src.attacks.ngram_coverage_attack.generate \
    --openai \
    --model gpt-4-0613 \
    --start_sentence 0 \
    --num_sentences 1 \
    --num_sequences 20 \
    --max_tokens 100 \
    --temperature 0.2 \
    --task_prompt_idx 5 \
    --task wikiMIA \
    --key_name input \
    --data_split val;
    
python3 -m src.attacks.ngram_coverage_attack.generate \
    --openai \
    --model gpt-4-0613 \
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
    --model gpt-4-0613 \
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
    --model gpt-4-0613 \
    --start_sentence 0 \
    --num_sentences 1 \
    --num_sequences 20 \
    --temperature 0.2 \
    --max_tokens 100 \
    --task_prompt_idx 2 \
    --task wikiMIA \
    --key_name input \
    --data_split test;


python3 -m src.attacks.ngram_coverage_attack.generate \
    --openai \
    --model gpt-4-0613 \
    --start_sentence 0 \
    --num_sentences 1 \
    --num_sequences 20 \
    --temperature 0.2 \
    --max_tokens 25 \
    --task_prompt_idx 0 \
    --task wikiMIA \
    --key_name input \
    --data_split test;

python3 -m src.attacks.ngram_coverage_attack.generate \
    --openai \
    --model gpt-4-0613 \
    --start_sentence 0 \
    --num_sentences 1 \
    --num_sequences 20 \
    --temperature 0.2 \
    --max_tokens 25 \
    --task_prompt_idx 1 \
    --task wikiMIA \
    --key_name input \
    --data_split test;


python3 -m src.attacks.ngram_coverage_attack.generate \
    --openai \
    --model gpt-4-0613 \
    --start_sentence 0 \
    --num_sentences 1 \
    --num_sequences 20 \
    --temperature 0.2 \
    --max_tokens 100 \
    --task_prompt_idx 2 \
    --task wikiMIA \
    --key_name input \
    --data_split test;
