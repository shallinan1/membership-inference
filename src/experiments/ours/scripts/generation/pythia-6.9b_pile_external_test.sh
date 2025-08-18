python3 -m code.experiments.ours.generate \
    --model EleutherAI/pythia-6.9b \
    --start_sentence 0 \
    --num_sentences 10 \
    --num_sequences 20 \
    --max_tokens 512 \
    --task_prompt_idx 1 \
    --task pile_external \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --model EleutherAI/pythia-6.9b \
    --start_sentence 0 \
    --num_sentences 5 \
    --num_sequences 20 \
    --max_tokens 512 \
    --task_prompt_idx 1 \
    --task pile_external \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --model EleutherAI/pythia-6.9b \
    --start_sentence 0 \
    --num_sentences 5 \
    --num_sequences 20 \
    --max_tokens 512 \
    --task_prompt_idx 0 \
    --task pile_external \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --model EleutherAI/pythia-6.9b \
    --start_sentence 0 \
    --num_sentences 5 \
    --num_sequences 20 \
    --max_tokens 64 \
    --task_prompt_idx 0 \
    --task pile_external \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --model EleutherAI/pythia-6.9b \
    --start_sentence 0 \
    --num_sentences 5 \
    --num_sequences 20 \
    --max_tokens 64 \
    --task_prompt_idx 1 \
    --task pile_external \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --model EleutherAI/pythia-6.9b \
    --start_sentence 0 \
    --num_sentences 5 \
    --num_sequences 20 \
    --max_tokens 64 \
    --task_prompt_idx 2 \
    --task pile_external \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --model EleutherAI/pythia-6.9b \
    --start_sentence 0 \
    --num_sentences 5 \
    --num_sequences 10 \
    --max_tokens 64 \
    --task_prompt_idx 0 1 2 \
    --task pile_external \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --model EleutherAI/pythia-6.9b \
    --num_sequences 20 \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_proportion_from_end 0.5 \
    --task_prompt_idx 0  \
    --task pile_external \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --model EleutherAI/pythia-6.9b \
    --num_sequences 50 \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_proportion_from_end 0.5 \
    --task_prompt_idx 0  \
    --task pile_external \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --model EleutherAI/pythia-6.9b \
    --num_sequences 20 \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_proportion_from_end 0.75 \
    --task_prompt_idx 0  \
    --task pile_external \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --model EleutherAI/pythia-6.9b \
    --num_sequences 50 \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_proportion_from_end 0.75 \
    --task_prompt_idx 0  \
    --task pile_external \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --model EleutherAI/pythia-6.9b \
    --num_sequences 50 \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_proportion_from_end 0.5 \
    --task_prompt_idx 1  \
    --task pile_external \
    --data_split test;


python3 -m code.experiments.ours.generate \
    --model EleutherAI/pythia-6.9b \
    --num_sequences 50 \
    --prompt_with_words_not_sent \
    --max_length_to_sequence_length \
    --num_proportion_from_end 0.25 \
    --task_prompt_idx 0  \
    --task pile_external \
    --data_split test;




