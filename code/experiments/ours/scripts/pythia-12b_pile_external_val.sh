CUDA_VISIBLE_DEVICES=0,1 python3 -m code.experiments.ours.generate \
    --model EleutherAI/pythia-12b \
    --start_sentence 0 \
    --num_sentences 10 \
    --num_sequences 20 \
    --max_tokens 512 \
    --task_prompt_idx 1 \
    --task pile_external \
    --data_split val;

CUDA_VISIBLE_DEVICES=0,1 python3 -m code.experiments.ours.generate \
    --model EleutherAI/pythia-12b \
    --start_sentence 0 \
    --num_sentences 5 \
    --num_sequences 20 \
    --max_tokens 512 \
    --task_prompt_idx 1 \
    --task pile_external \
    --data_split val;

CUDA_VISIBLE_DEVICES=0,1 python3 -m code.experiments.ours.generate \
    --model EleutherAI/pythia-12b \
    --start_sentence 0 \
    --num_sentences 5 \
    --num_sequences 20 \
    --max_tokens 512 \
    --task_prompt_idx 0 \
    --task pile_external \
    --data_split val;