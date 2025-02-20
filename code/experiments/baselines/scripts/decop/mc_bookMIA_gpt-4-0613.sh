python3 -m code.experiments.baselines.decop_mc \
    --target_model gpt-4-0613 \
    --paraphrase_model gpt-4o-2024-11-20 \
    --key_name snippet \
    --task bookMIA \
    --split train \
    --closed_model \
    --keep_n_sentences 5 \
    --sys_prompt_idx 0;
    
python3 -m code.experiments.baselines.decop_mc \
    --target_model gpt-4-0613 \
    --paraphrase_model gpt-4o-2024-11-20 \
    --key_name snippet \
    --task bookMIA \
    --split val \
    --closed_model \
    --keep_n_sentences 5 \
    --sys_prompt_idx 0;
    
python3 -m code.experiments.baselines.decop_mc \
    --target_model gpt-4-0613 \
    --paraphrase_model gpt-4o-2024-11-20 \
    --key_name snippet \
    --task bookMIA \
    --split test \
    --closed_model \
    --keep_n_sentences 5 \
    --sys_prompt_idx 0;