python3 -m code.experiments.baselines.decop_mc \
    --target_model EleutherAI/pythia-6.9b \
    --paraphrase_model gpt-4o-2024-11-20 \
    --key_name snippet \
    --task pile_external \
    --split train \
    --sys_prompt_idx 0;
    
python3 -m code.experiments.baselines.decop_mc \
    --target_model EleutherAI/pythia-6.9b \
    --paraphrase_model gpt-4o-2024-11-20 \
    --key_name snippet \
    --task pile_external \
    --split val \
    --sys_prompt_idx 0;
    
python3 -m code.experiments.baselines.decop_mc \
    --target_model EleutherAI/pythia-6.9b \
    --paraphrase_model gpt-4o-2024-11-20 \
    --key_name snippet \
    --task pile_external \
    --split test \
    --sys_prompt_idx 0;