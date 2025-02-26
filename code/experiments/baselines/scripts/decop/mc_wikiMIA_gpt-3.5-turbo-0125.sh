python3 -m code.experiments.baselines.decop_mc \
    --target_model gpt-3.5-turbo-0125 \
    --paraphrase_model gpt-4o-2024-11-20 \
    --key_name input \
    --task bookMIA \
    --split val \
    --closed_model \
    --sys_prompt_idx 0;

python3 -m code.experiments.baselines.decop_mc \
    --target_model gpt-3.5-turbo-0125 \
    --paraphrase_model gpt-4o-2024-11-20 \
    --key_name input \
    --task bookMIA \
    --split train \
    --closed_model \
    --sys_prompt_idx 0;
    
python3 -m code.experiments.baselines.decop_mc \
    --target_model gpt-3.5-turbo-0125 \
    --paraphrase_model gpt-4o-2024-11-20 \
    --key_name input \
    --task bookMIA \
    --split test \
    --closed_model \
    --sys_prompt_idx 0;