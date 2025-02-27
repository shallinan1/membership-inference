python3 -m code.experiments.baselines.decop_mc \
    --target_model gpt-3.5-turbo-instruct \
    --paraphrase_model gpt-4o-2024-11-20 \
    --key_name input \
    --task wikiMIA \
    --split val \
    --closed_model \
    --sys_prompt_idx 0;

python3 -m code.experiments.baselines.decop_mc \
    --target_model gpt-3.5-turbo-instruct \
    --paraphrase_model gpt-4o-2024-11-20 \
    --key_name input \
    --task wikiMIA \
    --split train \
    --closed_model \
    --sys_prompt_idx 0;
    
python3 -m code.experiments.baselines.decop_mc \
    --target_model gpt-3.5-turbo-instruct \
    --paraphrase_model gpt-4o-2024-11-20 \
    --key_name input \
    --task wikiMIA \
    --split test \
    --closed_model \
    --sys_prompt_idx 0;