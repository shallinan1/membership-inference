python3 -m code.experiments.baselines.decop_mc \
    --target_model huggyllama/llama-65b \
    --paraphrase_model gpt-4o-2024-11-20 \
    --key_name input \
    --task wikiMIA \
    --split val \
    --closed_model \
    --sys_prompt_idx 0;

python3 -m code.experiments.baselines.decop_mc \
    --target_model huggyllama/llama-65b \
    --paraphrase_model gpt-4o-2024-11-20 \
    --key_name input \
    --task wikiMIA \
    --split train \
    --closed_model \
    --sys_prompt_idx 0;
    
python3 -m code.experiments.baselines.decop_mc \
    --target_model huggyllama/llama-65b \
    --paraphrase_model gpt-4o-2024-11-20 \
    --key_name input \
    --task wikiMIA \
    --split test \
    --sys_prompt_idx 0;