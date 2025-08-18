python3 -m code.experiments.baselines.decop_mc \
    --target_model /gscratch/xlab/hallisky/cache/tulu-30b-finalized \
    --paraphrase_model gpt-4o-2024-11-20 \
    --key_name snippet \
    --task tulu_v1 \
    --split train \
    --sys_prompt_idx 0;
    
python3 -m code.experiments.baselines.decop_mc \
    --target_model /gscratch/xlab/hallisky/cache/tulu-30b-finalized \
    --paraphrase_model gpt-4o-2024-11-20 \
    --key_name snippet \
    --task tulu_v1 \
    --split val \
    --sys_prompt_idx 0;
    
python3 -m code.experiments.baselines.decop_mc \
    --target_model /gscratch/xlab/hallisky/cache/tulu-30b-finalized \
    --paraphrase_model gpt-4o-2024-11-20 \
    --key_name snippet \
    --task tulu_v1 \
    --split test \
    --sys_prompt_idx 0;