CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m code.experiments.baselines.decop_mc \
    --target_model huggyllama/llama-65b \
    --paraphrase_model gpt-4o-2024-11-20 \
    --key_name input \
    --task wikiMIA_update \
    --split test \
    --sys_prompt_idx 0;