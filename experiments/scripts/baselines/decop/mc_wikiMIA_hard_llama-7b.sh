python3 -m code.experiments.baselines.decop_mc \
    --target_model huggyllama/llama-30b \
    --paraphrase_model gpt-4o-2024-11-20 \
    --key_name input \
    --task wikiMIA_hard \
    --split test \
    --sys_prompt_idx 0;