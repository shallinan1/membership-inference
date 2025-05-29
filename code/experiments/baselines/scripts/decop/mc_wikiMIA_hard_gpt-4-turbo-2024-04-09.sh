python3 -m code.experiments.baselines.decop_mc \
    --target_model gpt-4-turbo-2024-04-09 \
    --paraphrase_model gpt-4o-2024-11-20 \
    --key_name input \
    --task wikiMIA_hard \
    --split test \
    --closed_model \
    --sys_prompt_idx 0;