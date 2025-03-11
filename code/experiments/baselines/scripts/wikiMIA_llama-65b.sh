python3 -m code.experiments.baselines.compute_text_probs \
    --target_model huggyllama/llama-65b \
    --key_name input \
    --task wikiMIA \
    --split test;

python3 -m code.experiments.baselines.compute_text_probs \
    --target_model huggyllama/llama-65b \
    --key_name input \
    --task wikiMIA \
    --split test;