python3 -m code.experiments.baselines.compute_text_probs \
    --target_model stabilityai/stablelm-base-alpha-3b-v2 \
    --key_name snippet \
    --task pile_external \
    --split train;