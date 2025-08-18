CUDA_VISIBLE_DEVICES=0 python3 -m code.experiments.baselines.compute_text_probs \
    --target_model allenai/OLMo-7B-0724-Instruct-hf \
    --key_name snippet \
    --task dolma_v17 \
    --split test;