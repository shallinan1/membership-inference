CUDA_VISIBLE_DEVICES=0 python3 -m code.experiments.baselines.compute_text_probs \
    --target_model openai-community/gpt2-xl \
    --key_name snippet \
    --task bookMIA \
    --split train;