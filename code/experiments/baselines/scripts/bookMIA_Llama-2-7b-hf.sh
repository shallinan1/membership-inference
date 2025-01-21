CUDA_VISIBLE_DEVICES=0 python3 -m code.experiments.baselines.compute_text_probs \
    --target_model meta-llama/Llama-2-7b-hf \
    --key_name snippet \
    --task bookMIA \
    --split train;