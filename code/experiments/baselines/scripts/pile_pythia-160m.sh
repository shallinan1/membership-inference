CUDA_VISIBLE_DEVICES=0 python3 -m code.experiments.baselines.compute_text_probs \
    --target_model EleutherAI/pythia-160m \
    --key_name snippet \
    --task pile_external \
    --split train;
