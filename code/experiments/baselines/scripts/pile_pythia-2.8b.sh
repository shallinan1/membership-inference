CUDA_VISIBLE_DEVICES=0 python3 -m code.experiments.baselines.compute_text_probs \
    --target_model EleutherAI/pythia-2.8b \
    --key_name snippet \
    --task pile_external \
    --split train;

CUDA_VISIBLE_DEVICES=0 python3 -m code.experiments.baselines.compute_text_probs \
    --target_model EleutherAI/pythia-2.8b \
    --key_name snippet \
    --task pile_external \
    --split val;

CUDA_VISIBLE_DEVICES=0 python3 -m code.experiments.baselines.compute_text_probs \
    --target_model EleutherAI/pythia-2.8b \
    --key_name snippet \
    --task pile_external \
    --split test;
