CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m code.experiments.baselines.compute_text_probs \
    --target_model huggyllama/llama-65b \
    --key_name input \
    --task wikiMIA_update \
    --split test;

python3 -m code.experiments.baselines.run_loss_baselines \
    --target_model_probs outputs/baselines/wikiMIA_update/test/probs/llama-65b.jsonl \
    --ref_model_probs outputs/baselines/wikiMIA_update/test/probs/llama-7b.jsonl \
    --key_name input;