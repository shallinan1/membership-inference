CUDA_VISIBLE_DEVICES=0 python3 -m code.experiments.baselines.compute_text_probs \
    --target_model huggyllama/llama-7b \
    --key_name input \
    --task wikiMIA_update \
    --split test;

python3 -m code.experiments.baselines.run_loss_baselines \
    --target_model_probs outputs/baselines/wikiMIA_update/test/probs/llama-7b.jsonl \
    --ref_model_probs outputs/baselines/wikiMIA_update/test/probs/llama-7b.jsonl \
    --key_name input;