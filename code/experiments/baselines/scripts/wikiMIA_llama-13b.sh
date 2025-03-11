CUDA_VISIBLE_DEVICES=0 python3 -m code.experiments.baselines.compute_text_probs \
    --target_model huggyllama/llama-13b \
    --key_name input \
    --task wikiMIA \
    --split test;

python3 -m code.experiments.baselines.run_loss_baselines \
    --target_model_probs outputs/baselines/wikiMIA/test/probs/llama-13b.jsonl \
    --ref_model_probs outputs/baselines/wikiMIA/test/probs/llama-7b.jsonl \
    --key_name input;