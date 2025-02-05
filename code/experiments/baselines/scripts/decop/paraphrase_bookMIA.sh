python3 -m code.experiments.baselines.decop_paraphrase \ 
    --paraphrase_model gpt-4o-2024-11-20 \
    --task bookMIA \
    --split val \
    --closed_model \
    --temperature 0.1 \
    --top_p 1 \
    --n 1 \
    --max_tokens 256