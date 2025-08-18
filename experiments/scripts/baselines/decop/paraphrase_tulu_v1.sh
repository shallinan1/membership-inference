python3 -m code.experiments.baselines.decop_paraphrase \
    --paraphrase_model gpt-4o-2024-11-20 \
    --task tulu_v1 \
    --split train \
    --key_name snippet \
    --closed_model \
    --temperature 0.1 \
    --top_p 1 \
    --max_tokens 2048;

python3 -m code.experiments.baselines.decop_paraphrase \
    --paraphrase_model gpt-4o-2024-11-20 \
    --task tulu_v1 \
    --split val \
    --key_name snippet \
    --closed_model \
    --temperature 0.1 \
    --top_p 1 \
    --max_tokens 2048;

python3 -m code.experiments.baselines.decop_paraphrase \
    --paraphrase_model gpt-4o-2024-11-20 \
    --task tulu_v1 \
    --split test \
    --key_name snippet \
    --closed_model \
    --temperature 0.1 \
    --top_p 1 \
    --max_tokens 2048
