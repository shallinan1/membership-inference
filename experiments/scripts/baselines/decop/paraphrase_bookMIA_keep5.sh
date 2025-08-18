python3 -m code.experiments.baselines.decop_paraphrase \
    --paraphrase_model gpt-4o-2024-11-20 \
    --task bookMIA \
    --split val \
    --key_name snippet \
    --closed_model \
    --temperature 0.1 \
    --top_p 1 \
    --keep_n_sentences 5 \
    --max_tokens 3072; # 3 paraphrases - 512 tokens at most each * 2 for verbosity buffer

python3 -m code.experiments.baselines.decop_paraphrase \
    --paraphrase_model gpt-4o-2024-11-20 \
    --task bookMIA \
    --split train \
    --key_name snippet \
    --closed_model \
    --temperature 0.1 \
    --top_p 1 \
    --keep_n_sentences 5 \
    --max_tokens 3072; # 3 paraphrases - 512 tokens at most each * 2 for verbosity buffer

python3 -m code.experiments.baselines.decop_paraphrase \
    --paraphrase_model gpt-4o-2024-11-20 \
    --task bookMIA \
    --split test \
    --key_name snippet \
    --closed_model \
    --temperature 0.1 \
    --top_p 1 \
    --keep_n_sentences 5 \
    --max_tokens 3072 # 3 paraphrases - 512 tokens at most each * 2 for verbosity buffer