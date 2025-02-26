python3 -m code.experiments.baselines.decop_paraphrase \
    --paraphrase_model gpt-4o-2024-11-20 \
    --task wikiMIA \
    --split val \
    --key_name input \
    --closed_model \
    --temperature 0.1 \
    --top_p 1 \
    --max_tokens 3072; # 3 paraphrases - 512 tokens at most each * 2 for verbosity buffer

python3 -m code.experiments.baselines.decop_paraphrase \
    --paraphrase_model gpt-4o-2024-11-20 \
    --task wikiMIA \
    --split train \
    --key_name input \
    --closed_model \
    --temperature 0.1 \
    --top_p 1 \
    --max_tokens 3072; # 3 paraphrases - 512 tokens at most each * 2 for verbosity buffer

python3 -m code.experiments.baselines.decop_paraphrase \
    --paraphrase_model gpt-4o-2024-11-20 \
    --task wikiMIA \
    --split test \
    --key_name input \
    --closed_model \
    --temperature 0.1 \
    --top_p 1 \
    --max_tokens 3072 # 3 paraphrases - 512 tokens at most each * 2 for verbosity buffer