python3 -m code.experiments.baselines.decop_results \
    --paraphrase_model gpt-4o-2024-11-20 \
    --task pile_external \
    --split train \
    --model pythia-2.8b\

    python3 -m code.experiments.baselines.decop_results \
    --paraphrase_model gpt-4o-2024-11-20 \
    --task pile_external \
    --split val \
    --model pythia-2.8b

    python3 -m code.experiments.baselines.decop_results \
    --paraphrase_model gpt-4o-2024-11-20 \
    --task pile_external \
    --split test \
    --model pythia-2.8b
