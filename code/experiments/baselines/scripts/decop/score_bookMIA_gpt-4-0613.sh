python3 -m code.experiments.baselines.decop_results \
    --paraphrase_model gpt-4o-2024-11-20 \
    --task bookMIA \
    --split train \
    --model gpt-4-0613\

python3 -m code.experiments.baselines.decop_results \
    --paraphrase_model gpt-4o-2024-11-20 \
    --task bookMIA \
    --split val \
    --model gpt-4-0613

python3 -m code.experiments.baselines.decop_results \
    --paraphrase_model gpt-4o-2024-11-20 \
    --task bookMIA \
    --split test \
    --model gpt-4-0613
