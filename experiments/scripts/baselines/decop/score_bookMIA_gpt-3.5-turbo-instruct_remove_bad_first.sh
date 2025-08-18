python3 -m code.experiments.baselines.decop_results \
    --paraphrase_model gpt-4o-2024-11-20 \
    --task bookMIA \
    --split val \
    --remove_bad_first \
    --model gpt-3.5-turbo-instruct;

python3 -m code.experiments.baselines.decop_results \
    --paraphrase_model gpt-4o-2024-11-20 \
    --task bookMIA \
    --split test \
    --remove_bad_first \
    --model gpt-3.5-turbo-instruct;

python3 -m code.experiments.baselines.decop_results \
    --paraphrase_model gpt-4o-2024-11-20 \
    --task bookMIA \
    --split train \
    --remove_bad_first \
    --model gpt-3.5-turbo-instruct;