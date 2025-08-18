python3 -m code.experiments.baselines.decop_results \
    --paraphrase_model gpt-4o-2024-11-20 \
    --task wikiMIA \
    --split val \
    --model gpt-3.5-turbo-instruct;

python3 -m code.experiments.baselines.decop_results \
    --paraphrase_model gpt-4o-2024-11-20 \
    --task wikiMIA \
    --split test \
    --model gpt-3.5-turbo-instruct;

python3 -m code.experiments.baselines.decop_results \
    --paraphrase_model gpt-4o-2024-11-20 \
    --task wikiMIA \
    --split train \
    --model gpt-3.5-turbo-instruct;
