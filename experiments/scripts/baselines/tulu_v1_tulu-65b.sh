python3 -m code.experiments.baselines.compute_text_probs \
    --target_model /gscratch/xlab/hallisky/cache/tulu-65b-finalized \
    --key_name snippet \
    --task tulu_v1 \
    --split train;

python3 -m code.experiments.baselines.compute_text_probs \
    --target_model /gscratch/xlab/hallisky/cache/tulu-65b-finalized \
    --key_name snippet \
    --task tulu_v1 \
    --split val;

python3 -m code.experiments.baselines.compute_text_probs \
    --target_model /gscratch/xlab/hallisky/cache/tulu-65b-finalized \
    --key_name snippet \
    --task tulu_v1 \
    --split test;