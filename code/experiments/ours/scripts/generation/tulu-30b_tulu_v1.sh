python3 -m code.experiments.ours.generate \
    --model /gscratch/xlab/hallisky/cache/tulu-30b-finalized \
    --num_sequences 20 \
    --max_tokens 512 \
    --task_prompt_idx 0 \
    --task tulu_v1 \
    --data_split train;

python3 -m code.experiments.ours.generate \
    --model /gscratch/xlab/hallisky/cache/tulu-30b-finalized \
    --num_sequences 20 \
    --max_tokens 512 \
    --task_prompt_idx 0 \
    --task tulu_v1 \
    --data_split val;

python3 -m code.experiments.ours.generate \
    --model /gscratch/xlab/hallisky/cache/tulu-30b-finalized \
    --num_sequences 20 \
    --max_tokens 512 \
    --task_prompt_idx 0 \
    --task tulu_v1 \
    --data_split test;