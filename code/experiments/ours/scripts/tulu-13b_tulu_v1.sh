CUDA_VISIBLE_DEVICES=0,1 python3 -m code.experiments.ours.generate \
    --model /gscratch/xlab/hallisky/cache/tulu-13b-finalized \
    --num_sequences 20 \
    --max_tokens 512 \
    --task_prompt_idx 0 \
    --task tulu_v1 \
    --data_split train;

CUDA_VISIBLE_DEVICES=0,1 python3 -m code.experiments.ours.generate \
    --model /gscratch/xlab/hallisky/cache/tulu-13b-finalized \
    --num_sequences 20 \
    --max_tokens 512 \
    --task_prompt_idx 0 \
    --task tulu_v1 \
    --data_split val;

CUDA_VISIBLE_DEVICES=0,1 python3 -m code.experiments.ours.generate \
    --model /gscratch/xlab/hallisky/cache/tulu-13b-finalized \
    --num_sequences 20 \
    --max_tokens 512 \
    --task_prompt_idx 0 \
    --task tulu_v1 \
    --data_split test;

python3 -m code.experiments.ours.generate \
    --model /gscratch/xlab/hallisky/cache/tulu-13b-finalized \
    --num_sequences 20 \
    --max_tokens 512 \
    --task_prompt_idx 0 \
    --task tulu_v1 \
    --temperature 0.2 \
    --data_split train;

python3 -m code.experiments.ours.generate \
    --model /gscratch/xlab/hallisky/cache/tulu-13b-finalized \
    --num_sequences 20 \
    --max_tokens 512 \
    --task_prompt_idx 0 \
    --task tulu_v1 \
    --temperature 0.2 \
    --data_split val;

python3 -m code.experiments.ours.generate \
    --model /gscratch/xlab/hallisky/cache/tulu-13b-finalized \
    --num_sequences 20 \
    --max_tokens 512 \
    --task_prompt_idx 0 \
    --task tulu_v1 \
    --temperature 0.2 \
    --data_split test;