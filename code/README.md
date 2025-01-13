Pipeline to use [ours] method

1. Generate

    ```
    python3 -m code.experiments.ours.generate \
        --openai \
        --model gpt-3.5-turbo-0125 \
        --start_sentence 1 \
        --num_sentences 5 \
        --num_sequences 20 \
        --max_tokens 512 \
        --task_prompt_idx 5 \
        --task bookMIA \
        --data_split train;
    ```

Generations will be stored in the `code/experiments/ours/[dataset]/generations/[split]` folder

2. Score

    With example commands to run:

    ```
    python3 -m code.helper.dj_search.dj_search_exact_LLM \
        --task bookMIA \
        --output_dir /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/coverages/ \
        --gen_data /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/generations/gpt-3.5-turbo-0125_maxTokens512_numSeq10_topP0.95_numSent3_startSent1_promptIdx5_len788.jsonl \
        --min_ngram 4 \
        --source_docs swj0419/BookMIA \
        --parallel;

    python3 -m code.helper.dj_search.dj_search_exact_LLM \
        --task bookMIA \
        --output_dir /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/coverages/ \
        --gen_data /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/generations/gpt-3.5-turbo-0125_maxTokens512_numSeq10_topP0.95_numSent5_startSent_promptIdx5_len788.jsonl \
        --min_ngram 4 \
        --parallel \
        --source_docs swj0419/BookMIA;  
    ```

    OR launch all scoring with a bash script:

    ```
    ./code/helper/dj_search/scripts/launch_all_search.sh
    ```

    Coverages will be stored in the `code/experiments/ours/[dataset]/coverages/[split]` folder

3. Get creativity index from coverages

    ```
        python3 -m code.experiments.ours.get_creativity_index \
        --coverage_path code/experiments/ours/outputs/bookMIA/coverages/train/Llama-2-7b-hf_maxTok512_minTok0_numSeq20_topP0.95_temp1.0_numSent1_startSent1_numWord-1_startWord-1_useSentF_promptIdx1_len494_2024-10-30-21:06:53_4_onedoc.jsonl
    ```

Or to run them all:

    ```
    ./code/experiments/ours/get_all_creativity_index.sh
    ```