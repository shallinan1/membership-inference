from unidecode import unidecode
from nltk.tokenize.casual import casual_tokenize
import numpy as np
import os
import json
from tqdm import tqdm
from IPython import embed
import argparse
from code.utils import load_json
import re

LOW_CI_BOUND=2
HIGH_CI_BOUND=12

def get_ngram_coverage(text, spans, min_gram):
    tokens = casual_tokenize(unidecode(text))
    flags = [False for _ in tokens]
    for span in spans:
        span_length = span['end_index'] - span['start_index']
        if span_length >= min_gram:
            flags[span['start_index']: span['end_index']] = [True] * span_length

    if len([f for f in flags if f]) == 0:
        return 0
    else: 
        coverage = len([f for f in flags if f]) / len(flags)
        return coverage

def compute_ci_statistic(outputs, min_ngram, max_ngram):
    total_coverages, total_stds = [], []
    ngram_list = list(range(min_ngram, max_ngram + 1))
    for output in outputs:
        coverages = []
        for min_ngram in ngram_list:
            coverage = get_ngram_coverage(output['text'], output['matched_spans'], min_ngram)
            coverages.append(coverage)

        total_coverages.append(sum(coverages))
        total_stds.append(np.std(coverages))

    return total_coverages

def main(args):
    data = load_json(args.coverage_path)
    CREATIVITY_CONSTANT = args.max_ngram - args.min_ngram + 1
    os.makedirs(args.output_dir, exist_ok=True)

    for cur_data in tqdm(data, leave=False, desc = "Iterating through original data"):
        cur_creativity = compute_ci_statistic(cur_data["coverage"], args.min_ngram, args.max_ngram)
        cur_data["creativity"] = [CREATIVITY_CONSTANT - c for c in cur_creativity]

    output_file = os.path.join(args.output_dir, os.path.basename(args.coverage_path).replace('.jsonl', f"_CI{args.min_ngram}-{args.max_ngram}.jsonl"))
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
        f.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSONL file and compute statistics.")
    parser.add_argument("--coverage_path", type=str, help="Path to the input JSONL file.")
    parser.add_argument("--output_dir", type=str, default = "experiments/ours/outputs/bookMIA/cis/")
    parser.add_argument("--min_ngram", type=int, default=LOW_CI_BOUND, help="Minimum n-gram for coverage calculation.")
    parser.add_argument("--max_ngram", type=int, default=HIGH_CI_BOUND, help="Maximum n-gram for coverage calculation.")
    
    args = parser.parse_args()
    main(args)

    """
    python3 -m code.experiments.ours.get_creativity_index \
    --coverage_path /gscratch/xlab/hallisky/membership-inference/outputs/ours/tulu_v1/coverages/val/tulu-7b-finalized_maxTok512_minTok0_numSeq20_topP0.95_temp1.0_numSent1_startSent-1_numWord-1_startWord-1_useSentF_promptIdx0_len92_2025-02-17-19:30:08_2_onedoc.jsonl \
    --output_dir  /gscratch/xlab/hallisky/membership-inference/outputs/ours/tulu_v1/creativities/val/ \
    --min_ngram 2 \
    --max_ngram 12

    python3 -m code.helper.dj_search.dj_search_exact_LLM \
    --task bookMIA \
    --output_dir /gscratch/xlab/hallisky/membership-inference/outputs/ours/bookMIA/coverages/train \
    --gen_data /gscratch/xlab/hallisky/membership-inference/outputs/ours/bookMIA/generations/train/gpt-3.5-turbo-0125_maxTok512_minTok0_numSeq20_topP0.95_temp1.0_numSent5_startSent1_numWord-1_startWord-1_useSentF_promptIdx5_len494_2025-01-11-23:06:39.jsonl \
    --min_ngram 4 \
    --parallel \
    --source_docs swj0419/BookMIA;

    """