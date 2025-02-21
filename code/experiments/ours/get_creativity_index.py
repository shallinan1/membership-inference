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
import nltk
tokenize_func = lambda x: nltk.tokenize.casual.casual_tokenize(x)
from unidecode import unidecode

LOW_CI_BOUND=2
HIGH_CI_BOUND=12

def get_ngram_coverage(text, spans, min_gram, ref_length):
    tokens = casual_tokenize(unidecode(text))
    flags = [False for _ in tokens]
    for span in spans:
        span_length = span['end_index'] - span['start_index']
        if span_length >= min_gram:
            flags[span['start_index']: span['end_index']] = [True] * span_length

    if len([f for f in flags if f]) == 0:
        coverage_gen, coverage_ref, coverage_total = 0, 0, 0
    else: 
        coverage_gen = len([f for f in flags if f]) / len(flags)
        coverage_ref = len([f for f in flags if f]) / ref_length
        coverage_total = (2 * len([f for f in flags if f])) / (len(flags) + ref_length)
    return {
        "coverages_gen_length": coverage_gen,
        "coverages_ref_length": coverage_ref, 
        "coverages_total_length": coverage_total
        }

def compute_ci_statistic(outputs, min_ngram, max_ngram, ref_length):
    total_coverages = []
    ngram_list = list(range(min_ngram, max_ngram + 1))
    for output in outputs:
        coverages = []
        for min_ngram in ngram_list:
            coverage_dict = get_ngram_coverage(output['text'], output['matched_spans'], min_ngram, ref_length)
            coverages.append(coverage_dict)

        total_coverages.append({key.replace("coverages", "creativities"): sum(d[key] for d in coverages) for key in coverages[0]})

    return total_coverages

def main(args):
    data = load_json(args.coverage_path)
    CREATIVITY_CONSTANT = args.max_ngram - args.min_ngram + 1
    os.makedirs(args.output_dir, exist_ok=True)

    for cur_data in tqdm(data, desc="Adding more coverages"):
        length_ref = len(tokenize_func(unidecode(cur_data["snippet_no_prompt"])))
        cur_coverages = []
        for coverage in cur_data["coverage"]:
            cur_coverages.append(
                get_ngram_coverage(coverage["text"], 
                                   coverage["matched_spans"], 
                                   min_gram=1, 
                                   ref_length=length_ref))
        for coverage_key in ["coverages_gen_length", "coverages_ref_length", "coverages_total_length"]:
            cur_data[coverage_key] = [c[coverage_key] for c in cur_coverages]
    
    for cur_data in tqdm(data, desc = "Iterating through original data"):
        length_ref = len(tokenize_func(unidecode(cur_data["snippet_no_prompt"])))
        cur_creativity = compute_ci_statistic(cur_data["coverage"], args.min_ngram, args.max_ngram, length_ref)

        # cur_data["creativity"] = [CREATIVITY_CONSTANT - c for c in cur_creativity]
        for creativity_key in ["creativities_gen_length", "creativities_ref_length", "creativities_total_length"]:
            cur_data[creativity_key] = [c[creativity_key] for c in cur_creativity]
    
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
    --min_ngram 1 \
    --max_ngram 12
    """