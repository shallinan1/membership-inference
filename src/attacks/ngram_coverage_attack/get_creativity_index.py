"""
Creativity Index Computation for N-Gram Coverage Attack

This module implements the creativity index computation component of the N-Gram Coverage 
Attack method for membership inference attacks against language models.

The module processes coverage analysis results by computing creativity indices across
multiple n-gram ranges and normalizes coverage statistics for comparative analysis.
It extends the n-gram coverage computation with weighted metrics that account for
varying n-gram lengths and unique span detection.

Pipeline:
    1. Load coverage analysis results from JSONL file (output of compute_ngram_coverage.py)
    2. Compute additional coverage metrics with unique span filtering
    3. Calculate creativity indices by summing coverage across n-gram ranges
    4. Generate both standard and unique-span versions of all metrics
    5. Save enriched results with creativity statistics for downstream analysis

Outputs:
    JSONL file containing:
    - Original coverage data from previous pipeline stage
    - Additional coverage metrics (gen_length, ref_length, total_length variants)
    - Creativity indices across configurable n-gram ranges
    - Both standard and unique-span filtered versions of all metrics

Hardcoded Configuration:
    - Default n-gram bounds: LOW_CI_BOUND=2, HIGH_CI_BOUND=12 for creativity index calculation
    - Tokenization: NLTK casual tokenizer for word-level processing
    - Min-gram filtering: Uses min_gram=1 for comprehensive coverage in additional metrics
    - Output naming: Appends "_CI{min_ngram}-{max_ngram}" suffix to input filename

Usage:
    python -m src.attacks.ngram_coverage_attack.get_creativity_index \
        --coverage_path PATH_TO_COVERAGE.jsonl \
        --output_dir OUTPUT_DIRECTORY \
        --min_ngram 1 \
        --max_ngram 12
"""

from unidecode import unidecode
from nltk.tokenize.casual import casual_tokenize
import os
from tqdm import tqdm
import argparse
from typing import List, Dict, Any
from src.utils.io_utils import load_jsonl, save_to_jsonl
import nltk

# Define constants for the argparser
LOW_CI_BOUND=2
HIGH_CI_BOUND=12

# Define global tokenization function
tokenize_func = lambda x: nltk.tokenize.casual.casual_tokenize(x)

# TODO checking for subsets too
def get_ngram_coverage(
    text: str, 
    spans: List[Dict[str, Any]], 
    min_gram: int, 
    ref_length: int, 
    unique_coverages: bool = False
) -> Dict[str, float]:
    tokens = casual_tokenize(unidecode(text))
    flags = [False for _ in tokens]
    seen = set()

    for span in spans:
        span_text = span['span_text']
        span_length = span['end_index'] - span['start_index']
        if span_length >= min_gram:
            if not unique_coverages or span_text not in seen:
                flags[span['start_index']: span['end_index']] = [True] * span_length
                
        seen.add(span_text)

    if len([f for f in flags if f]) == 0:
        coverage_gen, coverage_ref, coverage_total = 0, 0, 0
    else: 
        ref_adjusted_length = max(len(flags), ref_length)
        coverage_gen = len([f for f in flags if f]) / len(flags)
        coverage_ref = len([f for f in flags if f]) / ref_adjusted_length
        coverage_total = (2 * len([f for f in flags if f])) / (len(flags) + ref_adjusted_length)
    return {
        "coverages_gen_length": coverage_gen,
        "coverages_ref_length": coverage_ref, 
        "coverages_total_length": coverage_total
        }

def compute_ci_statistic(
    outputs: List[Dict[str, Any]], 
    min_ngram: int, 
    max_ngram: int, 
    ref_length: int, 
    unique_coverages: bool = False
) -> List[Dict[str, float]]:
    total_coverages = []
    ngram_list = list(range(min_ngram, max_ngram + 1))
    for output in outputs:
        coverages = []
        for cur_ngram in ngram_list:
            coverage_dict = get_ngram_coverage(output['text'], output['matched_spans'], cur_ngram, ref_length, unique_coverages)
            coverages.append(coverage_dict)

        total_coverages.append({key.replace("coverages", "creativities"): sum(d[key] for d in coverages) for key in coverages[0]})

    return total_coverages

def main(args: argparse.Namespace) -> None:
    data = load_jsonl(args.coverage_path)
    CREATIVITY_CONSTANT = args.max_ngram - args.min_ngram + 1
    os.makedirs(args.output_dir, exist_ok=True)

    for cur_data in tqdm(data, desc="Adding more coverages"):
        length_ref = len(tokenize_func(unidecode(cur_data["snippet_no_prompt"])))
        cur_coverages = []
        cur_coverages_unique = []
        for coverage in cur_data["coverage"]:
            cur_coverages.append(
                get_ngram_coverage(coverage["text"], 
                                   coverage["matched_spans"], 
                                   min_gram=1, 
                                   ref_length=length_ref,
                                   unique_coverages=False))
            cur_coverages_unique.append(
                get_ngram_coverage(coverage["text"], 
                                   coverage["matched_spans"], 
                                   min_gram=1, 
                                   ref_length=length_ref,
                                   unique_coverages=True))
        for coverage_key in ["coverages_gen_length", "coverages_ref_length", "coverages_total_length"]:
            cur_data[coverage_key] = [c[coverage_key] for c in cur_coverages]
            cur_data[coverage_key + "_unique"] = [c[coverage_key] for c in cur_coverages_unique]
    
    for cur_data in tqdm(data, desc = "Iterating through original data"):
        length_ref = len(tokenize_func(unidecode(cur_data["snippet_no_prompt"])))
        cur_creativity = compute_ci_statistic(cur_data["coverage"], args.min_ngram, args.max_ngram, length_ref, unique_coverages=False)
        cur_creativity_unique = compute_ci_statistic(cur_data["coverage"], args.min_ngram, args.max_ngram, length_ref, unique_coverages=True)
        # cur_data["creativity"] = [CREATIVITY_CONSTANT - c for c in cur_creativity]
        
        for creativity_key in ["creativities_gen_length", "creativities_ref_length", "creativities_total_length"]:
            cur_data[creativity_key] = [c[creativity_key] for c in cur_creativity]
            cur_data[creativity_key + "_unique"] = [c[creativity_key] for c in cur_creativity_unique]

    output_file = os.path.join(args.output_dir, os.path.basename(args.coverage_path).replace('.jsonl', f"_CI{args.min_ngram}-{args.max_ngram}.jsonl"))
    save_to_jsonl(data, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute creativity indices from n-gram coverage analysis results for membership inference attack evaluation.")
    parser.add_argument("--coverage_path", type=str, help="Path to the input JSONL file.")
    parser.add_argument("--output_dir", type=str, default = "experiments/ours/outputs/bookMIA/cis/")
    parser.add_argument("--min_ngram", type=int, default=LOW_CI_BOUND, help="Minimum n-gram for coverage calculation.")
    parser.add_argument("--max_ngram", type=int, default=HIGH_CI_BOUND, help="Maximum n-gram for coverage calculation.")
    
    args = parser.parse_args()
    main(args)
