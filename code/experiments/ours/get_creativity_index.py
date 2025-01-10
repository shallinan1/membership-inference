import json
from unidecode import unidecode
from nltk.tokenize.casual import casual_tokenize
import numpy as np
import numpy as np
import os
import itertools
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from sklearn.metrics import roc_curve, auc, accuracy_score
from IPython import embed
import argparse
from utils import load_jsonl, load_json, combine_lists, combine_dicts, combine_list_of_dicts, save_to_jsonl
import re

LOW_CI_BOUND=3
HIGH_CI_BOUND=12

def get_ngram_coverage(text, spans, min_gram):
    try:
        tokens = casual_tokenize(unidecode(text))
    except:
        from IPython import embed; embed()
    flags = [False for _ in tokens]
    for span in spans:
        span_length = span['end_index'] - span['start_index']
        if span_length >= min_gram:
            flags[span['start_index']: span['end_index']] = [True] * span_length

    coverage = len([f for f in flags if f]) / len(flags)
    return coverage

def compute_ci_statistic(outputs, min_ngram, max_ngram):

    avg_coverages, avg_std = [], []
    ngram_list = list(range(min_ngram, max_ngram + 1))
    for output in outputs:
        coverages = []
        for min_ngram in ngram_list:
            coverage = get_ngram_coverage(output['text'], output['matched_spans'], min_ngram)
            coverages.append(coverage)

        avg_coverages.append(np.sum(coverages))
        avg_std.append(np.std(coverages))

    return avg_coverages

def main(args):
    # Extract the directory and filename
    directory, filename_with_date = os.path.split(args.coverage_path)

    # Use a regular expression to find the date portion in the filename
    date_pattern = r'_\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2}'
    match = re.search(date_pattern, filename_with_date)

    if not match:
        raise ValueError("Date pattern not found in the file name.")

    # Extract prefix and suffix around the date portion
    prefix = filename_with_date[:match.start()]
    suffix = filename_with_date[match.end():]

    # Initialize a list to store matching file paths
    matching_paths = []
    
    # List all files in the directory and filter by matching prefix and suffix
    for filename in sorted(os.listdir(directory)):
        if filename.startswith(prefix) and filename.endswith(suffix):
            # Ensure there's a date-like pattern in the filename between prefix and suffix
            if re.search(date_pattern, filename):
                matching_paths.append(os.path.join(directory, filename))

    if args.coverage_path not in matching_paths[0]:
        return # Don't re-do computations

    combined_coverage_data = []

    # Iterate over the matching paths and combine the data
    for cur_path in matching_paths:
        coverage_data = load_json(cur_path)
        if not combined_coverage_data:
            combined_coverage_data = coverage_data
        else:
            combined_coverage_data = combine_lists(combined_coverage_data, coverage_data)

    # Further processing and output as needed
    print(len(matching_paths))

    if "train" in args.coverage_path:
        args.output_dir = os.path.join(args.output_dir, "train")
    elif "val" in args.coverage_path:
        args.output_dir = os.path.join(args.output_dir, "val")

    os.makedirs(args.output_dir, exist_ok=True)

    cis = [compute_ci_statistic(cur_data, args.min_ngram, args.max_ngram) for cur_data in tqdm(coverage_data, leave=False, desc = "Iterating through original data", position=1)]

    output_filename = os.path.join(args.output_dir, f"{filename_with_date.replace('.jsonl', '')}_CI_{args.min_ngram}_{args.max_ngram}.jsonl")

    save_to_jsonl(cis, output_filename)

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
    --coverage_path experiments/ours/outputs/bookMIA/coverages/train/Llama-2-7b-hf_maxTok512_minTok0_numSeq20_topP0.95_temp1.0_numSent1_startSent1_numWord-1_startWord-1_useSentF_promptIdx1_len494_2024-10-30-21:06:53_4_onedoc.jsonl

    """