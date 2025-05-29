from code.user_secrets import CACHE_PATH
import os
# Set up environment variables
os.environ["HF_HOME"] = CACHE_PATH
os.environ["HF_DATASETS_PATH"] = CACHE_PATH

import argparse
import os
from code.utils import load_jsonl, save_to_jsonl
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
import re
from collections import Counter
import zlib
import argparse


def ngrams(sequence, n) -> zip:
    """
    Generates n-grams from a sequence.
    """
    return zip(*[sequence[i:] for i in range(n)])

def rouge_n(candidate: list, reference: list, n=1) -> float:
    """
    Calculates the ROUGE-N score between a candidate and a reference.
    """
    if not candidate or not reference:
        return 0
    candidate_ngrams = list(ngrams(candidate, n))
    reference_ngrams = list(ngrams(reference, n))
    ref_words_count = Counter(reference_ngrams)
    cand_words_count = Counter(candidate_ngrams)
    overlap = ref_words_count & cand_words_count
    recall = sum(overlap.values()) / len(reference)
    precision = sum(overlap.values()) / len(candidate)
    return recall

def clean_text(text: str, model_name: str) -> str:
    """
    Removes specific special tokens from the text based on the model's output.
    """
    if "gpt-j-6B" in model_name or "pythia-6.9b" in model_name:
        return re.sub(r'<\|endoftext\|>', '', text)
    elif "Llama-2-7b" in model_name or "opt-6.7b" in model_name:
        text = re.sub(r'<s> ', '', text)
        return re.sub(r'</s>', '', text)
    return text

def get_suffix(text: str, prefix_ratio: float) -> list:
    """
    Extracts a suffix from the given text, based on the specified prefix ratio.
    The text length is computed dynamically from the input text.
    """
    words = text.split(" ")
    words = [word for word in words if word != ""]
    text_length = len(words)
    return words[round(text_length*prefix_ratio):]

def main(args):
    # Create output directory
    output_dir = f"outputs/baselines/{args.task}/{args.split}/SPL_overlaps"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all files in the SPL_generations directory
    input_dir = f"outputs/baselines/{args.task}/{args.split}/SPL_generations"
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist")
        return
    
    for filename in os.listdir(input_dir):
        if not filename.endswith('.jsonl'):
            continue
            
        print(f"Processing {filename}...")
        
        # Extract model name from filename
        model_name = filename.split('_')[0]  # Assuming filename format is "modelname_*.jsonl"
        
        # Load the data
        data = load_jsonl(os.path.join(input_dir, filename))
        
        # Process both regular and zlib versions
        for use_zlib in [False, True]:
            rouge_scores = []
            for item in tqdm(data, desc=f"Computing {'zlib ' if use_zlib else ''}ROUGE scores"):
                gold_text = item.get("input" if "wikiMIA" in args.task else "snippet", "")
                generations = item.get("generation", [])
                
                suffix_ref = get_suffix(gold_text, 0.5)
                item["rest_of_text"] = " ".join(suffix_ref)  # Set the rest_of_text field
                item_rouge_scores = []
                
                for gen in generations:
                    text_output = clean_text(gen, model_name)
                    suffix_cand = get_suffix(text_output, 0.5)
                    
                    if use_zlib:
                        zlib_cand = zlib.compress(" ".join(suffix_cand).encode('utf-8'))
                        item_rouge_scores.append(rouge_n(suffix_cand, suffix_ref, n=1) * len(zlib_cand))
                    else:
                        item_rouge_scores.append(rouge_n(suffix_cand, suffix_ref, n=1))
                
                rouge_scores.append(item_rouge_scores)
            
            # Save results by appending to existing file
            output_file = os.path.join(output_dir, filename.replace('.jsonl', '_zlib.jsonl' if use_zlib else '.jsonl'))
            for item, scores in zip(data, rouge_scores):
                item["rouge_scores"] = scores
                item["final_score"] = np.mean(scores) if scores else 0.0  # Compute average of rouge scores
            
            save_to_jsonl(data, output_file)
            print(f"Saved {'zlib ' if use_zlib else ''}results to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="bookMIA", help='The task (dataset)')
    parser.add_argument("--split", type=str, default="train", help='The data split')
    
    main(parser.parse_args())

    """
    python -m code.experiments.baselines.SPL_compute_overlap \
        --task bookMIA \
        --split train \
    """ 