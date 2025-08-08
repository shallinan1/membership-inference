import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
CACHE_PATH = os.getenv("CACHE_PATH")

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

def compute_lcs(text1, text2):
    """Compute the longest common subsequence between two texts using dynamic programming."""
    # Split into words
    words1 = text1.split()
    words2 = text2.split()
    
    # Create DP table
    m, n = len(words1), len(words2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if words1[i-1] == words2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Reconstruct the LCS
    lcs_words = []
    i, j = m, n
    while i > 0 and j > 0:
        if words1[i-1] == words2[j-1]:
            lcs_words.append(words1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    return dp[m][n], ' '.join(reversed(lcs_words))

def process_item(item):
    """Process a single item's generations against its gold text."""
    # Get the gold text (rest_of_text)
    gold_text = item.get("rest_of_text", "")
    
    # Get all generations
    generations = item.get("generation", [])
    
    # Compute LCS length and words for each generation
    lcs_lengths = []
    lcs_words = []
    for gen in generations:
        lcs_length, lcs_text = compute_lcs(gen, gold_text)
        lcs_lengths.append(lcs_length)
        lcs_words.append(lcs_text)
    
    # Add LCS lengths and words to the item
    item["lcs_lengths"] = lcs_lengths
    item["lcs_words"] = lcs_words
    item["final_score"] = max(lcs_lengths) if lcs_lengths else 0
    item["max_lcs_words"] = lcs_words[np.argmax(lcs_lengths)] if lcs_lengths else ""
    
    return item

def main(args):
    # Create output directory
    output_dir = f"outputs/baselines/{args.task}/{args.split}/VMA_overlaps"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all files in the VMA_generations directory
    input_dir = f"outputs/baselines/{args.task}/{args.split}/VMA_generations"
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist")
        return
    
    # Determine number of processes to use
    num_processes = min(cpu_count(), args.num_processes) if args.num_processes > 0 else cpu_count()
    print(f"Using {num_processes} processes")
    
    for filename in os.listdir(input_dir):
        if not filename.endswith('.jsonl'):
            continue
            
        print(f"Processing {filename}...")
        
        # Load the data
        data = load_jsonl(os.path.join(input_dir, filename))
        
        # Create a pool of workers
        with Pool(processes=num_processes) as pool:
            # Process items in parallel with progress bar
            processed_data = list(tqdm(
                pool.imap(process_item, data),
                total=len(data),
                desc="Computing LCS"
            ))
        
        # Save the results
        output_file = os.path.join(output_dir, filename)
        save_to_jsonl(processed_data, output_file)
        print(f"Saved results to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--task', type=str, default="bookMIA", help='The task (dataset)')
    parser.add_argument('--split', type=str, default="train", help='The data split')
    parser.add_argument('--num_processes', type=int, default=-1, 
                      help='Number of processes to use. Default (-1) uses all available CPUs.')
    
    main(parser.parse_args())

    """
    python -m code.experiments.baselines.VMA_compute_overlap --task bookMIA --split train;
    python -m code.experiments.baselines.VMA_compute_overlap --task wikiMIA --split test;
    python -m code.experiments.baselines.VMA_compute_overlap --task wikiMIA_hard --split test;
    python -m code.experiments.baselines.VMA_compute_overlap --task wikiMIA_update --split test;
    python -m code.experiments.baselines.VMA_compute_overlap --task tulu_v1 --split test;
    """ 