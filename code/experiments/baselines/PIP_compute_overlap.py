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
from rouge_score import rouge_scorer

def compute_rouge_l(text1, text2):
    """Compute the Rouge-L score between two texts."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(text1, text2)
    return scores['rougeL'].fmeasure

def process_item(item):
    """Process a single item's generations against its gold text."""
    # Get the gold text (rest_of_text)
    gold_text = item.get("rest_of_text", "")
    
    # Get all generations
    generations = item.get("generation", [])
    
    # Compute Rouge-L scores for each generation
    rouge_scores = []
    for gen in generations:
        score = compute_rouge_l(gen, gold_text)
        rouge_scores.append(score)
    
    # Add Rouge-L scores to the item
    item["rouge_scores"] = rouge_scores
    item["final_score"] = max(rouge_scores) if rouge_scores else 0
    item["best_generation"] = generations[np.argmax(rouge_scores)] if rouge_scores else ""
    
    return item

def main(args):
    # Create output directory
    output_dir = f"outputs/baselines/{args.task}/{args.split}/PIP_overlaps"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all files in the PIP_generations directory
    input_dir = f"outputs/baselines/{args.task}/{args.split}/PIP_generations"
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
                desc="Computing Rouge-L scores"
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
    python -m code.experiments.baselines.PIP_compute_overlap --task bookMIA --split train;
    python -m code.experiments.baselines.PIP_compute_overlap --task wikiMIA --split test;
    python -m code.experiments.baselines.PIP_compute_overlap --task wikiMIA_hard --split test;
    python -m code.experiments.baselines.PIP_compute_overlap --task wikiMIA_update --split test;
    python -m code.experiments.baselines.PIP_compute_overlap --task tulu_v1 --split test;
    """ 