import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
CACHE_PATH = os.getenv("CACHE_PATH")

# Set up environment variables
os.environ["HF_HOME"] = CACHE_PATH
os.environ["HF_DATASETS_PATH"] = CACHE_PATH

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse
import os
from code.utils import load_jsonl, save_to_jsonl, convert_to_tulu_v1_format
from IPython import embed
import torch
from sklearn.metrics import roc_auc_score
import random

def main(args):
    random.seed(0)
    data_path = f"data/{args.task}/split-random-overall/{args.split}.jsonl"

    if os.path.exists(data_path):
        data = load_jsonl(data_path)
    else:
        print("Please use valid data path. See README for valid data after preprocssing/downloading.")

    # Print AUROC score with random label
    labels = [d["label"] for d in data]

    # Make random labels
    random_predictions = [random.random() for _ in range(len(labels))]
    
    # Get ROC score
    auroc = roc_auc_score(labels, random_predictions)
    
    # Print score
    print(f"Random baseline AUROC for {args.task}/{args.split}: {auroc:.4f}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--task', type=str, default="pile_external", help="the task (dataset)")
    parser.add_argument('--split', type=str, default="train", help="the data split")
    main(parser.parse_args())

    """
    python3 -m code.experiments.baselines.random_baseline
    """

