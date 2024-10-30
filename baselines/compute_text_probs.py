from user_secrets import CACHE_PATH
import os
# Set up environment variables
os.environ["HF_HOME"] = CACHE_PATH
os.environ["HF_DATASETS_PATH"] = CACHE_PATH

import numpy as np
from pathlib import Path
import torch
import zlib
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
import argparse
import os
from pathlib import Path
import logging
import random
import json
import pandas as pd
from IPython import embed
import pandas as pd
from utils import load_jsonl
from IPython import embed
import torch

# TODO add sliding window when tokenized length of texts > model length (ie, gpt2-xl)
def getPerplexityProbsLoss(sentence, model, tokenizer):
    """
    Calculate the perplexity of a sentence given a language model and tokenizer.

    Parameters:
    - sentence (str): The input sentence for which perplexity is to be calculated.
    - model (torch.nn.Module): The pre-trained language model to use for evaluation.
    - tokenizer (transformers.PreTrainedTokenizer): The tokenizer corresponding to the model.

    Returns:
    - tuple:
        - float: The perplexity of the input sentence.
        - list: Log-probabilities of each token in the sentence.
        - float: The raw loss value from the model.
    
    Notes:
    Perplexity is calculated using the model's loss on the input sentence. This function also
    returns the log-probabilities for each token in the sentence based on the model's predictions.
    """
    input_ids = tokenizer(sentence, return_tensors="pt").input_ids.to('cuda')
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs.loss, outputs.logits
    
    # Apply softmax to the logits to get probabilities
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    # probabilities = torch.nn.functional.softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
    return torch.exp(loss).item(), all_prob, loss.item()

def main(args):
    # TODO Save all log probabilities for minkplusplus method
    model_name = args.target_model.split(os.sep)[-1]
    data_path_split = args.data_path.split(os.sep)
    data_name = data_path_split[1]
    data_split = data_path_split[-1][:-6] # Prune the jsonl

    output_dir = os.path.join(args.output_dir, data_name, data_split, "probs")
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    if os.path.exists(args.data_path):
        data = load_jsonl(args.data_path)
    else:
        print("Please use valid data path. See README for valid data after preprocssing/downloading.")

    model = AutoModelForCausalLM.from_pretrained(args.target_model, device_map='auto')
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    print("Model one loaded")


    for d in tqdm(data):
        text = d[args.key_name]
        perplexity, log_probs, loss = getPerplexityProbsLoss(text, model, tokenizer)
        perplexity_lower, log_probs_lower, loss_lower = getPerplexityProbsLoss(text.lower(), model, tokenizer)

        d["perplexity"] = perplexity
        d["log_probs"] = log_probs
        d["loss"] = loss
        d["perplexity_lower"] = perplexity_lower
        d["log_probs_lower"] = log_probs_lower
        d["loss_lower"] = loss_lower
    
    # Save to JSONL format
    output_file = os.path.join(output_dir, model_name + '.jsonl')
    with open(output_file, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--target_model', type=str, default="text-davinci-003", help="the model to attack: huggyllama/llama-65b, text-davinci-003")
    parser.add_argument('--output_dir', type=str, default="baselines/")
    parser.add_argument('--data_path', type=str, default="data/bookMIA/split-random-overall/val.jsonl", help="the dataset to evaluate: default is WikiMIA")
    parser.add_argument('--length', type=int, default=64, help="the length of the input text to evaluate. Choose from 32, 64, 128, 256")
    parser.add_argument('--key_name', type=str, default="input", help="the key name corresponding to the input text. Selecting from: input, parapgrase")
    main(parser.parse_args())


    """
    CUDA_VISIBLE_DEVICES=0 python3 -m baselines.compute_text_probs \
        --target_model meta-llama/Llama-2-7b-hf \
        --key_name snippet;

    CUDA_VISIBLE_DEVICES=0 python3 -m baselines.compute_text_probs \
        --target_model openai-community/gpt2-xl \
        --key_name snippet;
    
    

    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m baselines.compute_text_probs \
        --target_model meta-llama/Llama-2-70b-hf \
        --key_name snippet;
    """

