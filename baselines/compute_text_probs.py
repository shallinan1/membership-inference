import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level='ERROR')

from user_secrets import CACHE_PATH
import os
# Set up environment variables
os.environ["HF_HOME"] = CACHE_PATH
os.environ["HF_DATASETS_PATH"] = CACHE_PATH

import numpy as np
from pathlib import Path
import openai
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

# def inference(model1, tokenizer1,text, ex, modelname1):
#     pred = {}

#     perplexity, probabilities, loss = getPerplexityProbLoss(text, model1, tokenizer1, gpu=model1.device)
#     p_lower, _, p_lower_likelihood = getPerplexityProbLoss(text.lower(), model1, tokenizer1, gpu=model1.device)

#    # ppl
#     pred["ppl"] = p1
#     # Ratio of log ppl of large and small models
#     pred["ppl/Ref_ppl (calibrate PPL to the reference model)"] = p1_likelihood-p_ref_likelihood
 
#     # Ratio of log ppl of lower-case and normal-case
#     pred["ppl/lowercase_ppl"] = -(np.log(p_lower) / np.log(p1)).item()
#     # Ratio of log ppl of large and zlib
#     zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))
#     pred["ppl/zlib"] = np.log(p1)/zlib_entropy
#     # min-k prob
#     for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
#         k_length = int(len(all_prob)*ratio)
#         topk_prob = np.sort(all_prob)[:k_length]
#         pred[f"Min_{ratio*100}% Prob"] = -np.mean(topk_prob).item()

#     ex["pred"] = pred
#     return ex

def evaluate_data(test_data, model1, model2, tokenizer1, tokenizer2, col_name, modelname1, modelname2):
    print(f"all data size: {len(test_data)}")
    all_output = []
    test_data = test_data
    for ex in tqdm(test_data): 
        text = ex[col_name]
        new_ex = inference(model1, model2, tokenizer1, tokenizer2, text, ex, modelname1, modelname2)
        all_output.append(new_ex)
    return all_output

def main(args):
    # TODO save naming based on split of bookmia used
    model_name = args.target_model.split("/")[-1]
    args.output_dir = f"{args.output_dir}/{args.key_name}"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load dataset
    if os.path.exists(args.data_path):
        data = load_jsonl(args.data_path)
    else:
        print("Please use valid data path. See README for valid data after preprocssing/downloading.")

    model = AutoModelForCausalLM.from_pretrained(args.target_model, device_map='auto')
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    print("Model one loaded")


    for d in tqdm(data[:3]):
        text = d[args.key_name]
        perplexity, log_probs, loss = getPerplexityProbsLoss(text, model, tokenizer)
        perplexity_lower, log_probs_lower, loss_lower = getPerplexityProbsLoss(text.lower(), model, tokenizer)

        d["perplexity"] = perplexity
        d["log_probs"] = log_probs
        d["loss"] = loss
        d["perplexity_lower"] = perplexity_lower
        d["log_probs_lower"] = log_probs_lower
        d["loss_lower"] = loss_lower
    
    embed()

    # Save to JSONL format
    output_file = os.path.join(args.output_dir, model_name + '.jsonl')
    with open(output_file, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

    # model1, tokenizer1 = load_model(args.ref_model)
    # if "jsonl" in args.data:
    #     data = load_jsonl(f"{args.data}")
    # else: # load data from huggingface
    #     dataset = load_dataset(args.data, split=f"WikiMIA_length{args.length}")
    #     data = convert_huggingface_data_to_list_dic(dataset)
#     embed()
    # all_output = evaluate_data(final_subset, model1, model2, tokenizer1, tokenizer2, args.key_name, args.target_model, args.ref_model)
    
    # fig_fpr_tpr(all_output, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--target_model', type=str, default="text-davinci-003", help="the model to attack: huggyllama/llama-65b, text-davinci-003")
    parser.add_argument('--output_dir', type=str, default="baselines/bookMIA_out")
    parser.add_argument('--data_path', type=str, default="data/bookMIA/split-random-overall/val.jsonl", help="the dataset to evaluate: default is WikiMIA")
    parser.add_argument('--length', type=int, default=64, help="the length of the input text to evaluate. Choose from 32, 64, 128, 256")
    parser.add_argument('--key_name', type=str, default="input", help="the key name corresponding to the input text. Selecting from: input, parapgrase")
    main(parser.parse_args())


    """

    CUDA_VISIBLE_DEVICES=0 python3 -m baselines.compute_text_probs \
        --target_model meta-llama/Llama-2-7b-hf \
        --key_name snippet;


    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m baselines.compute_text_probs \
        --target_model meta-llama/Llama-2-70b-hf \
        --key_name snippet;

    """

