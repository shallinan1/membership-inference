from code.user_secrets import CACHE_PATH
import os
# Set up environment variables
os.environ["HF_HOME"] = CACHE_PATH
os.environ["HF_DATASETS_PATH"] = CACHE_PATH

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse
import os
from code.utils import load_jsonl, save_to_jsonl
from IPython import embed
import torch

# TODO add sliding window when tokenized length of texts > model length (ie, gpt2-xl)
def get_perplexity_probs_loss(sentence, model, tokenizer):
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
    data_path = f"data/{args.task}/split-random-overall/{args.split}.jsonl"

    # TODO Save all log probabilities for mink++ method
    model_name = args.target_model.split(os.sep)[-1]

    output_dir = f"outputs/baselines/{args.task}/{args.split}/probs"
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(data_path):
        data = load_jsonl(data_path)
    else:
        print("Please use valid data path. See README for valid data after preprocssing/downloading.")

    model = AutoModelForCausalLM.from_pretrained(args.target_model, device_map='auto')
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)

    for d in tqdm(data):
        text = d[args.key_name]
        perplexity, log_probs, loss = get_perplexity_probs_loss(text, model, tokenizer)
        perplexity_lower, log_probs_lower, loss_lower = get_perplexity_probs_loss(text.lower(), model, tokenizer)

        d["perplexity"] = perplexity
        d["log_probs"] = log_probs
        d["loss"] = loss
        d["perplexity_lower"] = perplexity_lower
        d["log_probs_lower"] = log_probs_lower
        d["loss_lower"] = loss_lower
    
    # Save to JSONL format
    output_file = os.path.join(output_dir, model_name + '.jsonl')
    save_to_jsonl(data, output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--target_model', type=str, default="text-davinci-003", help="the model to attack: huggyllama/llama-65b, text-davinci-003")
    parser.add_argument('--task', type=str, default="pile_external", help="the task (dataset)")
    parser.add_argument('--split', type=str, default="train", help="the data split")
    parser.add_argument('--key_name', type=str, default="input", help="the key name corresponding to the input text. Selecting from: input, parapgrase")
    main(parser.parse_args())

