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

# TODO add sliding window when tokenized length of texts > model length (ie, gpt2-xl)
# TODO re-look and vigorously analyze this
def get_perplexity_probs_loss(sentence, model, tokenizer, mask_after_token=None):
    """
    Calculate the perplexity of a sentence given a language model and tokenizer.

    Parameters:
    - sentence (str): The input sentence for which perplexity is to be calculated.
    - model (torch.nn.Module): The pre-trained language model to use for evaluation.
    - tokenizer (transformers.PreTrainedTokenizer): The tokenizer corresponding to the model.
    - mask_after_token (str, optional): The token after which the loss should be computed. 
                                        If None, loss is computed for all tokens.
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
    labels = input_ids.clone()

    mask_index = 0
    if mask_after_token is not None:
        mask_token_ids = tokenizer(mask_after_token, add_special_tokens=False).input_ids[1:] # TODO better way to do this
        
        # Search for the first occurrence of the token sequence in the sentence's tokenized form
        for i in range(len(input_ids[0]) - len(mask_token_ids) + 1):
            if (input_ids[0, i : i + len(mask_token_ids)] == torch.tensor(mask_token_ids, device=input_ids.device)).all():
                mask_index = i + len(mask_token_ids)  # Start computing loss after this token
                break  # Stop at the first match

        labels[:, :mask_index] = -100  # Mask loss for all tokens before and including mask_after_token

        assert mask_index != 0

    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
    loss, logits = outputs.loss, outputs.logits
    
    # Apply softmax to the logits to get probabilities
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    # probabilities = torch.nn.functional.softmax(logits, dim=-1)

    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        if i >= mask_index - 1:  # Only include non-masked tokens
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

    if "tulu_v1" in args.task: # Need to do tuluv1 processing here
        # Applying the prompt format to tulu_v1
        for d in data:
            d["snippet"] = convert_to_tulu_v1_format(d["messages"])

    model = AutoModelForCausalLM.from_pretrained(args.target_model, device_map='auto', trust_remote_code=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)

    for d in tqdm(data):
        text = d[args.key_name]
        perplexity, log_probs, loss = get_perplexity_probs_loss(text, model, tokenizer, "<|assistant|>" if "tulu_v1" in args.task else None)
        perplexity_lower, log_probs_lower, loss_lower = get_perplexity_probs_loss(text.lower(), model, tokenizer, "<|assistant|>" if "tulu_v1" in args.task else None)

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
    parser.add_argument('--key_name', type=str, default="input", help="the key name corresponding to the input text. Selecting from: input, paraphrase")
    main(parser.parse_args())

