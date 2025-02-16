from code.user_secrets import CACHE_PATH, OPENAI_API_KEY
from openai import OpenAI

import os
# Set up environment variables
os.environ["HF_HOME"] = CACHE_PATH
os.environ["HF_DATASETS_PATH"] = CACHE_PATH
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from code.utils import load_jsonl
import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
from code.experiments.utils import plot_roc_curve
import pandas as pd
from torch import nn
import torch
import itertools
import pandas as pd
from math import factorial
from code.utils import load_jsonl, save_to_jsonl, convert_to_tulu_v1_open
# from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT # Don't have this right now

system_prompts = {
    "tulu-7b-finalized": [
        """\
You are entering a multiple choice questions exam. You should directly answer each question by choosing the correct option. Be concise and straight to the point in your answer. Output only the letter corresponding to the correct answer.

Format your answer as '<put correct answer here>.'""",
       
       """\
You are entering a multiple choice questions exam. You should directly answer each question by choosing the correct option. Be concise and straight to the point in your answer. Output only the letter corresponding to the correct answer.

Format your answer as '<correct letter>'."""
    ],
    "gpt-3.5-turbo-0125": [
        f"""You are entering a multiple choice questions exam. You should directly answer each question by choosing the correct option. Be concise and straight to the point in your answer. Output only the letter corresponding to the correct answer."""
    ]
}
for model in ["tulu-13b-finalized", "tulu-30b-finalized", "tulu-65b-finalized"]:
    system_prompts[model] = system_prompts["tulu-7b-finalized"]

softmax = nn.Softmax(dim=0)
mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
QA_prompt = f"""You are entering a multiple choice questions exam. You should directly answer each question by choosing the correct option. Be concise and straight to the point in your answer. Output only the letter corresponding to the correct answer."""
def Query_LLM_Closed(data_type, model_name, query_data, document_name, author_name):
    
    if(data_type == "BookTection"):
        extra_prompt = f"""Question: Which of the following passages is verbatim from the \"{document_name}\" book by {author_name}?\nOptions:\n"""
    elif(data_type == "arXivTection"):
        extra_prompt = f"""Question: Which of the following passages is verbatim from the arXiv paper \"{document_name}\"?"""
    
    if model_name == "ChatGPT":
        prompt = extra_prompt + 'A. ' + query_data[0] + '\n' + 'B. ' + query_data[1] + '\n' + 'C. ' + query_data[2] + '\n' + 'D. ' + query_data[3] + '\n' + 'Answer: '
        response = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt= prompt,
            max_tokens=1,
            temperature=0,
            seed = 2319,
            logprobs = 4,
            logit_bias={32:+100, 33:+100, 34:+100, 35:+100}#Increase probabilities of tokens A,B,C,D equally, such that model answers one of those.
        )
        dict_probs = response.choices[0].logprobs.top_logprobs[0]
        logits = torch.tensor([dict_probs["A"], dict_probs["B"], dict_probs["C"], dict_probs["D"]], dtype=torch.float32)
        probabilities = softmax(logits)
        return probabilities
    else:
        prompt = QA_prompt + extra_prompt + 'A. ' + query_data[0] + '\n' + 'B. ' + query_data[1] + '\n' + 'C. ' + query_data[2] + '\n' + 'D. ' + query_data[3]
        completion = anthropic.completions.create(
            model="claude-2",
            max_tokens_to_sample=1,
            prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT} Answer: ",
            temperature=0)
        return completion.completion.strip()

# TODO make this into indvidiaul functions for better style
def format_multiple_choice(task, data):
    all_mc_prompts = []
    for d in data:
        cur_mc_prompts = []
        for perm in d["permutations"]:
            if task == "bookMIA":
                extra_prompt = f"""Question: Which of the following passages is verbatim from the book \"{d["book"].removesuffix(".txt")}\"?\nOptions:\n"""
            elif task == "tulu_v1":
                extra_prompt = f"""Question: Which of the following responses is verbatim from the prompt \"{d["user_turn"]}\"?\nOptions:\n"""

            cur_perm = perm["permutation"]
            prompt = extra_prompt +  'A. ' + cur_perm[0] + '\n' + 'B. ' + cur_perm[1] + '\n' + 'C. ' + cur_perm[2] + '\n' + 'D. ' + cur_perm[3] + '\n\n' + 'Answer:'    
            cur_mc_prompts.append(prompt)

        all_mc_prompts.append(cur_mc_prompts)
    return all_mc_prompts

def generate_batch(texts, model, tokenizer, batch_size=2, max_length=2048):
    """Generates text for batches and extracts probabilities of A, B, C, D."""
    
    max_new_tokens = 24
    all_results = []
    max_length_hit = 0

    # Iterate over the input texts in batches
    for i in tqdm(range(0, len(texts), batch_size), "Iterating over batches"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize the batch
        batch_inputs = tokenizer(batch_texts, 
                                 truncation=True, 
                                 max_length=max_length, 
                                 padding=True, 
                                 return_tensors="pt").to("cuda")
        
        if batch_inputs["input_ids"].shape == max_length:
            max_length_hit += 1

        # Generate text for the batch
        with torch.no_grad():
            outputs = model.generate(
                **batch_inputs,
                max_new_tokens=max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True
            )
        
        # Extract the logits for the generated tokens
        logits = outputs.scores  # This is a tuple of logits for each generated token
        first_new_token_logits = logits[0]  # Shape: [batch_size, vocab_size]
        
        # Get the probabilities for the tokens corresponding to A, B, C, D
        probs = torch.softmax(first_new_token_logits, dim=-1)
        token_ids = [tokenizer.convert_tokens_to_ids(token) for token in ['A', 'B', 'C', 'D']]
        
        # Extract the probabilities for A, B, C, D
        option_probs = probs[:, token_ids]  # Shape: [batch_size, 4]
        
        # Store the results
        all_results.append(option_probs.cpu().numpy())

    print(f"Max length hit: {max_length_hit}")
    # Concatenate all batch results
    all_results = np.concatenate(all_results, axis=0)
    return all_results

def make_permutations(original, paraphrases):
    items = [original] + paraphrases
    # Generate all permutations of the items
    permutations = list(itertools.permutations(items))

    result = []
    for perm in permutations:
        # Find the index of the true item in the current permutation
        true_index = perm.index(original)
        perm_dict = {
            "permutation": perm,
            "true_index": true_index
        }
        result.append(perm_dict)

    return result 

def main(args):
    data_path = f"outputs/baselines/{args.task}/{args.split}/paraphrases/{args.paraphrase_model}_keep{args.keep_n_sentences}.jsonl"
    if args.remove_bad_first:
        data_path = data_path.replace(".jsonl", "_remove-bad-first.jsonl")

    if args.closed_model:
        model_name = args.target_model
    else:
        model_name = args.target_model.split(os.sep)[-1]

    output_dir = f"outputs/baselines/{args.task}/{args.split}/probs"
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(data_path):
        data = load_jsonl(data_path)
    else:
        print(f"Data path {data_path} does not exist. Please use valid data path. See README for valid data after preprocssing/downloading.")
        import sys; sys.exit()

    bad_paraphrase_count = 0
    for d in data:
        if len(d["paraphrases"]) != 3: # Error generating paraphrases at previous step
            bad_paraphrase_count += 1
            d["paraphrases"] = [d[args.key_name]] * 3

        d["permutations"] = make_permutations(d[args.key_name], d["paraphrases"])
    print(f"Bad paraphrase count: {bad_paraphrase_count}")

    # Make the prompts
    system_prompt = system_prompts[model_name][args.sys_prompt_idx]
    all_mc_prompts = format_multiple_choice(args.task, data)

    if args.closed_model:
        if "claude" not in args.target_model:
            # Process bookMIA data
            embed()
        else:
            pass
        # elif args.target_model == "Claude": # TODO change type?
        #     claude_api_key = "Insert yout Claude key here"
        #     # anthropic = Anthropic(api_key=claude_api_key)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.target_model, device_map='auto', trust_remote_code=True)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.target_model)

        if "tulu_v1" in args.task: 
            for cur_mc_prompt, d in zip(all_mc_prompts, data):
                formatted_prompts = [convert_to_tulu_v1_open(f"{system_prompt}\n\n{c}") for c in cur_mc_prompt]
                d["decop_formatted_prompts"] = formatted_prompts
                d["decop_truth_index"] = [p["true_index"] for p in d["permutations"]]

            # Query the language model with the flattened prompts
            flattened_prompts = list(itertools.chain.from_iterable(d["decop_formatted_prompts"] for d in data))
            print(len(data), len(flattened_prompts))
            outputs = generate_batch(flattened_prompts, model, tokenizer, batch_size=args.batch_size, max_length=2048 if "tulu" in model_name else 4096)

            # Unflatten the generations - into batches of 24 length each
            perm_length = factorial(args.num_paraphrases + 1)
            unflattened_probs = np.split(outputs, len(outputs) // perm_length)

            for d, u in zip(data, unflattened_probs):
                d["decop_probs"] = u.tolist()

    output_path = data_path.replace("paraphrases", "decop_probs")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_to_jsonl(data, output_path)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--target_model', type=str)
    parser.add_argument('--paraphrase_model', type=str)
    parser.add_argument('--task', type=str, default="pile_external", help="the task (dataset)")
    parser.add_argument('--split', type=str, default="train", help="the data split")
    parser.add_argument('--key_name', type=str, default="input", help="the key name corresponding to the input text. Selecting from: input, paraphrase")
    parser.add_argument('--closed_model', action="store_true")
    parser.add_argument("--sys_prompt_idx", type=int, default=0)

    parser.add_argument("--num_paraphrases", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--remove_bad_first", action="store_true")
    parser.add_argument("--keep_n_sentences", type=int, default=-1)

    main(parser.parse_args())

    """
    python3 -m code.experiments.baselines.decop_mc \
    --target_model /gscratch/xlab/hallisky/cache/tulu-7b-finalized \
    --paraphrase_model gpt-4o-2024-11-20 \
    --key_name snippet \
    --task tulu_v1 \
    --split val \
    --sys_prompt_idx 0 \
    --batch_size 6 \
    --keep_n_sentences 5

    python3 -m code.experiments.baselines.decop_mc \
    --target_model /gscratch/xlab/hallisky/cache/tulu-13b-finalized \
    --paraphrase_model gpt-4o-2024-11-20 \
    --key_name snippet \
    --task tulu_v1 \
    --split val \
    --sys_prompt_idx 0 \
    --batch_size 6 \
    --keep_n_sentences 5

    python3 -m code.experiments.baselines.decop_mc \
    --closed_model \
    --target_model gpt-3.5-turbo-0125 \
    --paraphrase_model gpt-4o-2024-11-20 \
    --key_name snippet \
    --task bookMIA \
    --split val \
    --sys_prompt_idx 0 \
    --batch_size 6 \
    --keep_n_sentences 5
    """