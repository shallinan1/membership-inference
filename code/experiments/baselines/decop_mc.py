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
import pandas as pd
from torch import nn
import torch
import itertools
import pandas as pd
from math import factorial
from code.utils import load_jsonl, save_to_jsonl, convert_to_tulu_v1_open
# from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT # Don't have this right now
from code.helper.generation.vllm_generate import ModelGenerator 
import time
from code.helper.generation.openai_parallel_generate import openai_parallel_generate, requests_limits_dict, requests_url_dict
import asyncio
import re
import random
random.seed(0)
import tiktoken

def get_token_id(model_name, text):
    enc = tiktoken.encoding_for_model(model_name)
    return enc.encode(text)

def extract_examples(text):
    pattern = r"(Example [A-Z]:\s*)(.*?)(?=\n\nExample [A-Z]:|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    examples = [match[1].removesuffix("---").removesuffix("\n\n") for match in matches]
    return examples

def extract_examples_modified(text):
    cleaned_text = re.sub(r"\*\*(Example [A-Z]:)\*\*", r"\1", text)
    
    pattern = r"(Example [A-Z]:\s*)(.*?)(?=Example [A-Z]:|$)"
    matches = re.findall(pattern, cleaned_text, re.DOTALL)
    
    examples = [match[1].removesuffix("\n\n---\n\n").removesuffix("---").strip() for match in matches]
    return examples

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
for model in ["gpt-4-0613", "gpt-4-0314", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-instruct"]:
    system_prompts[model] = system_prompts["gpt-3.5-turbo-0125"]

# TODO make this into indvidiaul functions for better style
def format_multiple_choice(task, data):
    all_mc_prompts = []
    for d in data:
        cur_mc_prompts = []
        for perm in d["permutations"]:
            cur_perm = perm["permutation"]

            if task == "bookMIA":
                extra_prompt = f"""Question: Which of the following passages is verbatim from an existing book?\nOptions:\n""" # bookMIA doesn't store this metadata cleanly
                prompt = extra_prompt +  'A. ' + cur_perm[0] + '\n' + 'B. ' + cur_perm[1] + '\n' + 'C. ' + cur_perm[2] + '\n' + 'D. ' + cur_perm[3] + '\n' + 'Answer: '    
            elif task == "tulu_v1":
                extra_prompt = f"""Question: Which of the following responses is verbatim from the prompt \"{d["user_turn"]}\"?\nOptions:\n"""
                prompt = extra_prompt +  'A. ' + cur_perm[0] + '\n' + 'B. ' + cur_perm[1] + '\n' + 'C. ' + cur_perm[2] + '\n' + 'D. ' + cur_perm[3] + '\n\n' + 'Answer: '    
            elif task == "wikiMIA":
                extra_prompt = f"""Question: Which of the following passages is verbatim from an existing Wikipedia article?\nOptions:\n"""
                prompt = extra_prompt +  'A. ' + cur_perm[0] + '\n' + 'B. ' + cur_perm[1] + '\n' + 'C. ' + cur_perm[2] + '\n' + 'D. ' + cur_perm[3] + '\n' + 'Answer: '    

            cur_mc_prompts.append(prompt)
        all_mc_prompts.append(cur_mc_prompts)
    return all_mc_prompts

def make_permutations(original, paraphrases):
    # Assign a unique identifier to each item
    items = [(original, "original")] + [(p, f"paraphrase_{i}") for i, p in enumerate(paraphrases)]
    permutations = list(itertools.permutations(items))
    result = []

    for perm in permutations:
        # Find the index of the true item in the current permutation
        true_index = next(i for i, (item, label) in enumerate(perm) if label == "original")
        perm_dict = {
            "permutation": [item for item, label in perm],
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

    bad_paraphrases = 0
    still_bad_paraphrases = 0
    bad_lengths = {k: 0 for k in [0,1,2]}
    for d in data:
        generation = d["raw_paraphrases"]
        paraphrases = extract_examples(generation)
        if len(paraphrases) != 3:
            bad_paraphrases += 1
            paraphrases = extract_examples_modified(generation)
            
            if len(paraphrases) != 3:
                still_bad_paraphrases += 1
                if len(paraphrases) == 0:
                    paraphrases = [d[args.key_name]] * 3
                    bad_lengths[0] += 1
                elif len(paraphrases) == 1:
                    paraphrases = paraphrases * 3
                    bad_lengths[1] += 1
                else:
                    paraphrases = paraphrases + random.sample(paraphrases, 1)
                    bad_lengths[2] += 1
        d["paraphrases"] = paraphrases
        d["permutations"] = make_permutations(d[args.key_name], d["paraphrases"])
    
    print(f"{bad_paraphrases} bad paraphrases out of {len(data)}")
    print(f"{still_bad_paraphrases} still bad paraphrases out of {len(data)}")
    print(bad_lengths)

    # Make the prompts
    system_prompt = system_prompts[model_name][args.sys_prompt_idx]
    all_mc_prompts = format_multiple_choice(args.task, data)

    if args.closed_model:
        if "claude" not in args.target_model:
            # Process bookMIA data
            if "bookMIA" in args.task or "wikiMIA" in args.task:
                for cur_mc_prompt, d in zip(all_mc_prompts, data):
                    formatted_prompts = [f"{system_prompt}\n{c}" for c in cur_mc_prompt]
                    d["decop_formatted_prompts"] = formatted_prompts
                    d["decop_truth_index"] = [p["true_index"] for p in d["permutations"]]

                # Query the language model with the flattened prompts
                flattened_prompts = list(itertools.chain.from_iterable(d["decop_formatted_prompts"] for d in data))
                print(len(data), len(flattened_prompts))

            # Make the requests for the API
            requests = []
            for i, prompt in enumerate(flattened_prompts):
                request_id = i
                cur_request = {
                    "model": args.target_model,
                    "max_tokens": 1,
                    "temperature": 0,
                    "seed": args.seed,
                    "n": 1, # Hardcode this
                    "metadata": {"request_id": request_id},
                }
                if "instruct" in args.target_model or "davinci" in args.target_model:
                    cur_request = cur_request | {
                        "prompt": prompt,
                        "logprobs": 10
                    }
                else:
                    cur_request = cur_request | {
                        "messages": [{"role": "user", "content": prompt}],
                        "logprobs": True,
                        "top_logprobs": 10
                    }

                requests.append(cur_request)
            
            if "instruct" not in args.target_model:
                print(f"Example prompt\n\n{requests[0]['messages'][0]['content']}")
            else:
                print(f"Example prompt\n\n{requests[0]['prompt']}")

            max_requests_per_minute = requests_limits_dict[args.target_model]["max_requests_per_minute"]
            max_tokens_per_minute = requests_limits_dict[args.target_model]["max_tokens_per_minute"]
            request_url = requests_url_dict[args.target_model]

            print(f"Using rate limits\n------\nMax requests per minute: {max_requests_per_minute}\nMax tokens per minute: {max_tokens_per_minute}")
            embed()

            full_generations = asyncio.run(openai_parallel_generate(
                requests, 
                args, 
                max_requests_per_minute=max_requests_per_minute, 
                max_tokens_per_minute=max_tokens_per_minute,
                request_url=request_url
                ))
            
            indexed_results = {}
            for result in full_generations:
                request_id = result[2]["request_id"] # Extract request_id from metadata
                indexed_results[request_id] = result[1]  # API response is the second element

            # Save the generations now in case there is an error?
            embed()

            outputs = []
            bad_gens = 0
            sum_zero = 0
            for i in range(len(full_generations)):
                try:
                    current_logprobs = indexed_results[i]["choices"][0]["logprobs"]["content"][0]["top_logprobs"] # Generation 1, Index 1
                except:
                    bad_gens += 1
                    current_logprobs = [{'token': letter, 'logprob': random.random() - 5} for letter in ["A", "B", "C", "D"]]
                current_probs_dict = {}
                for current_logprob in current_logprobs:
                    current_probs_dict[current_logprob["token"]] = np.exp(current_logprob["logprob"])
                
                probs_list = []
                for key in ["A", "B", "C", "D"]:
                    cur_value = 0.0 if key not in current_probs_dict else current_probs_dict[key]
                    probs_list.append(cur_value)
                if sum(probs_list) == 0: # Check for special case
                    sum_zero += 1

                probs_list = torch.softmax(torch.tensor(probs_list), dim=0).tolist()
                outputs.append(probs_list)
            outputs = np.array(outputs)

            # Unflatten the generations - into batches of 24 length each
            perm_length = factorial(args.num_paraphrases + 1)
            unflattened_probs = np.split(outputs, len(outputs) // perm_length)

            for d, u in zip(data, unflattened_probs):
                d["decop_probs"] = u.tolist()

            print("Bad gens", bad_gens, "\nSum Zero", sum_zero)
        else:
            pass
    else:
        generator = ModelGenerator(
            model=args.target_model,
            tokenizer=args.target_model,
            seed=args.seed,
            hf_token=args.hf_token,
            cache_dir=CACHE_PATH,
        )

        if "tulu_v1" in args.task: 
            for cur_mc_prompt, d in zip(all_mc_prompts, data):
                formatted_prompts = [convert_to_tulu_v1_open(f"{system_prompt}\n\n{c}") for c in cur_mc_prompt]
                d["decop_formatted_prompts"] = formatted_prompts
                d["decop_truth_index"] = [p["true_index"] for p in d["permutations"]]

            # Query the language model with the flattened prompts
            flattened_prompts = list(itertools.chain.from_iterable(d["decop_formatted_prompts"] for d in data))
            print(len(data), len(flattened_prompts))

            # Get tokenized letters
            letter_tokens = []
            for letter in ["A","B","C","D"]:
                letter_tokens.append(generator.tokenizer.convert_tokens_to_ids(letter))

            # Generation
            final_prompts, all_text_outputs, all_prompt_logprobs, all_output_logprobs = generator.generate_vllm(
                prompts=flattened_prompts,
                temperature=0,
                sample=False,
                max_new_tokens=2,
                min_tokens=1,
                n=1
            )

            # Get the probs
            outputs = []
            for output_logprobs in all_output_logprobs:
                current_logprobs = output_logprobs[0][0] # Generation 1, Index 1

                logprobs_list = []
                cur_keys = list(current_logprobs.keys())
                for key in cur_keys:
                    logprobs_list.append(current_logprobs[key].logprob)
                probs_list = torch.softmax(torch.tensor(logprobs_list), dim=0).tolist()
                for key, prob in zip(cur_keys, probs_list):
                    setattr(current_logprobs[key], "prob", prob)

                inner_outputs = []
                for letter_token in letter_tokens:
                    letter_token_prob = 0
                    if letter_token in current_logprobs:
                        letter_token_prob = current_logprobs[letter_token].prob
                    inner_outputs.append(letter_token_prob)
                outputs.append(inner_outputs)

            outputs = np.array(outputs)

            # Unflatten the generations - into batches of 24 length each
            perm_length = factorial(args.num_paraphrases + 1)
            unflattened_probs = np.split(outputs, len(outputs) // perm_length)

            for d, u in zip(data, unflattened_probs):
                d["decop_probs"] = u.tolist()

    output_path = data_path.replace("paraphrases", "decop_probs")
    output_path = output_path.replace(".jsonl", f"_{model_name}.jsonl")
    print(output_path)

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

    parser.add_argument("--remove_bad_first", action="store_true")
    parser.add_argument("--keep_n_sentences", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--hf_token', type=str, default=None, help='Pass in tokenizer manually. Optional.')

    main(parser.parse_args())