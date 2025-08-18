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
from code.utils import load_jsonl, save_to_jsonl, convert_to_tulu_v1_format, remove_first_sentence_if_needed
import asyncio
from code.helper.generation.openai_parallel_generate import openai_parallel_generate, requests_limits_dict, requests_url_dict
from code.helper.generation.vllm_generate import ModelGenerator
from code.helper.generation.prompt_formatting import task_prompts_dict_book, make_prompts
import random
from datetime import datetime

def get_prefix(text: str, prefix_ratio: float) -> str:
    num_words = len(text.split())
    num_prefix_words = int(num_words * prefix_ratio)
    prefix = " ".join(text.split()[:num_prefix_words])
    return prefix

def main(args):
    # Set up random seed
    random.seed(args.seed)
    
    # Sampling parameters
    temperature = 1.0
    max_tokens = 1024
    top_k = 50
    top_p = 1.0  # For vLLM
    num_sequences = 10
    prefix_ratio = 0.5
    min_tokens = 0  # Hard-coded to 0
    
    # Get model string and check if it's valid
    model_str = args.model.split("/")[-1]
    if args.task not in task_prompts_dict_book:
        print(f"Task {args.task} not found in task_prompts_dict_book. Valid tasks are: {list(task_prompts_dict_book.keys())}")
        return
    if model_str not in task_prompts_dict_book[args.task]:
        print(f"Model {model_str} not found in task_prompts_dict_book for task {args.task}. Valid models are: {list(task_prompts_dict_book[args.task].keys())}")
        return
    
    # Load data
    data_path = f"data/{args.task}/split-random-overall/{args.split}.jsonl"
    if os.path.exists(data_path):
        data = load_jsonl(data_path)
    else:
        print("Please use valid data path. See README for valid data after preprocessing/downloading.")
        return

    if "tulu_v1" in args.task:
        for d in data:
            d["snippet"] = convert_to_tulu_v1_format(d["messages"])
    
    if args.remove_bad_first:  # Remove ill-formatted first sentence
        for d in data:
            d[args.key_name] = remove_first_sentence_if_needed(d[args.key_name])

    # Create output directory
    output_dir = f"outputs/baselines/{args.task}/{args.split}/SPL_generations"
    os.makedirs(output_dir, exist_ok=True)

    # Split input texts using prefix ratio
    prompt_texts = []
    rest_of_texts = []
    for d in data:
        full_text = d[args.key_name]
        prompt_text = get_prefix(full_text, prefix_ratio)
        rest_of_text = full_text[len(prompt_text):].strip()
        prompt_texts.append(prompt_text)
        rest_of_texts.append(rest_of_text)

    # Get the base task prompt from the dictionary
    base_task_prompt = task_prompts_dict_book[args.task][model_str][0]
    # Override just the task_prompt field
    base_task_prompt["task_prompt"] = "According to the original document, please complete the following text with more than 150 words: "

    # Format prompts using make_prompts
    prompts = make_prompts(
        prompts=prompt_texts,
        task_prompt=base_task_prompt["task_prompt"],
        task_postprompt=base_task_prompt["task_postprompt"],
        task_preprompt=base_task_prompt["task_preprompt"],
        model_name=model_str,
        prompt_key="lightest"
    )

    if args.openai:
        # OpenAI generation
        requests = []
        for i, prompt in enumerate(prompts):
            request_id = i
            cur_request = {
                "model": args.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.95,  # OpenAI specific
                "seed": args.seed,
                "n": num_sequences,
                "metadata": {"request_id": request_id},
            }
            
            if "instruct" in args.model or "davinci" in args.model:
                cur_request["prompt"] = prompt
            else:
                cur_request["messages"] = [{"role": "user", "content": prompt}]
            
            requests.append(cur_request)

        max_requests_per_minute = requests_limits_dict[args.model]["max_requests_per_minute"]
        max_tokens_per_minute = requests_limits_dict[args.model]["max_tokens_per_minute"]
        request_url = requests_url_dict[args.model]

        full_generations = asyncio.run(openai_parallel_generate(
            requests,
            max_requests_per_minute=max_requests_per_minute, 
            max_tokens_per_minute=max_tokens_per_minute,
            request_url=request_url
        ))

        # Process results
        indexed_results = {}
        unknown_id_generations = [] # Special case where the request_id is not returned
        for result in full_generations:
            try:
                request_id = result[2]["request_id"] # Extract request_id from metadata
                indexed_results[request_id] = result[1]  # API response is the second element
            except:
                unknown_id_generations.append(result[1])

        if len(unknown_id_generations) != 0:
            len_unknown = len(unknown_id_generations)
            print("Error on ids of ", len_unknown)
            for i in range(len(requests)):
                if i not in indexed_results:
                    indexed_results[i] = unknown_id_generations.pop()

        # Process results using indexed_results
        for i in range(len(requests)):
            try:
                response = indexed_results[i]
                if "instruct" in args.model or "davinci" in args.model:
                    data[i]["generation"] = [choice["text"] for choice in response["choices"]]
                else:
                    data[i]["generation"] = [choice["message"]["content"] for choice in response["choices"]]
            except:
                data[i]["generation"] = ["ERROR!"] * num_sequences

    else:
        # HuggingFace generation
        generator = ModelGenerator(
            model=args.model,
            tokenizer=args.model if not args.tokenizer else args.tokenizer,
            seed=args.seed,
            hf_token=args.hf_token,
            cache_dir=CACHE_PATH,
        )
        
        # Generate using vLLM
        final_prompts, all_text_outputs, _, _ = generator.generate_vllm(
            prompts=prompts,
            temperature=temperature,
            max_new_tokens=max_tokens,
            min_tokens=min_tokens,  # Hard-coded to 0
            top_k=top_k,
            top_p=top_p,
            n=num_sequences
        )

        # Add generations to data
        for i, generations in enumerate(all_text_outputs):
            data[i]["generation"] = generations

    # Add prompt and rest of text to data
    for i, (prompt, rest) in enumerate(zip(prompt_texts, rest_of_texts)):
        data[i]["prompt"] = prompt
        data[i]["rest_of_text"] = rest

    # Save results
    output_file = os.path.join(output_dir, f"{model_str}.jsonl")
    save_to_jsonl(data, output_file)
    print(f"Saved generations to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, required=True, help='Model to use for generation (HuggingFace model name or OpenAI model)')
    parser.add_argument('--task', type=str, default="pile_external", help='The task (dataset)')
    parser.add_argument('--split', type=str, default="train", help='The data split')
    parser.add_argument('--key_name', type=str, default="input", help='The key name corresponding to the input text')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--tokenizer', type=str, default=None, help='Optional custom tokenizer for HuggingFace models')
    parser.add_argument('--hf_token', type=str, default=None, help='HuggingFace token for private models')
    parser.add_argument('--openai', action='store_true', help='Use OpenAI API instead of HuggingFace')
    parser.add_argument('--remove_bad_first', action='store_true', help='Remove ill-formatted first sentence')
    
    main(parser.parse_args())

    """
    python -m code.experiments.baselines.SPL_generate \
        --model gpt-3.5-turbo-1106 \
        --task bookMIA \
        --split train \
        --openai \
        --key_name snippet \
        --remove_bad_first

    python -m code.experiments.baselines.SPL_generate \
        --model gpt-3.5-turbo-0125 \
        --task bookMIA \
        --split train \
        --openai \
        --key_name snippet \
        --remove_bad_first

    python -m code.experiments.baselines.SPL_generate \
        --model gpt-3.5-turbo-instruct \
        --task bookMIA \
        --split train \
        --openai \
        --key_name snippet \
        --remove_bad_first

    # WikiMIA commands
    python -m code.experiments.baselines.SPL_generate \
        --model gpt-3.5-turbo-1106 \
        --task wikiMIA \
        --split test \
        --openai \
        --remove_bad_first

    python -m code.experiments.baselines.SPL_generate \
        --model gpt-3.5-turbo-0125 \
        --task wikiMIA \
        --split test \
        --openai \
        --remove_bad_first

    python -m code.experiments.baselines.SPL_generate \
        --model gpt-3.5-turbo-instruct \
        --task wikiMIA \
        --split test \
        --openai \
        --remove_bad_first

    # HuggingFace Llama models for wikiMIA
    python -m code.experiments.baselines.SPL_generate \
        --model huggyllama/llama-7b \
        --task wikiMIA \
        --split test \
        --remove_bad_first

    python -m code.experiments.baselines.SPL_generate \
        --model huggyllama/llama-13b \
        --task wikiMIA \
        --split test \
        --remove_bad_first

    python -m code.experiments.baselines.SPL_generate \
        --model huggyllama/llama-30b \
        --task wikiMIA \
        --split test \
        --remove_bad_first

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m code.experiments.baselines.SPL_generate \
        --model huggyllama/llama-65b \
        --task wikiMIA \
        --split test \
        --remove_bad_first

    # HuggingFace Llama models for wikiMIA_hard
    python -m code.experiments.baselines.SPL_generate \
        --model huggyllama/llama-7b \
        --task wikiMIA_hard \
        --split test \
        --remove_bad_first

    python -m code.experiments.baselines.SPL_generate \
        --model huggyllama/llama-13b \
        --task wikiMIA_hard \
        --split test \
        --remove_bad_first

    python -m code.experiments.baselines.SPL_generate \
        --model huggyllama/llama-30b \
        --task wikiMIA_hard \
        --split test \
        --remove_bad_first

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m code.experiments.baselines.SPL_generate \
        --model huggyllama/llama-65b \
        --task wikiMIA_hard \
        --split test \
        --remove_bad_first

    python -m code.experiments.baselines.SPL_generate \
        --model gpt-3.5-turbo-1106 \
        --task wikiMIA_hard \
        --split test \
        --openai \
        --remove_bad_first

    python -m code.experiments.baselines.SPL_generate \
        --model gpt-3.5-turbo-0125 \
        --task wikiMIA_hard \
        --split test \
        --openai \
        --remove_bad_first

    python -m code.experiments.baselines.SPL_generate \
        --model gpt-3.5-turbo-instruct \
        --task wikiMIA_hard \
        --split test \
        --openai \
        --remove_bad_first

    python -m code.experiments.baselines.SPL_generate \
        --model gpt-4o-2024-11-20 \
        --task wikiMIA_hard \
        --split test \
        --openai \
        --remove_bad_first

    python -m code.experiments.baselines.SPL_generate \
        --model gpt-4o-mini-2024-07-18 \
        --task wikiMIA_hard \
        --split test \
        --openai \
        --remove_bad_first

    # Caution (Cost)
    python -m code.experiments.baselines.SPL_generate \
        --model gpt-4-turbo-2024-04-09 \
        --task wikiMIA_hard \
        --split test \
        --openai \
        --remove_bad_first
    
    # Tulu
    python -m code.experiments.baselines.SPL_generate \
        --model allenai/tulu-v1-llama2-7b \
        --task tulu_v1 \
        --split test \
        --key_name snippet;

    python -m code.experiments.baselines.SPL_generate \
        --model allenai/tulu-v1-llama2-13b \
        --task tulu_v1 \
        --split test \
        --key_name snippet;

    python -m code.experiments.baselines.SPL_generate \
        --model /gscratch/xlab/hallisky/cache/tulu-7b-finalized \
        --task tulu_v1 \
        --split test \
        --key_name snippet;

    python -m code.experiments.baselines.SPL_generate \
        --model /gscratch/xlab/hallisky/cache/tulu-13b-finalized \
        --task tulu_v1 \
        --split test \
        --key_name snippet;

    python -m code.experiments.baselines.SPL_generate \
        --model /gscratch/xlab/hallisky/cache/tulu-30b-finalized \
        --task tulu_v1 \
        --split test \
        --key_name snippet;

    python -m code.experiments.baselines.SPL_generate \
        --model allenai/tulu-v1-llama2-70b \
        --task tulu_v1 \
        --split test \
        --key_name snippet;

    python -m code.experiments.baselines.SPL_generate \
        --model /gscratch/xlab/hallisky/cache/tulu-65b-finalized \
        --task tulu_v1 \
        --split test \
        --key_name snippet;

    # New WikiMIA (wikiMIA_update)
    python -m code.experiments.baselines.SPL_generate \
        --model huggyllama/llama-7b \
        --task wikiMIA_update \
        --split test \
        --remove_bad_first

    python -m code.experiments.baselines.SPL_generate \
        --model huggyllama/llama-13b \
        --task wikiMIA_update \
        --split test \
        --remove_bad_first

    python -m code.experiments.baselines.SPL_generate \
        --model huggyllama/llama-30b \
        --task wikiMIA_update \
        --split test \
        --remove_bad_first

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m code.experiments.baselines.SPL_generate \
        --model huggyllama/llama-65b \
        --task wikiMIA_update \
        --split test \
        --remove_bad_first

    python -m code.experiments.baselines.SPL_generate \
        --model gpt-3.5-turbo-1106 \
        --task wikiMIA_update \
        --split test \
        --openai \
        --remove_bad_first

    python -m code.experiments.baselines.SPL_generate \
        --model gpt-3.5-turbo-0125 \
        --task wikiMIA_update \
        --split test \
        --openai \
        --remove_bad_first

    python -m code.experiments.baselines.SPL_generate \
        --model gpt-3.5-turbo-instruct \
        --task wikiMIA_update \
        --split test \
        --openai \
        --remove_bad_first

    python -m code.experiments.baselines.SPL_generate \
        --model gpt-4o-2024-11-20 \
        --task wikiMIA_update \
        --split test \
        --openai \
        --remove_bad_first

    python -m code.experiments.baselines.SPL_generate \
        --model gpt-4o-mini-2024-07-18 \
        --task wikiMIA_update \
        --split test \
        --openai \
        --remove_bad_first

    # Caution (Cost)
    python -m code.experiments.baselines.SPL_generate \
        --model gpt-4-turbo-2024-04-09 \
        --task wikiMIA_update \
        --split test \
        --openai \
        --remove_bad_first
    """