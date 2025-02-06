import os
from code.user_secrets import CACHE_PATH, OPENAI_API_KEY
# Set up environment variables
os.environ["HF_HOME"] = CACHE_PATH
os.environ["HF_DATASETS_PATH"] = CACHE_PATH
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse
import os
from code.utils import load_jsonl, save_to_jsonl, remove_first_sentence_if_needed
from IPython import embed
import torch
import string
import argparse
import re
import asyncio
import logging
import time
from code.helper.generation.api_request_parallel_processor import process_api_requests

def extract_examples(text):
    pattern = r"(Example [A-Z]:\s*)(.*?)(?=\n\nExample [A-Z]:|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    examples = [match[1].removesuffix("---").removesuffix("\n\n") for match in matches]
    return examples

# Reported in paper
prompt_template = string.Template("""Rewrite this entire text (all sentences with no exception) expressing the same meaning using different\
words. Aim to keep the rewriting similar in length to the original text.
Do it three times. The text to be rewritten is identified as <Example A>.
Format your output as:
Example B: <insert paraphrase B>

Example C: <insert paraphrase C>

Example D: <insert paraphrase D>

-
Example A: ${ref_text}""")

async def generate_paraphrases(requests, args):
    start_time = time.time()
    
    results = await process_api_requests(
        requests=requests,
        request_url="https://api.openai.com/v1/chat/completions",
        api_key=OPENAI_API_KEY, 
        max_requests_per_minute=10000, 
        max_tokens_per_minute=30000000,  
        token_encoding_name="cl100k_base",
        max_attempts=5, 
        logging_level=logging.WARNING 
    )

    end_time = time.time()
    print(f"Total generation time: {end_time - start_time:.2f} seconds for {len(requests)} samples")

    return results

def main(args):
    # Load in the data
    data_path = f"data/{args.task}/split-random-overall/{args.split}.jsonl"

    if args.closed_model:
        model = args.paraphrase_model
    else:
        model = args.paraphrase_model.split(os.sep)[-1]

    output_dir = f"outputs/baselines/{args.task}/{args.split}/paraphrases"
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(data_path):
        data = load_jsonl(data_path)
        print(f"Length of data: {len(data)}")
    else:
        print("Please use valid data path. See README for valid data after preprocssing/downloading.")

    outputs = []
    if args.closed_model and "claude" not in model:
        for d in tqdm(data, desc="Generating paraphrases"):
            if args.remove_bad_first:
                d[args.key_name] = remove_first_sentence_if_needed(d[args.key_name])

            if "bookMIA" in args.task:
                prompt = prompt_template.substitute(ref_text=d[args.key_name])
            elif "tulu_v1" in args.task:
                pass
            elif "wikiMIA" in args.task:
                pass

            # Make the requests for the API
            requests = []
            for i, d in enumerate(data):
                d["request_id"] = i
                if args.remove_bad_first:
                    d[args.key_name] = remove_first_sentence_if_needed(d[args.key_name])

                prompt = prompt_template.substitute(ref_text=d[args.key_name])
                requests.append({
                    "model": args.paraphrase_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "n": 1, # Hardcode this
                    "metadata": {"request_id": d["request_id"]},
                })

        full_generations = asyncio.run(generate_paraphrases(requests, args))

        indexed_results = {}
        for result in full_generations:
            request_id = result[2]["request_id"] # Extract request_id from metadata
            indexed_results[request_id] = result[1]  # API response is the second element

        blank_paraphrases = 0
        # Map results back to the original data order
        for d in data:
            request_id = d["request_id"]
            assert request_id in indexed_results

            generation = indexed_results[request_id]["choices"][0]["message"]["content"]
            paraphrases = extract_examples(generation)
            if len(paraphrases) == 0:
                blank_paraphrases += 1
                paraphrases = [d[args.key_name]] * 3
            d["paraphrases"] = paraphrases

        print(f"{blank_paraphrases} blank paraphrases out of {len(data)}")
    else:
        pass

    save_path= os.path.join(output_dir, f"{model}.jsonl")
    if args.remove_bad_first:
        save_path = save_path.replace(".jsonl", "_remove-bad-first.jsonl")
    save_to_jsonl(data, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--paraphrase_model', type=str, default="gpt-4o-2024-11-20", help="paraphrase model")
    parser.add_argument('--task', type=str, default="pile_external", help="the task (dataset)")
    parser.add_argument('--split', type=str, default="train", help="the data split")
    parser.add_argument('--key_name', type=str, default="input", help="the key name corresponding to the input text. Selecting from: input, paraphrase")
    parser.add_argument('--closed_model', action="store_true")
    
    # Decoding parameters
    parser.add_argument("--temperature", type=float, default=0.1) # Reported in paper
    parser.add_argument("--top_p", type=float, default=1) # Reported in paper
    parser.add_argument("--max_tokens", type=int, default=256, help="max tokens to use") # Reported in paper

    # Other hypers
    parser.add_argument("--remove_bad_first", action="store_true")

    main(parser.parse_args())
