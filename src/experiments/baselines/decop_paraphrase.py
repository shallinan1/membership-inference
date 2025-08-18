import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
CACHE_PATH = os.getenv("CACHE_PATH")

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
import asyncio
import time
from code.helper.generation.openai_parallel_generate import openai_parallel_generate
from code.experiments.ours.utils import extract_chunk_sentence

# Reported in paper
prompt_template = string.Template("""Rewrite this entire text (all sentences with no exception) expressing the same meaning using different \
words. Aim to keep the rewriting similar in length to the original text.
Do it three times. The text to be rewritten is identified as <Example A>.
Format your output as:
Example B: <insert paraphrase B>

Example C: <insert paraphrase C>

Example D: <insert paraphrase D>

-
Example A: ${ref_text}""")

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
            if "tulu_v1" in args.task:
                # This is the v2 version - paraphrasing both user and assistant response # TODO
                if False:
                    d["user_turn"] = d["messages"][0]["content"]
                    # TODO finish this
                    # d[args.key_name] = f'User: {d["messages"][0]["content"]}\nAssistant: {d["messages"][1]["content"]}'
                else:
                    # This is the v1 version - only paraphrasing the assistant response
                    d["user_turn"] = d["messages"][0]["content"]
                    d[args.key_name] = d["messages"][1]["content"]

                assert d["messages"][0]["role"] == "user"
                assert d["messages"][1]["role"] == "assistant"
            # elif "pile_external" in args.task:

            if args.remove_bad_first:
                d[args.key_name] = remove_first_sentence_if_needed(d[args.key_name])
            if args.keep_n_sentences != -1:
                d[args.key_name] = extract_chunk_sentence(d[args.key_name], 0, args.keep_n_sentences, use_last=True)[0]

        requests = []
        for i, d in enumerate(data):
            d["request_id"] = i
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
            
        print(f'Example prompt\n\n{requests[0]["messages"][0]["content"]}')
        embed()
        full_generations = asyncio.run(openai_parallel_generate(requests, args))

        indexed_results = {}
        for result in full_generations:
            request_id = result[2]["request_id"] # Extract request_id from metadata
            indexed_results[request_id] = result[1]  # API response is the second element

        for d in data: # Map results back to the original data order
            request_id = d["request_id"]
            assert request_id in indexed_results
            generation = indexed_results[request_id]["choices"][0]["message"]["content"]
            d["raw_paraphrases"] = generation
    else:
        pass

    save_path = os.path.join(output_dir, f"{model}_keep{args.keep_n_sentences}.jsonl")
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
    parser.add_argument("--keep_n_sentences", type=int, default=-1)

    main(parser.parse_args())
