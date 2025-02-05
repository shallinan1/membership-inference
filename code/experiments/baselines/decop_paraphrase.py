from code.helper.generation.openai_generate import get_gpt_output
import argparse
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
import string

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
        for d in data:
            
            # BookMIA formatting

            # Tulu formatting

            prompt = prompt_template.substitute(ref_text=ref_text)
            get_gpt_output(prompt, 
                           model=model, 
                           temperature=args.temperature,
                           max_tokens=args.max_tokens,
                           top_p=args.top_p,
                           n=args.n
                           )
    else:
        pass

    embed()
    save_path= os.path.join(output_dir, f"{model}.jsonl")
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
    parser.add_argument("--n", type=int, default=1, help="number of paraphrases") # Reported in paper
    parser.add_argument("--max_tokens", type=int, default=256, help="max tokens to use") # Reported in paper

    main(parser.parse_args())
