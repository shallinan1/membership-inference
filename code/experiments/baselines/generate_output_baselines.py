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
    ]
}
for model in ["tulu-13b-finalized", "tulu-30b-finalized", "tulu-65b-finalized"]:
    system_prompts[model] = system_prompts["tulu-7b-finalized"]

def generate_open(text, model_args_name):
    prompt = get_prompt(text)
    max_new_tokens = 4 if model_args_name == 'LLaMA2-7B' else 2
    score_index = 2 if model_args_name == 'LLaMA2-7B' else 1
    with torch.autocast('cuda', dtype=torch.float16):
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
        try:
            outputs = model.generate(**inputs,
                                    max_new_tokens=max_new_tokens,
                                    do_sample = False,
                                    eos_token_id=model.config.eos_token_id,
                                    pad_token_id=model.config.eos_token_id,
                                    return_dict_in_generate=True, 
                                    output_scores=True,)

            try: 
                a = outputs["scores"][score_index][0][tokenizer("A").input_ids[-1]]
                b = outputs["scores"][score_index][0][tokenizer("B").input_ids[-1]]
                c = outputs["scores"][score_index][0][tokenizer("C").input_ids[-1]]
                d = outputs["scores"][score_index][0][tokenizer("D").input_ids[-1]]
            except Exception as e:
                print("Error in Probabilities")
                result = {"Text Output": "None", "A_Logit": 0, "B_Logit": 0,"C_Logit": 0, "D_Logit":0}
                return result
        except Exception as e:
            print("CUDA out of memory error, skipping here", e)
            result = {"Text Output": "None", "A_Logit": 0, "B_Logit": 0,"C_Logit": 0, "D_Logit":0}
            return result

    result = {"Text Output": "",
              "A_Logit": a,
              "B_Logit": b,
              "C_Logit": c,
              "D_Logit": d}
    return result

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

# Function to extract float values from tensors
def extract_float_values(tensor_list):
    float_values = [tensor_item.item() for tensor_item in tensor_list]
    return float_values

def process_files(data, passage_size, model):
    document = pd.DataFrame(data)
    unique_ids = document['ID'].unique().tolist()

    for i in tqdm(range(len(unique_ids))):

        document_name = unique_ids[i]

        document_aux = document[(document['ID'] == unique_ids[i])]
        document_aux = document_aux.reset_index(drop=True)
        document_aux = generate_permutations(document_df = document_aux)

        A_probabilities, B_probabilities, C_probabilities, D_probabilities, Max_Label = ([] for _ in range(5))

        parts = document_name.split('_-_')
        document_name = parts[0].replace('_', ' ')
        print(f"Starting book - {document_name}")

        if model == "ChatGPT":
            for j in tqdm(range(len(document_aux))):
                probabilities = Query_LLM(data_type = data_type, model_name=model, query_data=document_aux.iloc[j], document_name=document_name)
                A_probabilities.append(probabilities[0])
                B_probabilities.append(probabilities[1])
                C_probabilities.append(probabilities[2])
                D_probabilities.append(probabilities[3])
                Max_Label.append(mapping.get(torch.argmax(probabilities).item(), 'Unknown'))
            float_list1 = extract_float_values(A_probabilities)
            float_list2 = extract_float_values(B_probabilities)
            float_list3 = extract_float_values(C_probabilities)
            float_list4 = extract_float_values(D_probabilities)
            document_aux["A_Probability"] = float_list1
            document_aux["B_Probability"] = float_list2
            document_aux["C_Probability"] = float_list3
            document_aux["D_Probability"] = float_list4
            document_aux["Max_Label_NoDebias"] = Max_Label 

        else:
            pass
            # for j in tqdm(range(len(document_aux))):
            #     Max_Label.append(Query_LLM(data_type = data_type, model_name=model, query_data=document_aux.iloc[j], document_name=document_name))
            # document_aux["Claude2.1"] = Max_Label

        # TODO save the data

def generate_batch(texts, model, tokenizer, batch_size=50):
    embed()
    """Generates text for batches and extracts probabilities of A, B, C, D."""
    
    # Convert text prompts to tokenized inputs
    tokenized_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to("cuda")
    
    max_new_tokens = 5
    all_results = []  # Store batch results
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_inputs = {k: v[i:i + batch_size] for k, v in tokenized_inputs.items()}

            # Generate text outputs
            outputs = model.generate(**batch_inputs,
                                     max_new_tokens=max_new_tokens,
                                     do_sample=False,
                                     return_dict_in_generate=True,
                                     output_scores=True)

            embed()

            decoded_texts = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            last_token_logits = outputs.scores[-1]  # Shape: (batch_size, vocab_size)

            choices = ["A", "B", "C", "D"]
            choice_token_ids = torch.tensor(
                [tokenizer(choice, add_special_tokens=False).input_ids[-1] for choice in choices]
            ).to("cuda")  # Shape: (4,)

            # Compute softmax over vocab dimension (batched)
            probs = torch.softmax(last_token_logits, dim=-1)  # Shape: (batch_size, vocab_size)

            # Extract probabilities for "A", "B", "C", "D" (batched indexing)
            choice_probs = probs[:, choice_token_ids]  # Shape: (batch_size, 4)

            batch_results = [
                {
                    "Generated Text": decoded_texts[j],
                    "A_Prob": choice_probs[j, 0].item(),
                    "B_Prob": choice_probs[j, 1].item(),
                    "C_Prob": choice_probs[j, 2].item(),
                    "D_Prob": choice_probs[j, 3].item()
                }
                for j in range(len(decoded_texts))
            ]

            all_results.extend(batch_results)

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
    data_path = f"outputs/baselines/{args.task}/{args.split}/paraphrases/{args.paraphrase_model}.jsonl"
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
        print("Please use valid data path. See README for valid data after preprocssing/downloading.")

    if args.closed_model:
        if "claude" not in args.target_model:
            # Process bookMIA data
            pass
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

            for cur_mc_prompt, d in zip(all_mc_prompts, data):
                formatted_prompts = [convert_to_tulu_v1_open(f"{system_prompt}\n\n{c}") for c in cur_mc_prompt]
                d["decop_formatted_prompts"] = formatted_prompts
                d["decop_truth_index"] = [p["true_index"] for p in d["permutations"]]

            # Query the language model with the flattened prompts

            flattened_prompts = list(itertools.chain.from_iterable(d["decop_formatted_prompts"] for d in data))
            outputs = generate_batch(flattened_prompts, model, tokenizer, batch_size=48)

            # Unflatten the generations - into batches of 24 length each
            perm_length = factorial(args.num_paraphrases + 1)

            embed()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--target_model', type=str)
    parser.add_argument('--paraphrase_model', type=str)
    parser.add_argument('--task', type=str, default="pile_external", help="the task (dataset)")
    parser.add_argument('--split', type=str, default="train", help="the data split")
    parser.add_argument('--key_name', type=str, default="input", help="the key name corresponding to the input text. Selecting from: input, paraphrase")
    parser.add_argument('--closed_model', action="store_true")
    parser.add_argument("--sys_prompt_idx", type=int, default=0)
    parser.add_argument("--remove_bad_first", action="store_true")
    parser.add_argument("--num_paraphrases", type=int, default=3)
    main(parser.parse_args())

    """
    python3 -m code.experiments.baselines.generate_output_baselines \
    --target_model /gscratch/xlab/hallisky/cache/tulu-7b-finalized \
    --paraphrase_model gpt-4o-2024-11-20 \
    --key_name snippet \
    --task tulu_v1 \
    --split val \
    --sys_prompt_idx 0 \
    """