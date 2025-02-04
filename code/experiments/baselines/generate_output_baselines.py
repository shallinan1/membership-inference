from datasets import load_dataset
from code.user_secrets import CACHE_PATH, OPENAI_API_KEY
from openai import OpenAI

# Set up environment variables
os.environ["HF_HOME"] = CACHE_PATH
os.environ["HF_DATASETS_PATH"] = CACHE_PATH
import argparse
from sklearn.metrics import roc_curve, auc, accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from code.utils import load_jsonl
import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
import json
from code.experiments.utils import plot_roc_curve
import pandas as pd
import sys
import os
from tqdm import tqdm
from torch import nn
import torch
# from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT # Don't have this right now

import pandas as pd
from itertools import permutations

def generate_permutations(document_df):
    variables = [0, 1, 2, 3]
    permuted_variables = list(permutations(variables))
    results = []

    for perm in permuted_variables:
        fifth_variable_position = perm.index(0)
        fifth_variable = fifth_variable_position
        result_row = list(perm) + [fifth_variable]
        results.append(result_row)

    columns = variables + ['Answer']
    base_permutations_df = pd.DataFrame(results, columns=columns)
    new_column_names = ['Example_A', 'Example_B', 'Example_C', 'Example_D', 'Answer']
    base_permutations_df.columns = new_column_names


    multiplication_factor = len(document_df)
    full_base_permutations = pd.concat([base_permutations_df] * multiplication_factor, ignore_index=True)

    new_df_aux = pd.DataFrame(index=range(len(base_permutations_df)), columns=base_permutations_df.columns[:-1])

    new_df = pd.DataFrame(index=range(0), columns=base_permutations_df.columns[:-1])
    mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    for j in range(len(document_df)):
        for i in range(len(base_permutations_df)):
            new_df_aux.at[i, 'Example_A'] = document_df.iloc[j, full_base_permutations.at[i, 'Example_A']]
            new_df_aux.at[i, 'Example_B'] = document_df.iloc[j, full_base_permutations.at[i, 'Example_B']]
            new_df_aux.at[i, 'Example_C'] = document_df.iloc[j, full_base_permutations.at[i, 'Example_C']]
            new_df_aux.at[i, 'Example_D'] = document_df.iloc[j, full_base_permutations.at[i, 'Example_D']]
            new_df_aux.at[i, 'True Answer'] = mapping[full_base_permutations.at[i, 'Answer']]
        new_df = pd.concat([new_df, new_df_aux], ignore_index=True)

    new_df['ID'] = document_df.at[0, 'ID']
    new_df['Label'] = document_df.at[0, 'Label']
    return new_df

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

def Query_LLM_Open(data_type, query_data, document_name, author_name, model_args_name):

    if(data_type == "BookTection"):
        extra_prompt = f"""Question: Which of the following passages is verbatim from the \"{document_name}\" book by {author_name}?\nOptions:\n"""
    elif(data_type == "arXivTection"):
        extra_prompt = f"""Question: Which of the following passages is verbatim from the arXiv paper \"{document_name}\"?\nOptions:\n"""
    
    prompt = extra_prompt +  'A. ' + query_data[0] + '\n' + 'B. ' + query_data[1] + '\n' + 'C. ' + query_data[2] + '\n' + 'D. ' + query_data[3] + '\n\n' + 'Answer:'    
    generated_text = generate(prompt, model_args_name)
    return generated_text


# Function to extract float values from tensors
def extract_float_values(tensor_list):
    float_values = [tensor_item.item() for tensor_item in tensor_list]
    return float_values

def process_files(data_type, passage_size, model):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if (data_type == "BookTection"):
        document = load_dataset("avduarte333/BookTection")
    else:
        document = load_dataset("avduarte333/arXivTection")

    document = pd.DataFrame(document["train"])
    unique_ids = document['ID'].unique().tolist()
    if data_type == "BookTection":
        document = document[document['Length'] == passage_size]
        document = document.reset_index(drop=True)
        
    for i in tqdm(range(len(unique_ids))):

        document_name = unique_ids[i]
        if data_type == "BookTection":
            out_dir = os.path.join(script_dir, f'DECOP_{data_type}_{passage_size}')
        else:
            out_dir = os.path.join(script_dir, f'DECOP_{data_type}')

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if data_type == "BookTection":
            fileOut = os.path.join(out_dir, f'{document_name}_Paraphrases_Oversampling_{passage_size}.xlsx')
        else:
            fileOut = os.path.join(out_dir, f'{document_name}_Paraphrases_Oversampling.xlsx')

        #Check if file was previously create (i.e. already evaluated on ChatGPT or Claude)
        if os.path.exists(fileOut):
            document_aux = pd.read_excel(fileOut)
        else:
            document_aux = document[(document['ID'] == unique_ids[i])]
            document_aux = document_aux.reset_index(drop=True)
            document_aux = generate_permutations(document_df = document_aux)


        A_probabilities, B_probabilities, C_probabilities, D_probabilities, Max_Label = ([] for _ in range(5))

        if data_type == "BookTection":
            parts = document_name.split('_-_')
            document_name = parts[0].replace('_', ' ')
            author_name = parts[1].replace('_', ' ')
            print(f"Starting book - {document_name} by {author_name}")
        else:
            author_name = ""

        if model == "ChatGPT":
            for j in tqdm(range(len(document_aux))):
                probabilities = Query_LLM(data_type = data_type, model_name=model, query_data=document_aux.iloc[j], document_name=document_name, author_name=author_name)
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
            for j in tqdm(range(len(document_aux))):
                Max_Label.append(Query_LLM(data_type = data_type, model_name=model, query_data=document_aux.iloc[j], document_name=document_name, author_name=author_name))
            document_aux["Claude2.1"] = Max_Label

        # TODO save the data

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

    
    if args.closed_model:
        if args.target_model == "ChatGPT": # TODO change this
            api_key = "Insert your OpenAI key here"
            client = OpenAI(api_key=api_key)
        elif args.target_model == "Claude": # TODO change type?
            claude_api_key = "Insert yout Claude key here"
            # anthropic = Anthropic(api_key=claude_api_key)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.target_model, device_map='auto', trust_remote_code=True)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.target_model)

    process_files(args.task, args.split, args.target_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--target_model', type=str, default="text-davinci-003", help="the model to attack: huggyllama/llama-65b, text-davinci-003")
    parser.add_argument('--task', type=str, default="pile_external", help="the task (dataset)")
    parser.add_argument('--split', type=str, default="train", help="the data split")
    parser.add_argument('--key_name', type=str, default="input", help="the key name corresponding to the input text. Selecting from: input, paraphrase")
    parser.add_argument('--closed_model', action="store_true")

    main(parser.parse_args())