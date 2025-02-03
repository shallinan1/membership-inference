import pandas as pd
from oversample_labels_fn import generate_permutations
import sys
import os
from tqdm import tqdm

from torch import nn
import torch

from openai import OpenAI
# from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT # Don't have this right now
from datasets import load_dataset

from code.user_secrets import CACHE_PATH
import os
# Set up environment variables
os.environ["HF_HOME"] = CACHE_PATH
os.environ["HF_DATASETS_PATH"] = CACHE_PATH
import argparse
from sklearn.metrics import roc_curve, auc, accuracy_score
import zlib
from tqdm import tqdm
from code.utils import load_jsonl
import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
import json
from code.experiments.utils import plot_roc_curve


softmax = nn.Softmax(dim=0)
mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
QA_prompt = f"""You are entering a multiple choice questions exam. You should directly answer each question by choosing the correct option. Be concise and straight to the point in your answer. Output only the letter corresponding to the correct answer."""
def Query_LLM(data_type, model_name, query_data, document_name, author_name):
    
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



        document_aux.to_excel(fileOut, index=False)
        print(f"Completed book - {document_name}!")





if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python <name_of_file.py> --data <data_file> --target_model <model_name> [--length <passage_size>]")
        print("<passage_size> is only mandatory for BookTection and should be one of: <small>, <medium>, or <large>")
        sys.exit(1)

    data_index = sys.argv.index("--data")
    model_index = sys.argv.index("--target_model")
    
    data_type = sys.argv[data_index + 1]
    model = sys.argv[model_index + 1]

    if model == "ChatGPT":
        api_key = "Insert your OpenAI key here"
        client = OpenAI(api_key=api_key)
    elif model == "Claude":
        claude_api_key = "Insert yout Claude key here"
        anthropic = Anthropic(api_key=claude_api_key)
    else:
        print("Available models are: <ChatGPT> or <Claude>")
        sys.exit()

    if data_type == "BookTection":
        if "--length" not in sys.argv:
            print("Passage size (--length) is mandatory for BookTection data.")
            sys.exit(1)
        passage_size_index = sys.argv.index("--length")
        passage_size = sys.argv[passage_size_index + 1]

        if passage_size not in ["small", "medium", "large"]:
            print("Invalid passage_size. Available options are: <small>, <medium>, or <large>")
            sys.exit(1)
    elif data_type == "arXivTection":
        # For arXivTection data, set passage_size to a default value
        passage_size = "default_value"  # Replace with an appropriate default value
    else:
        print("Invalid data_file. Available options are: BookTection or arXivTection")
        sys.exit(1)

    process_files(data_type, passage_size, model)



# def inference(model1, tokenizer1,text, ex, modelname1):
#     p_lower, _, p_lower_likelihood = getPerplexityProbLoss(text.lower(), model1, tokenizer1, gpu=model1.device)
#     # Ratio of log ppl of lower-case and normal-case
#     pred["ppl/lowercase_ppl"] = -(np.log(p_lower) / np.log(p1)).item()
#     return ex

# TODO: need to get all log probs to run this method
# def mink_pp(log_probs, all_log_probs, ratio):
#     mu = (torch.exp(log_probs) * log_probs).sum(-1)
#     sigma = (torch.exp(log_probs) * torch.square(log_probs)).sum(-1) - ch.square(mu)
#     scores = (np.array(target _prob) - mu.numpy()) / sigma.sqrt().numpy()
    
#     return -np.mean(sorted(scores)[:int(len(scores) * k)])

def mink_attack(log_probs, ratio):
    k_length = max(int(len(log_probs)*ratio), 1)
    topk_prob = np.sort(log_probs)[:k_length]
    return -np.mean(topk_prob).item()

def zlib_attack(loss, text):
    return loss/len(zlib.compress(bytes(text, 'utf-8')))

strategies = {# "Perplexity": { "func": lambda x: -x["loss"]}, # This is the same as loss
              "Loss": {"func": lambda x: -x["loss"]},
              "Zlib": {"func": lambda x: -zlib_attack(x["loss"], x["snippet"])},
              "ReferenceLoss": {"func": lambda x, y: y - x},
              "MinK-0.05": {"func": lambda x: -mink_attack(x["log_probs"], 0.05)},
              "MinK-0.1": {"func": lambda x: -mink_attack(x["log_probs"], 0.1)},
              "MinK-0.2": {"func": lambda x: -mink_attack(x["log_probs"], 0.2)},
              "MinK-0.3": {"func": lambda x: -mink_attack(x["log_probs"], 0.3)},
              "MinK-0.4": {"func": lambda x: -mink_attack(x["log_probs"], 0.4)},
              "MinK-0.5": {"func": lambda x: -mink_attack(x["log_probs"], 0.5)},
              "MinK-0.6": {"func": lambda x: -mink_attack(x["log_probs"], 0.6)},
              }

def main(args):
    target_model_name = args.target_model_probs.split(os.sep)[-1][:-6]

    base_dir = os.path.dirname(os.path.dirname(args.target_model_probs))  # Up one level from 'probs'
    output_dir = os.path.join(base_dir, 'results', target_model_name)
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    print(f"Saving to {output_dir}")

    input_path_parts = args.target_model_probs.split(os.sep)
    dataset, split = input_path_parts[2], input_path_parts[3]

    results = load_jsonl(args.target_model_probs) # Load in the probs from file
    gen_labels = [g["label"] for g in results]

    all_scores = {}
    for strategy in strategies:
        strategy_values = strategies[strategy]

        if strategy == "ReferenceLoss":
            if args.ref_model_probs is not None:
                for ref_model_path in args.ref_model_probs:
                    ref_model_name = ref_model_path.split(os.sep)[-1][:-6]
                    if strategy not in all_scores:
                        all_scores[strategy] = {}

                    results_ref = load_jsonl(ref_model_path)
                    assert len(results_ref) == len(results)
                    assert [g["label"] for g in results] == gen_labels

                    scores = [strategy_values["func"](orig["loss"], ref["loss"]) for orig, ref in zip(results, results_ref)]

                    fpr, tpr, thresholds = roc_curve(gen_labels, scores)
                    roc_auc = auc(fpr, tpr)
                    all_scores[strategy][ref_model_name] = {}
                    all_scores[strategy][ref_model_name]["roc_auc"] = roc_auc

                    plot_title = f"{dataset} ({split}): {strategy}, {target_model_name} ({ref_model_name} ref)"
                    plot_roc_curve(fpr, tpr, roc_auc, plot_title, os.path.join(plot_dir, f"{strategy}_{ref_model_name}"))
        else:
            scores = [strategy_values["func"](r) for r in results]

            fpr, tpr, thresholds = roc_curve(gen_labels, scores)
            roc_auc = auc(fpr, tpr)
            all_scores[strategy] = {}
            all_scores[strategy]["roc_auc"] = roc_auc

            plot_title=f"{dataset} ({split}): {strategy}, {target_model_name}"
            plot_roc_curve(fpr, tpr, roc_auc, plot_title, os.path.join(plot_dir, f"{strategy}.png"))

    output_file_path = os.path.join(output_dir, f"scores.json")
    with open(output_file_path, 'w') as f:
        json.dump(all_scores, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--target_model_probs', type=str, default=None)
    parser.add_argument('--ref_model_probs', type=str, nargs="+", default=None)
    main(parser.parse_args())


    """
    python3 -m code.experiments.baselines.run_loss_baselines \
        --target_model_probs outputs/baselines/pile_external/train/probs/pythia-1.4b.jsonl \
        --ref_model_probs outputs/baselines/pile_external/train/probs/stablelm-base-alpha-3b-v2.jsonl;
    """

