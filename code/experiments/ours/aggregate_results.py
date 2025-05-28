import os
import json
import pandas as pd
import argparse
from IPython import embed
import re
from tqdm import tqdm

def process_use_sentence(text):
    return text.removeprefix("useSent")[0] == "T"

def process_remove_bad(text):
    if "-" in text:
        return text[-1] == "T"
    return False
        
def model_name_to_hypers(model_name):
    model_name_split = model_name.split("_")
    return {
        "model": model_name_split[0],
        "max_tokens": int(''.join(re.findall(r'-?\d+\.?\d*',model_name_split[1]))),
        "min_tokens": int(''.join(re.findall(r'-?\d+\.?\d*',model_name_split[2]))),
        "num_sequences": int(''.join(re.findall(r'-?\d+\.?\d*',model_name_split[3]))),
        "top_p": float(''.join(re.findall(r'-?\d+\.?\d*',model_name_split[4]))),
        "temperature": float(''.join(re.findall(r'-?\d+\.?\d*',model_name_split[5]))),
        "num_sentences": int(''.join(re.findall(r'-?\d+\.?\d*',model_name_split[6]))),
        "start_sentence": int(''.join(re.findall(r'-?\d+\.?\d*',model_name_split[7]))),
        "num_words_from_end": -1 if "-" in model_name_split[8] else int(model_name_split[8].removeprefix("numWordFromEnd").split("-")[0]),
        "num_proportion_from_end": -1 if "-" not in model_name_split[8] else float(model_name_split[8].removeprefix("numWordFromEnd").split("-")[-1]),
        "max_length_to_sequence": model_name_split[9].removeprefix("maxLenSeq"),
        "use_sentence": process_use_sentence(model_name_split[10]),
        "remove_bad_first": process_remove_bad(model_name_split[10]), 
        "prompt_index": model_name_split[11].removeprefix("promptIdx"),
        "data_length": int(''.join(re.findall(r'-?\d+\.?\d*',model_name_split[12]))),
        "date": model_name_split[13],
        "min_ngram": int(model_name_split[14]), 
        "search_type": model_name_split[15],
        "creativity_min_ngram": int(re.findall(r'-?\d+\.?\d*', model_name_split[16].split("-")[0])[0]),
        "creativity_max_ngram": model_name_split[16].split("-")[1]
        }

# Define a custom sort key dynamically
def custom_sort_key(model):
    # Extract the numeric part from the model name
    try:
        size = float(re.search(r'(\d+\.?\d*)b', model).group(1))
    except:
        size = model
    return size

def main(args):
    # Define the base directory
    base_dir = f"outputs/ours/{args.task}"
    subfolders = ["train", "test", "val"]

    # Initialize rows for each metric
    rows_auc = []
    rows_tpr_01 = []
    rows_tpr_05 = []
    rows_tpr_1 = []
    rows_tpr_5 = []

    # Iterate through each data split subfolder
    for split in tqdm(subfolders, desc="high level folders"):
        split_path = os.path.join(base_dir, "scores", split)
        if not os.path.exists(split_path):
            print(f"Path does not exist: {split_path}")
            continue

        # Iterate through each model folder in the results subfolder
        for model in tqdm(os.listdir(split_path), desc="individual files"):
            model_path = os.path.join(split_path, model)
            scores_file = os.path.join(model_path, "scores.json")

            if not os.path.isfile(scores_file):
                print(f"File does not exist: {scores_file}")
                continue

            # Load the scores.json file
            with open(scores_file, "r") as f:
                scores = json.load(f)

            # Extract hyperparameters
            hypers = model_name_to_hypers(model)
            
            # Create rows for each metric
            row_auc = {"split": split}
            row_tpr_01 = {"split": split}
            row_tpr_05 = {"split": split}
            row_tpr_1 = {"split": split}
            row_tpr_5 = {"split": split}
            
            for key, value in scores.items():
                row_auc[key] = value.get("roc_auc", None)
                row_tpr_01[key] = value.get("tpr_at_1pct_fpr", None)  # 0.01%
                row_tpr_05[key] = value.get("tpr_at_50pct_fpr", None)  # 0.5%
                row_tpr_1[key] = value.get("tpr_at_100pct_fpr", None)   # 1%
                row_tpr_5[key] = value.get("tpr_at_500pct_fpr", None)   # 5%
            
            # Add hyperparameters to all rows
            row_auc.update(hypers)
            row_tpr_01.update(hypers)
            row_tpr_05.update(hypers)
            row_tpr_1.update(hypers)
            row_tpr_5.update(hypers)
            
            rows_auc.append(row_auc)
            rows_tpr_01.append(row_tpr_01)
            rows_tpr_05.append(row_tpr_05)
            rows_tpr_1.append(row_tpr_1)
            rows_tpr_5.append(row_tpr_5)

    # Create DataFrames and save to CSV
    base_output = f"outputs/ours/{args.task}/scores"
    os.makedirs(base_output, exist_ok=True)
    
    # Function to process and save a DataFrame
    def process_and_save_df(rows, metric_name):
        df = pd.DataFrame(rows).sort_values(
            by=["split", "model"],
            key=lambda col: col.map(custom_sort_key) if col.name == "model" else col
        )
        df = df[["split", "model"] + [col for col in df.columns if col != "model" and col != "split"]]
        df = df.sort_values(by=["split", "model", "prompt_index", "num_words_from_end", "num_proportion_from_end", 
                               "num_sequences", "max_length_to_sequence", "temperature", "max_tokens", 
                               "remove_bad_first", "min_ngram", "creativity_min_ngram"])
        
        output_csv = os.path.join(base_output, f"aggregated_scores_{metric_name}.csv")
        df.to_csv(output_csv, index=False)
        print(f"Aggregated {metric_name} scores saved to {output_csv}")

    # Process and save all metrics
    process_and_save_df(rows_auc, "roc_auc")
    process_and_save_df(rows_tpr_01, "tpr_at_0.01pct_fpr")
    process_and_save_df(rows_tpr_05, "tpr_at_0.5pct_fpr")
    process_and_save_df(rows_tpr_1, "tpr_at_1pct_fpr")
    process_and_save_df(rows_tpr_5, "tpr_at_5pct_fpr")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--task', type=str, default="pile_external")
    main(parser.parse_args())

    
"""
python3 -m code.experiments.ours.aggregate_results --task tulu_v1
"""