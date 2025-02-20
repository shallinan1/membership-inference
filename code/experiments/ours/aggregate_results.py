import os
import json
import pandas as pd
import argparse
from IPython import embed
import re

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
        "num_words": int(''.join(re.findall(r'-?\d+\.?\d*',model_name_split[8]))),
        "start_word": int(''.join(re.findall(r'-?\d+\.?\d*',model_name_split[9]))),
        "use_sentence": False if model_name_split[10][-1] == "F" else True, 
        "prompt_index": int(''.join(re.findall(r'-?\d+\.?\d*',model_name_split[11]))),
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

    rows = []

    # Iterate through each data split subfolder
    for split in subfolders:
        split_path = os.path.join(base_dir, "scores", split)
        if not os.path.exists(split_path):
            print(f"Path does not exist: {split_path}")
            continue

        # Iterate through each model folder in the results subfolder
        for model in os.listdir(split_path):
            model_path = os.path.join(split_path, model)
            scores_file = os.path.join(model_path, "scores.json")

            if not os.path.isfile(scores_file):
                print(f"File does not exist: {scores_file}")
                continue

            # Load the scores.json file
            with open(scores_file, "r") as f:
                scores = json.load(f)

            # Flatten and extract the data
            row = {"split": split}
            for key, value in scores.items():
                # if key == "ReferenceLoss": # TODO add reference to this
                #     for sub_key, sub_value in value.items():
                #         metric_name = f"{key}-{sub_key}"
                #         row[metric_name] = sub_value.get("roc_auc", None)
                # else:
                row[key] = value.get("roc_auc", None)
            hypers = model_name_to_hypers(model)
            row.update(hypers)
            rows.append(row)

    # Create a DataFrame and save to CSV
    output_csv = f"outputs/ours/{args.task}/scores/aggregated_scores.csv"
    df = pd.DataFrame(rows).sort_values(
        by=["split", "model"],
        key=lambda col: col.map(custom_sort_key) if col.name == "model" else col
    )
    df = df[["split", "model"] + [col for col in df.columns if col != "model" and col != "split"]]
    df.to_csv(output_csv, index=False)

    print(f"Aggregated scores saved to {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--task', type=str, default="pile_external")
    main(parser.parse_args())

    
"""
python3 -m code.experiments.ours.aggregate_results --task tulu_v1
"""