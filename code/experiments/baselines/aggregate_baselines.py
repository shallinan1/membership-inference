import os
import json
import pandas as pd
import argparse
from IPython import embed

def main(args):
    # Define the base directory
    base_dir = f"outputs/baselines/{args.task}"
    subfolders = ["train", "test", "val"]

    rows = []

    # Iterate through each data split subfolder
    for split in subfolders:
        split_path = os.path.join(base_dir, split, "results")
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
            row = {"split": split, "model": model}
            for key, value in scores.items():
                if key == "ReferenceLoss":
                    for sub_key, sub_value in value.items():
                        metric_name = f"{key}-{sub_key}"
                        row[metric_name] = sub_value.get("roc_auc", None)
                else:
                    row[key] = value.get("roc_auc", None)
            rows.append(row)

    # Create a DataFrame and save to CSV
    output_csv = f"outputs/baselines/{args.task}/aggregated_scores.csv"
    df = pd.DataFrame(rows).sort_values(by=["split", "model"])
    df.to_csv(output_csv, index=False)

    print(f"Aggregated scores saved to {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--task', type=str, default="pile_external")
    main(parser.parse_args())

    
"""
python3 -m code.experiments.baselines.aggregate_baselines
"""