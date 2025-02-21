from code.utils import save_to_jsonl, load_json
from IPython import embed
import numpy as np
import argparse
import os
from IPython import embed
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, accuracy_score

def save_plot(data_member, data_nonmember, metric, stat, save_path):
    plt.figure(figsize=(6, 4))

    # Generate x values for jitter
    x_nonmembers = np.ones(len(data_nonmember)) + np.random.normal(0, 0.05, len(data_nonmember))
    x_members = np.ones(len(data_member)) * 1.5 + np.random.normal(0, 0.05, len(data_member))

    # Scatter plot of all points
    plt.scatter(x_nonmembers, data_nonmember, label='Non-Members', color='red', alpha=0.6, zorder=3)
    plt.scatter(x_members, data_member, label='Members', color='blue', alpha=0.6, zorder=3)

    gen_labels = [0] * len(x_nonmembers) + [1] * len(x_members)
    fpr, tpr, thresholds = roc_curve(gen_labels, data_nonmember + data_member)
    roc_auc = auc(fpr, tpr)

    plt.text(1.05, max(max(data_member), max(data_nonmember)) * 0.95, 
             f'ROC AUC = {roc_auc:.2f}', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    plt.xticks([1, 1.5], ['Non-Members', 'Members'])
    plt.ylabel(f'{metric} ({stat})')
    plt.title(f'Comparison of {metric} ({stat})')
    plt.grid(alpha=0.1, zorder=-1)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()


def main(args):
    data_folder = f"outputs/ours/{args.task}/creativities/{args.split}/"
    output_folder = f"code/analysis/plots/{args.task}/{args.split}/"

    # Iterate through the files in the data folder and load_json each of the files
    for filename in tqdm(os.listdir(data_folder)):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(data_folder, filename)
            print(f"Loading {file_path}")
            
            data = load_json(file_path)

            metrics = ["coverages_gen_length", "coverages_ref_length", "coverages_total_length", "creativity"]
            member_data = {k: [] for k in metrics}
            nonmember_data = {k: [] for k in metrics}

            for d in data:
                if d["label"] == 0:
                    for metric in metrics:
                        nonmember_data[metric].append(d[metric]) # This is a list of coverage for each data point
                else:
                    for metric in metrics:
                        member_data[metric].append(d[metric]) # This is a list of coverage for each data point

            for metric in metrics:
                for stat, func in zip(["mean", "median", "max", "min"], 
                                    [np.mean, np.median, np.max, np.min]):
                    
                    # Apply the function to each inner list
                    member_stat_values = [func(x) for x in member_data[metric]]
                    nonmember_stat_values = [func(x) for x in nonmember_data[metric]]
                    
                    if member_stat_values and nonmember_stat_values:
                        save_path = os.path.join(output_folder, filename, f"{metric}_{stat}.png")
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)

                        # Pass processed statistics to the plotting function
                        save_plot(member_stat_values, nonmember_stat_values, metric, stat, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--task', type=str, default="pile_external", help="the task (dataset)")
    parser.add_argument('--split', type=str, default=None)

    main(parser.parse_args())

"""
python3 -m code.analysis.plot_ours --task bookMIA --split train
"""