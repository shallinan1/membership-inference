
from user_secrets import CACHE_PATH
import os
# Set up environment variables
os.environ["HF_HOME"] = CACHE_PATH
os.environ["HF_DATASETS_PATH"] = CACHE_PATH
import argparse
from sklearn.metrics import roc_curve, auc, accuracy_score
import zlib
from tqdm import tqdm
from utils import load_jsonl
import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
import json
import torch

# def inference(model1, tokenizer1,text, ex, modelname1):

#     perplexity, probabilities, loss = getPerplexityProbLoss(text, model1, tokenizer1, gpu=model1.device)
#     p_lower, _, p_lower_likelihood = getPerplexityProbLoss(text.lower(), model1, tokenizer1, gpu=model1.device)

#     # Ratio of log ppl of large and small models
#     pred["ppl/Ref_ppl (calibrate PPL to the reference model)"] = p1_likelihood-p_ref_likelihood
 
#     # Ratio of log ppl of lower-case and normal-case
#     pred["ppl/lowercase_ppl"] = -(np.log(p_lower) / np.log(p1)).item()

#     return ex


# Function to plot ROC curve
def plot_roc_curve(fpr, tpr, roc_auc, strategy_title, save_path):
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # Diagonal line for random guess
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{strategy_title} ROC Curve')
    plt.grid(alpha=0.15)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

# TODO: need to get all log probs to run this method
# def mink_pp(log_probs, all_log_probs, ratio):
#     mu = (torch.exp(log_probs) * log_probs).sum(-1)
#     sigma = (torch.exp(log_probs) * torch.square(log_probs)).sum(-1) - ch.square(mu)
#     scores = (np.array(target_prob) - mu.numpy()) / sigma.sqrt().numpy()
    
#     return -np.mean(sorted(scores)[:int(len(scores) * k)])

def mink_attack(log_probs, ratio):
    k_length = int(len(log_probs)*ratio)
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
    dataset = input_path_parts[1]   # 'bookMIA'
    split = input_path_parts[2]     # 'val'

    # Load in the probs from file
    results = load_jsonl(args.target_model_probs)
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

                    plot_roc_curve(fpr, tpr, roc_auc, f"{dataset} ({split}): {strategy}, {target_model_name} ({ref_model_name} ref)", f"{strategy}_{ref_model_name}", plot_dir)
        else:
            scores = [strategy_values["func"](r) for r in results]

            fpr, tpr, thresholds = roc_curve(gen_labels, scores)
            roc_auc = auc(fpr, tpr)
            all_scores[strategy] = {}
            # all_scores[strategy]["fpr"] = fpr
            # all_scores[strategy]["tpr"] = tpr
            # all_scores[strategy]["thresholds"] = thresholds
            all_scores[strategy]["roc_auc"] = roc_auc

            plot_roc_curve(fpr, tpr, roc_auc, f"{dataset} ({split}): {strategy}, {target_model_name}", strategy, plot_dir)

    output_file_path = os.path.join(output_dir, f"scores.json")
    with open(output_file_path, 'w') as f:
        json.dump(all_scores, f, indent=4)

        # # Calculate accuracy for each threshold
        # accuracy_scores = []
        # for threshold in thresholds:
        #     y_pred = np.where(covs >= threshold, 1, 0)
        #     accuracy_scores.append(accuracy_score(gen_labels, y_pred))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--target_model_probs', type=str, default=None)
    parser.add_argument('--ref_model_probs', type=str, nargs="+", default=None)
    main(parser.parse_args())


    """
    python3 -m baselines.run_baselines \
        --target_model_probs baselines/bookMIA/val/probs/Llama-2-7b-hf.jsonl ;
    python3 -m baselines.run_baselines \
        --target_model_probs baselines/bookMIA/val/probs/Llama-2-70b-hf.jsonl \
        --ref_model_probs baselines/bookMIA/val/probs/Llama-2-7b-hf.jsonl \
    """

