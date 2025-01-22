
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
    python3 -m code.experiments.baselines.run_baselines \
        --target_model_probs /gscratch/xlab/hallisky/membership-inference/outputs/baselines/pile_external/train/probs/pythia-1.4b.jsonl \
        --ref_model_probs /gscratch/xlab/hallisky/membership-inference/outputs/baselines/pile_external/train/probs/stablelm-base-alpha-3b-v2.jsonl;
    """

