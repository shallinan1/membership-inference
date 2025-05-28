import argparse
from code.utils import load_jsonl, save_to_jsonl, convert_to_tulu_v1_open
from IPython import embed
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score

def main(args):
    # Load in the data
    data_path = f"outputs/baselines/{args.task}/{args.split}/decop_probs/{args.paraphrase_model}_keep{args.keep_n_sentences}.jsonl"
    if args.remove_bad_first:
        data_path = data_path.replace(".jsonl", "_remove-bad-first.jsonl")
    data_path = data_path.replace(".jsonl", f"_{args.model}.jsonl")

    if args.no_help:
        data_path = data_path.replace(".jsonl", f"_nohelp.jsonl")

    data = load_jsonl(data_path)
    all_predicted_idx = []

    score_by_label = {0: [], 1: []}
    for d in data:
        predicted_idx = np.argmax(d["decop_probs"], axis=1)
        all_predicted_idx.extend(predicted_idx.tolist())
        score = np.mean(predicted_idx == np.array(d["decop_truth_index"]))
        d["score"] = score
        score_by_label[d["label"]].append(score)
    
    gen_labels = [g["label"] for g in data]
    scores = [g["score"] for g in data]
    fpr, tpr, thresholds = roc_curve(gen_labels, scores)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Compute TPR at different FPR thresholds
    for target_fpr in [0.0001, 0.005, 0.01, 0.05]:  # 0.5%, 1%, 5%
        valid_indices = np.where(fpr <= target_fpr)[0]
        if len(valid_indices) > 0:
            idx = valid_indices[-1]  # Get the last (highest) FPR that's <= target
            tpr_at_fpr = tpr[idx]
        else:
            tpr_at_fpr = 0.0  # If no FPR <= target, set TPR to 0
        print(f"TPR at {target_fpr*100}% FPR: {tpr_at_fpr:.4f}")
    
    print("\nPredicted indices distribution:")
    print(np.unique(all_predicted_idx, return_counts=True))
    print("\nMean scores by label:")
    print(f"Non-members (0): {np.round(np.mean(score_by_label[0]),4)}")
    print(f"Members (1): {np.round(np.mean(score_by_label[1]),4)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--target_model', type=str)
    parser.add_argument('--paraphrase_model', type=str)
    parser.add_argument('--task', type=str, default="pile_external", help="the task (dataset)")
    parser.add_argument('--split', type=str, default="train", help="the data split")

    parser.add_argument("--remove_bad_first", action="store_true")
    parser.add_argument("--no_help", action="store_true")

    parser.add_argument("--keep_n_sentences", type=int, default=-1)
    parser.add_argument("--model", type=str)

    main(parser.parse_args())

    """
    python3 -m code.experiments.baselines.decop_results \
    --paraphrase_model gpt-4o-2024-11-20 \
    --task tulu_v1 \
    --split val \
    --model tulu-65b-finalized

    python3 -m code.experiments.baselines.decop_results \
    --paraphrase_model gpt-4o-2024-11-20 \
    --task tulu_v1 \
    --split val \
    --keep_n_sentences 5

    python3 -m code.experiments.baselines.decop_results \
    --paraphrase_model gpt-4o-2024-11-20 \
    --task bookMIA \
    --split val \
    --model gpt-3.5-turbo-0125 \
    --keep_n_sentences 5 \
    --remove_bad_first

    python3 -m code.experiments.baselines.decop_results \
    --paraphrase_model gpt-4o-2024-11-20 \
    --task bookMIA \
    --split train \
    --model gpt-3.5-turbo-0125 \
    --keep_n_sentences 5 \
    --remove_bad_first

    python3 -m code.experiments.baselines.decop_results \
    --paraphrase_model gpt-4o-2024-11-20 \
    --task bookMIA \
    --split train \
    --model gpt-3.5-turbo-1106 \
    --keep_n_sentences 5 \
    --remove_bad_first

    python3 -m code.experiments.baselines.decop_results \
    --paraphrase_model gpt-4o-2024-11-20 \
    --task wikiMIA \
    --split val \
    --model gpt-3.5-turbo-1106

    python3 -m code.experiments.baselines.decop_results \
    --paraphrase_model gpt-4o-2024-11-20 \
    --task wikiMIA \
    --split test \
    --model gpt-3.5-turbo-1106 
    """