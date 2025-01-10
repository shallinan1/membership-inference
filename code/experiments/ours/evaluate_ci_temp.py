# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
import os
from IPython import embed
from utils import load_jsonl, load_json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score
from scipy.stats import skew, kurtosis
import argparse
from utils import load_jsonl, load_json, combine_lists, combine_dicts, combine_list_of_dicts
import pandas as pd
from experiments.ours.utils import load_all_files, get_all_gen_paths, BookMIALength

LOW_CI_BOUND=3
HIGH_CI_BOUND=12
CREATIVITY_CONSTANT = HIGH_CI_BOUND - LOW_CI_BOUND + 1

def main(args):
    all_stats = {}
    all_metrics = {}

    for split in ["train", "val"]:

        cur_path = args.gen_path

        if split == "val":
            cur_path = cur_path.replace("train", split)
            cur_path = cur_path.replace(str(BookMIALength.TRAIN.value), str(BookMIALength.VAL.value))
            cur_path = get_all_gen_paths(cur_path)[0]
    
        # Load in the train files
        no_empty_gen_data, no_empty_coverage_data, no_empty_cis, ci_path = load_all_files(cur_path,
                                                                                args.all_doc,
                                                                                args.min_ngram,
                                                                                LOW_CI_BOUND,
                                                                                HIGH_CI_BOUND)

        gen_labels = [g["label"] for g in no_empty_gen_data]
        cis = [[n / CREATIVITY_CONSTANT for n in sublist] for sublist in no_empty_cis]
        covs = [[c["coverage"] for c in cur_data] for cur_data in no_empty_coverage_data]
        # TODO add lengths here as well

        # Compute the descriptive statistics and store them in a dictionary
        all_stats_list = []

        # Iterate over the zipped lists of data for each data_name
        for data_elements in zip(cis, covs):  # Add more data lists if needed
            combined_stats = {}
            for data_name, data in zip(["ci", "cov"], data_elements):  # Adjust names as needed
                combined_stats.update(calculate_statistics(data, data_name))

  
            all_stats_list.append(combined_stats)

        all_stats[split] = all_stats_list
        all_metrics[split] = {"cis": cis, "covs": covs, "gen_labels": gen_labels, "ci_path": ci_path}

    fpr, tpr, thresholds = roc_curve(gen_labels_val, [np.mean(c) for c in covs])
    roc_auc = auc(fpr, tpr)
    # Compute accuracy for each threshold
    accuracies = []
    for i, threshold in enumerate(thresholds):
        tn_rate = 1 - fpr[i]  # True Negative Rate = 1 - FPR
        fn_rate = 1 - tpr[i]  # False Negative Rate = 1 - TPR
        accuracy = (tpr[i] + tn_rate) / 2  # This is a balanced accuracy
        accuracies.append(accuracy)

    # Find the maximum accuracy and the corresponding threshold
    max_accuracy_index = np.argmax(accuracies)
    max_accuracy = accuracies[max_accuracy_index]
    results["baseline"] = {"max_acc":max_accuracy, "roc_auc": roc_auc}

    df = pd.DataFrame.from_dict(results, orient='index').reset_index()
    df.rename(columns={'index': 'model_name'}, inplace=True)

    save_file = os.path.join(args.save_path, os.path.basename(all_metrics["train"]["ci_path"]).replace(".jsonl", ".csv"))
    print(save_file)
    os.makedirs(os.path.dirname(save_file),exist_ok=True)
    df.to_csv(save_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_ngram', default=5)
    parser.add_argument('--all_doc', action="store_true")
    parser.add_argument('--save_path', type=str, default = "experiments/ours/bookMIA/logistic")
    parser.add_argument('--gen_path', type=str)

    main(parser.parse_args())
    """
    python3 -m experiments.ours.train_logistic_regression \
    --min_ngram 4 \
    --gen_path /gscratch/xlab/hallisky/membership-inference/experiments/ours/bookMIA/generations/train/gpt-4o-2024-05-13_maxTok512_minTok0_numSeq20_topP0.95_temp1.0_numSent3_startSent1_numWord-1_startWord-1_useSentF_promptIdx5_len494_2024-11-07-04:49:18.jsonl
    
    """