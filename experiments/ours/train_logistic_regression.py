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

# Function to calculate statistics for a given data list
def calculate_statistics(data, prefix):
    return {
        f"{prefix}_mean": np.mean(data),
        f"{prefix}_median": np.median(data),
        f"{prefix}_std": np.std(data, ddof=1) if len(data) > 1 else 0,  # Sample standard deviation
        f"{prefix}_iqr": np.percentile(data, 75) - np.percentile(data, 25),
        # f"{prefix}_skewness": skew(data), # Cannot be computed for single values
        # f"{prefix}_kurtosis": kurtosis(data), # Cannot be computed for single values
        f"{prefix}_min": np.min(data),
        f"{prefix}_max": np.max(data)
    }



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

                for bin_count in (5, 10, 20):
                    bins = np.linspace(0, 1, bin_count)
                    reg_dist = np.histogram(data, bins)[0] / len(data)

                    log_bins = np.geomspace(0.05,1.05,bin_count) - 0.05
                    log_dist = np.histogram(data, log_bins)[0] / len(data)
                    
                    combined_stats.update({f"{data_name}_regdist_bins{bin_count}": reg_dist})
                    combined_stats.update({f"{data_name}_logdist_bins{bin_count}": log_dist})

            all_stats_list.append(combined_stats)

        all_stats[split] = all_stats_list
        all_metrics[split] = {"cis": cis, "covs": covs, "gen_labels": gen_labels, "ci_path": ci_path}
    
    # Function to build a feature vector from a selected list of features in alphabetical order
    def build_feature_vector(statistics_dict, selected_features):
        # Sort the selected features alphabetically
        sorted_features = sorted(selected_features)
        feature_vector = []
        
        for feature in sorted_features:
            assert feature in statistics_dict
            # Check if the feature exists in the statistics dictionary
            value = statistics_dict[feature]
            
            # Check if the value is iterable (like a list or numpy array) and extend if so
            if isinstance(value, (list, np.ndarray)):
                feature_vector.extend(value)  # Extend the vector with elements of the iterable
            else:
                feature_vector.append(value)  # Append the scalar value

        return feature_vector

    feature_sets = [
        ["ci_mean", "ci_median", "ci_std", "ci_iqr","ci_min", "ci_max"],
        ["ci_mean", "ci_std"],
        ["ci_std"],
        ["ci_median", "ci_std", "ci_max"],
        ["ci_mean", "ci_std", "ci_max"],
        ["ci_mean", "ci_max"],
        ["cov_mean", "cov_median", "cov_std", "cov_iqr","cov_min", "cov_max"],
        ["cov_mean", "cov_std"],
        ["cov_std"],
        ["cov_median", "cov_std", "cov_max"],
        ["cov_mean", "cov_std", "cov_max"],
        ["cov_mean", "cov_max"],
        # Cov and ci
        ["cov_mean", "cov_max", "ci_mean", "ci_max"],
        ["cov_mean", "ci_mean"],
        ["cov_max", "ci_max"],
        ["cov_mean", "cov_max", "ci_mean", "ci_max", "cov_std", "ci_std"],
        ["ci_regdist_bins5"],
        ["ci_regdist_bins10"],
        ["ci_regdist_bins20"],
        ["ci_logdist_bins5"],
        ["ci_logdist_bins10"],
        ["ci_logdist_bins20"],
        ["cov_regdist_bins5"],
        ["cov_regdist_bins10"],
        ["cov_regdist_bins20"],
        ["cov_logdist_bins5"],
        ["cov_logdist_bins10"],
        ["cov_logdist_bins20"],
    ]

    results = {}
    for cur_features in tqdm(feature_sets):
        # TODO for penalty in l1, l2
        for penalty,solver in [("l1",'liblinear'), ("l2",'lbfgs')]:
            # Generate the feature vector
            feature_vector_train = np.array([build_feature_vector(cur_stats, cur_features) for cur_stats in all_stats["train"]])
            gen_labels_train = all_metrics["train"]["gen_labels"]
            log_reg = LogisticRegression(max_iter=1000, penalty=penalty, solver=solver)  # Initialize the Logistic Regression model

            # Train the model
            log_reg.fit(feature_vector_train, gen_labels_train)

            # Make predictions on the test set
            feature_vector_val = np.array([build_feature_vector(cur_stats, cur_features) for cur_stats in all_stats["val"]])
            gen_labels_val = all_metrics["val"]["gen_labels"]

            y_pred = log_reg.predict(feature_vector_val)
            y_pred_proba = log_reg.predict_proba(feature_vector_val)[:,1]

            # Evaluate the model
            accuracy = accuracy_score(gen_labels_val, y_pred)
            report = classification_report(gen_labels_val, y_pred)

            fpr, tpr, thresholds = roc_curve(gen_labels_val, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            # print("\nClassification Report:\n", report)
            results[f'{penalty}_{"-".join(sorted(cur_features))}'] = {'acc': accuracy,
                                                                      'roc_auc': roc_auc,
                                                                      'penalty': penalty,
                                                                      'features': cur_features,
                                                                      'features_coefficients': log_reg.coef_[0],
                                                                      'features_intercept': log_reg.intercept_[0]}

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
    results["baseline"] = {"acc":max_accuracy, "roc_auc": roc_auc}

    df = pd.DataFrame.from_dict(results, orient='index').reset_index()
    df.rename(columns={'index': 'model_name'}, inplace=True)

    save_file = os.path.join(args.save_path, os.path.basename(all_metrics["train"]["ci_path"]).replace(".jsonl", ".csv"))
    print(save_file)
    os.makedirs(os.path.dirname(save_file),exist_ok=True)
    df.to_csv(save_file, index=False)
    embed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_ngram', default=5)
    parser.add_argument('--all_doc', action="store_true")
    parser.add_argument('--save_path', type=str, default = "experiments/ours/bookMIA/logistic")
    parser.add_argument('--gen_path', type=str)

    main(parser.parse_args())
    """
    python3 -m experiments.ours.train_logistic_regression \
    --min_ngram 5 \
    --gen_path /gscratch/xlab/hallisky/membership-inference/experiments/ours/bookMIA/generations/train/gpt-4o-2024-05-13_maxTok512_minTok0_numSeq20_topP0.95_temp1.0_numSent3_startSent1_numWord-1_startWord-1_useSentF_promptIdx5_len494_2024-11-07-04:49:18.jsonl
    
    """