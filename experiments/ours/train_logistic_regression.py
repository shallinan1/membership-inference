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
    # Assemble all the gen_paths
    gen_path_start = args.gen_path.rsplit("2024-", 1)[0]

    # Get all paths that might have the same starting filename, excluding the date
    directory = os.path.dirname(args.gen_path)
    gen_paths = []

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        # Check if the filename starts with the desired prefix
        if filename.startswith(os.path.basename(gen_path_start)):
            full_path = os.path.join(directory, filename)
            gen_paths.append(full_path)

    gen_paths = sorted(gen_paths) # Important - sort!

    # Exit if we are not the alphabetically first one to reduce redandcy
    if args.gen_path not in gen_paths[0]:
        return

    doc_string = "onedoc" if not args.all_doc else "alldoc"

    coverage_paths = [cur_path.replace(".jsonl", f"_{args.min_ngram}_{doc_string}.jsonl").replace("generations", "coverages") for cur_path in gen_paths]
    ci_paths = [cur_path.replace(".jsonl", f"_CI_{LOW_CI_BOUND}_{HIGH_CI_BOUND}.jsonl").replace("coverages", "cis") for cur_path in coverage_paths]
    for c in coverage_paths:
        assert os.path.exists(c)
    for i, c in enumerate(ci_paths):
        if i != 0:
            assert not os.path.exists(c) # Shouldn't exist!
    ci_path = ci_paths[0]

    # Initialize empty lists for combined generation data and coverage data
    combined_gen_data = []
    combined_coverage_data = []
    # Iterate through the found generation JSONL files and load their data
    for gen_path, coverage_path in zip(gen_paths, coverage_paths):
        # Load the jsonl for generation data
        gen_data = load_jsonl(gen_path)
        coverage_data = load_json(coverage_path)

        # Combine the generation data
        if not combined_gen_data:
            combined_gen_data = gen_data
        else:
            combined_gen_data = combine_list_of_dicts(combined_gen_data, gen_data)

        if not combined_coverage_data:
            combined_coverage_data = coverage_data
        else:
            combined_coverage_data = combine_lists(combined_coverage_data, coverage_data)

    combined_ci_data = load_jsonl(ci_path)

    assert len(combined_gen_data) == len(combined_coverage_data) == len(combined_ci_data)
    
    # Update all stats dict
    no_empty_gen_data = []
    no_empty_coverage_data = []
    no_empty_cis = []
    omitted = 0
    for i, c in enumerate(combined_coverage_data):
        if len(c) > 0:
            no_empty_gen_data.append(combined_gen_data[i])
            no_empty_coverage_data.append(combined_coverage_data[i])
            no_empty_cis.append(combined_ci_data[i])
        else:
            omitted += 1
    if omitted > 0:
        print(f"omitted something")

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
        ["ci_median", "ci_std", "ci_max"],
        ["ci_mean", "ci_std", "ci_max"],
        ["ci_mean", "ci_max"],
        ["cov_mean", "cov_median", "cov_std", "cov_iqr","cov_min", "cov_max"],
        ["cov_mean", "cov_std"],
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
            feature_vector = np.array([build_feature_vector(cur_stats, cur_features) for cur_stats in all_stats_list])
            log_reg = LogisticRegression(max_iter=1000, penalty=penalty, solver=solver)  # Initialize the Logistic Regression model

            # Train the model
            log_reg.fit(feature_vector, gen_labels)

            # Make predictions on the test set
            y_pred = log_reg.predict(feature_vector)
            y_pred_proba = log_reg.predict_proba(feature_vector)[:,1]

            # Evaluate the model
            accuracy = accuracy_score(gen_labels, y_pred)
            report = classification_report(gen_labels, y_pred)

            fpr, tpr, thresholds = roc_curve(gen_labels, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            # print("\nClassification Report:\n", report)
            results[f'{penalty}_{"-".join(sorted(cur_features))}'] = {}
            results[f'{penalty}_{"-".join(sorted(cur_features))}']['acc'] = accuracy
            results[f'{penalty}_{"-".join(sorted(cur_features))}']['roc_auc'] = roc_auc
            results[f'{penalty}_{"-".join(sorted(cur_features))}']['penalty'] = penalty
            results[f'{penalty}_{"-".join(sorted(cur_features))}']['features'] = cur_features
            results[f'{penalty}_{"-".join(sorted(cur_features))}']['feature_coefficients'] = log_reg.coef_[0]
            results[f'{penalty}_{"-".join(sorted(cur_features))}']['feature_intercept'] = log_reg.intercept_[0]

    fpr, tpr, thresholds = roc_curve(gen_labels, [np.mean(c) for c in covs])
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

    save_file = os.path.join(args.save_path, os.path.basename(ci_path).replace(".jsonl", ".csv"))
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