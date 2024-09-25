"""
python3 -m tasks.bookMIA.analysis
"""

import json
from unidecode import unidecode
from nltk.tokenize.casual import casual_tokenize
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import itertools
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from sklearn.metrics import roc_curve, auc, accuracy_score
from IPython import embed

LOW_CI_BOUND=3
HIGH_CI_BOUND=12
CREATIVITY_CONSTANT = HIGH_CI_BOUND - LOW_CI_BOUND
    

# Load gen_path as jsonl
def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

# Load coverage_path as a json
def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def get_ngram_coverage(text, spans, min_gram):
    try:
        tokens = casual_tokenize(unidecode(text))
    except:
        from IPython import embed; embed()
    flags = [False for _ in tokens]
    for span in spans:
        span_length = span['end_index'] - span['start_index']
        if span_length >= min_gram:
            flags[span['start_index']: span['end_index']] = [True] * span_length

    coverage = len([f for f in flags if f]) / len(flags)
    return coverage

def compute_ci_statistic(outputs, min_ngram, max_ngram, add_output_file=None, threshold=None):

    avg_coverages, avg_std = [], []
    ngram_list = list(range(min_ngram, max_ngram + 1))
    for output in outputs:
        coverages = []
        for min_ngram in ngram_list:
            coverage = get_ngram_coverage(output['text'], output['matched_spans'], min_ngram)
            coverages.append(coverage)

        avg_coverages.append(np.sum(coverages))
        avg_std.append(np.std(coverages))

        # print(f'percentage of tokens with a matched {min_ngram}-grams: {np.average(coverages):.3f}, std: {np.std(coverages):.3f}')
        #print(f'percentage of tokens outside of a matched {min_ngram}-grams: {1- np.average(coverages):.3f}, std: {np.std(coverages):.3f}')
    # auc, curvature = sum(avg_coverages), get_curvature(avg_coverages)
    # return auc, curvature

    return avg_coverages

def process_combination(params):
    prompt_idx, start_sent, num_sent, model, all_doc, aggregate = params
    
    if "gpt-3.5-turbo" in model:
        model_name = "ChatGPT"
    else:
        model_name = "GPT-4o-mini"

    # Open the jsonl
    gen_path = f"/gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/generations/{model}_maxTokens512_numSeq10_topP0.95_numSent{num_sent}_startSent{start_sent}_promptIdx{prompt_idx}_len100.jsonl"
    if not os.path.exists(gen_path):
        return

    doc_string = "alldoc" if all_doc else "onedoc"

    coverage_path = gen_path.replace(".jsonl", f"_{min_ngram}_{doc_string}.jsonl").replace("generations", "coverages")
    if not os.path.exists(coverage_path):
        return
    
    gen_data = load_jsonl(gen_path)
    coverage_data = load_json(coverage_path)
    cis = [compute_ci_statistic(cur_data, LOW_CI_BOUND, HIGH_CI_BOUND) for cur_data in tqdm(coverage_data, leave=False, desc = "Iterating through cur_data")]
    covs = [[c["coverage"] for c in cur_data] for cur_data in coverage_data]

    old_covs = covs.copy()
    old_cis = cis.copy()

    if aggregate == "mean":
        covs = np.array([np.mean(inner_list) for inner_list in old_covs])
        cis = np.array([np.mean(inner_list) for inner_list in old_cis])
    elif aggregate == "max":
        covs = np.array([np.max(inner_list) for inner_list in old_covs])
        cis = np.array([np.max(inner_list) for inner_list in old_cis])

    # Convert the CIs 
    cis = np.array([CREATIVITY_CONSTANT-c for c in cis])

    gen_labels = [g["label"] for g in gen_data]

    data_0 = [cis[i] for i in range(len(cis)) if gen_labels[i] == 0]
    data_1 = [cis[i] for i in range(len(cis)) if gen_labels[i] == 1]

    # Calculate means and medians
    mean_0, median_0, std0 = np.mean(data_0), np.median(data_0), np.std(data_0)
    mean_1, median_1, std1 = np.mean(data_1), np.median(data_1), np.std(data_1)

    # Create a violin plot with matplotlib - for Creativity Index
    plt.figure(figsize=(6, 5))
    plt.violinplot([data_0, data_1], showmeans=True)
    # Add scatter points for each category
    x_positions_0 = np.random.normal(1, 0.03, size=len(data_0))  # jitter for category 0
    x_positions_1 = np.random.normal(2, 0.03, size=len(data_1))  # jitter for category 1
    plt.scatter(x_positions_0, data_0, color='black', alpha=0.4)
    plt.scatter(x_positions_1, data_1, color='black', alpha=0.4)
    # Adding mean and median text
    plt.text(1.25, mean_0, f'Mean: {mean_0:.2f}', ha='center', va='bottom', color='blue', fontsize=10)
    plt.text(1.25, median_0, f'Median: {median_0:.2f}', ha='center', va='top', color='green', fontsize=10)
    plt.text(1.75, mean_1, f'Mean: {mean_1:.2f}', ha='center', va='bottom', color='blue', fontsize=10)
    plt.text(1.75, median_1, f'Median: {median_1:.2f}', ha='center', va='top', color='green', fontsize=10)
    # Adding labels and title
    plt.xticks([1, 2], ["Unseen Data", "Seen Data"])
    plt.ylabel(f'{aggregate} Creativity Index')
    plt.title(f'CI for BookMIA, {model_name}, min_ngram {min_ngram}, {doc_string}, prompt {prompt_idx}, agg {aggregate}')
    plt.ylim(top=min(HIGH_CI_BOUND, max(mean_0 + 2*std0, mean_1 + 2*std1)), bottom=max(min(mean_0 - 2*std0, mean_1 - 2*std1),0))
    plt.grid(alpha=0.2, axis='y')
    plt.tight_layout()

    folder_path = "/gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/plots"
    if not os.path.exists(os.path.join(folder_path, model_name)):
        os.makedirs(os.path.join(folder_path, model_name), exist_ok=True)
    plt.savefig(os.path.join(folder_path, model_name, f"promptIdx{prompt_idx}_minNgram{min_ngram}_{doc_string}_{aggregate}_numSent{num_sent}_CI{LOW_CI_BOUND}-{HIGH_CI_BOUND}.png"), dpi=200, bbox_inches="tight")

    # Plotting for Coverages
    data_0 = [covs[i] for i in range(len(covs)) if gen_labels[i] == 0]
    data_1 = [covs[i] for i in range(len(covs)) if gen_labels[i] == 1]
    # Calculate means and medians
    mean_0, median_0, std0 = np.mean(data_0), np.median(data_0), np.std(data_0)
    mean_1, median_1, std1 = np.mean(data_1), np.median(data_1), np.std(data_1)
    # Create a violin plot with matplotlib
    plt.figure(figsize=(6, 5))
    plt.violinplot([data_0, data_1], showmeans=True)
    # Add scatter points for each category
    x_positions_0 = np.random.normal(1, 0.03, size=len(data_0))  # jitter for category 0
    x_positions_1 = np.random.normal(2, 0.03, size=len(data_1))  # jitter for category 1
    plt.scatter(x_positions_0, data_0, color='black', alpha=0.4)
    plt.scatter(x_positions_1, data_1, color='black', alpha=0.4)
    # Adding mean and median text
    plt.text(1.25, mean_0, f'Mean: {mean_0:.2f}', ha='center', va='bottom', color='blue', fontsize=10)
    plt.text(1.25, median_0, f'Median: {median_0:.2f}', ha='center', va='top', color='green', fontsize=10)
    plt.text(1.75, mean_1, f'Mean: {mean_1:.2f}', ha='center', va='bottom', color='blue', fontsize=10)
    plt.text(1.75, median_1, f'Median: {median_1:.2f}', ha='center', va='top', color='green', fontsize=10)
    # Adding labels and title
    plt.xticks([1, 2], ["Unseen Data", "Seen Data"])
    plt.ylabel(f'{aggregate} Coverage')

    plt.title(f'Cov for BookMIA, {model_name}, min_ngram {min_ngram}, {doc_string}, prompt {prompt_idx}, agg {aggregate}')
    plt.ylim(top=max(mean_0 + 2*std0, mean_1 + 2*std1), bottom=0)
    plt.grid(alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, model_name, f"promptIdx{prompt_idx}_minNgram{min_ngram}_{doc_string}_{aggregate}_numSent{num_sent}_cov.png"), dpi=200, bbox_inches="tight")

    # ROC AUC curves for cov
    fpr, tpr, thresholds = roc_curve(gen_labels, covs)
    roc_auc = auc(fpr, tpr)
    # Calculate accuracy for each threshold
    accuracy_scores = []
    for threshold in thresholds:
        y_pred = np.where(covs >= threshold, 1, 0)
        accuracy_scores.append(accuracy_score(gen_labels, y_pred))
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # Diagonal line for random guess
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Cov ROC Curve for BookMIA, {model_name}, min_ngram {min_ngram}, {doc_string}, prompt {prompt_idx}, agg {aggregate}')
    plt.grid(alpha=0.15)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, model_name, f"promptIdx{prompt_idx}_minNgram{min_ngram}_{doc_string}_{aggregate}_numSent{num_sent}_cov_rocauc.png"), dpi=200, bbox_inches="tight")

    # ROC AUC curves for CI
    gen_labels = [1-g for g in gen_labels]
    fpr, tpr, thresholds = roc_curve(gen_labels, cis)
    roc_auc = auc(fpr, tpr)

    # Calculate accuracy for each threshold
    accuracy_scores = []
    for threshold in thresholds:
        y_pred = np.where(cis >= threshold, 1, 0)
        accuracy_scores.append(accuracy_score(gen_labels, y_pred))

    # Plotting the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # Diagonal line for random guess
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'CI ROC Curve for BookMIA, {model_name}, min_ngram {min_ngram}, {doc_string}, prompt {prompt_idx}, agg {aggregate}')
    plt.grid(alpha=0.15)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, model_name, f"promptIdx{prompt_idx}_minNgram{min_ngram}_{doc_string}_{aggregate}_numSent{num_sent}_CI{LOW_CI_BOUND}-{HIGH_CI_BOUND}_rocauc.png"), dpi=200, bbox_inches="tight")

if __name__ == "__main__":
    min_ngram=4
    prompt_idxs = list(range(10))
    start_sents = list(range(10))
    num_sents = list(range(10))
    models = ["gpt-3.5-turbo-0125", "gpt-4o-mini-2024-07-18"]
    all_docs = [True, False]
    aggregates = ["mean", "max"]

    combinations = list(itertools.product(prompt_idxs, start_sents, num_sents, models, all_docs, aggregates))
    num_workers = min(cpu_count(), len(combinations))  # Limit to available CPU cores or number of tasks
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_combination, combinations), total=len(combinations), desc="Processing in parallel"))

    # for combo in combinations:
    #     process_combination(combo)