"""
python3 -m tasks.bookMIA.analysis
python3 -m tasks.bookMIA.analysis --parallel
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
import argparse

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
    min_ngram, prompt_idx, start_sent, num_sent, model, all_doc, aggregate = params
    
    if "gpt-3.5-turbo" in model:
        model_name = "ChatGPT"
    elif "gpt-4o-" in model:
        model_name = "GPT-4o"
    else:
        model_name = "GPT-4o-mini"

    # Open the jsonl
    gen_path = f"/gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/generations/{model}_maxTokens512_numSeq10_topP0.95_numSent{num_sent}_startSent{start_sent}_promptIdx{prompt_idx}_len788.jsonl"
    if not os.path.exists(gen_path):
        return

    doc_string = "alldoc" if all_doc else "onedoc"

    coverage_path = gen_path.replace(".jsonl", f"_{min_ngram}_{doc_string}.jsonl").replace("generations", "coverages")
    if not os.path.exists(coverage_path):
        return
    
    try:   
        gen_data = load_jsonl(gen_path)
        coverage_data = load_json(coverage_path)
    except:
        return
    if len(gen_data) != len(coverage_data):
        return # Don't run on incomplete data
    
    gen_labels = [g["label"] for g in gen_data]
    # Collect book_ids and assign unique colors to them
    book_ids = [g["book_id"] for g in gen_data]  # Assuming `book_id` exists in each generation data
    unique_book_ids = list(set(book_ids))  # Get unique book_ids
    color_map = plt.get_cmap('tab20', int(len(unique_book_ids)/2))
    book_id_to_color = {}
    
    # book_id_to_jitter = {book_id: np.random.normal(0, 0.03) for book_id in set(unique_book_ids)} # Random jitter
    jitters = np.linspace(-0.05, 0.05, int(len(unique_book_ids)/2))
    book_id_to_jitter = {}

    label_0_indices = []
    for g, b in zip(gen_labels, book_ids):
        if g == 0:
            label_0_indices.append(b)

    label_0_idx = 0
    label_1_idx = 0
    for i, book_id in enumerate(unique_book_ids):  # Map each book_id to a color
        if book_id in label_0_indices:
            book_id_to_color[book_id] = color_map(label_0_idx)
            book_id_to_jitter[book_id] = jitters[label_0_idx]
            label_0_idx += 1
        else:
            book_id_to_color[book_id] = color_map(label_1_idx)
            book_id_to_jitter[book_id] = jitters[label_1_idx]
            label_1_idx += 1

    cis = [compute_ci_statistic(cur_data, LOW_CI_BOUND, HIGH_CI_BOUND) for cur_data in tqdm(coverage_data, leave=False, desc = "Iterating through cur_data")]
    covs = [[c["coverage"] for c in cur_data] for cur_data in coverage_data]

    if aggregate == "mean":
        covs = np.array([np.mean(inner_list) for inner_list in covs])
        cis = np.array([np.mean(inner_list) for inner_list in cis])
    elif aggregate == "max":
        covs = np.array([np.max(inner_list) for inner_list in covs])
        cis = np.array([np.max(inner_list) for inner_list in cis])
    else:
        pass
        # TODO code here to plot the distributions themself

    # Make save folders
    folder_path = "/gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/plots"
    if not os.path.exists(os.path.join(folder_path, model_name)):
        os.makedirs(os.path.join(folder_path, model_name), exist_ok=True)

    # Convert the CIs 
    cis = np.array([CREATIVITY_CONSTANT-c for c in cis])

    data_0_cis = [cis[i] for i in range(len(cis)) if gen_labels[i] == 0]
    data_1_cis = [cis[i] for i in range(len(cis)) if gen_labels[i] == 1]
    data_0_covs = [covs[i] for i in range(len(covs)) if gen_labels[i] == 0]
    data_1_covs = [covs[i] for i in range(len(covs)) if gen_labels[i] == 1]

    # TODO look into why it sometimes has a nan in it??
    # Calculate means and medians
    mean_0_cis, median_0_cis, std0_cis = np.nanmean(data_0_cis), np.nanmedian(data_0_cis), np.nanstd(data_0_cis)
    mean_1_cis, median_1_cis, std1_cis = np.nanmean(data_1_cis), np.nanmedian(data_1_cis), np.nanstd(data_1_cis)
    mean_0_covs, median_0_covs, std0_covs = np.nanmean(data_0_covs), np.nanmedian(data_0_covs), np.nanstd(data_0_covs)
    mean_1_covs, median_1_covs, std1_covs = np.nanmean(data_1_covs), np.nanmedian(data_1_covs), np.nanstd(data_1_covs)


    ### PLOT: Plotting for CIs
    plt.figure(figsize=(6, 5))
    violin_parts = plt.violinplot([data_0_cis, data_1_cis], showmeans=True)
    # Set the color and transparency
    for partname in ('bodies', 'cmeans', 'cbars', 'cmins', 'cmaxes'):
        if partname == 'bodies':
            for body in violin_parts[partname]:
                body.set_facecolor('black')
                body.set_edgecolor('black')
                body.set_alpha(0.1)
        else:
            violin_parts[partname].set_edgecolor('black')
            violin_parts[partname].set_alpha(0.1)

    # Add scatter points for each category
    x_positions_0 = np.random.normal(1, 0.03, size=len(data_0_cis))  # jitter for category 0
    x_positions_1 = np.random.normal(2, 0.03, size=len(data_1_cis))  # jitter for category 1
    plt.scatter(x_positions_0, data_0_cis, color='black', alpha=0.2)
    plt.scatter(x_positions_1, data_1_cis, color='black', alpha=0.2)
    # Adding mean and median text
    plt.text(1.25, mean_0_cis, f'Mean: {mean_0_cis:.2f}', ha='center', va='bottom', color='blue', fontsize=10)
    plt.text(1.25, median_0_cis, f'Median: {median_0_cis:.2f}', ha='center', va='top', color='green', fontsize=10)
    plt.text(1.75, mean_1_cis, f'Mean: {mean_1_cis:.2f}', ha='center', va='bottom', color='blue', fontsize=10)
    plt.text(1.75, median_1_cis, f'Median: {median_1_cis:.2f}', ha='center', va='top', color='green', fontsize=10)
    # Adding labels and title
    plt.xticks([1, 2], ["Unseen Data", "Seen Data"])
    plt.ylabel(f'{aggregate} Creativity Index')
    plt.title(f'CI for BookMIA, {model_name}, min_ngram {min_ngram}, {doc_string}, prompt {prompt_idx}, agg {aggregate}')
    plt.ylim(top=min(HIGH_CI_BOUND-LOW_CI_BOUND + 0.25, max(mean_0_cis + 2*std0_cis, mean_1_cis + 2*std1_cis)), 
             bottom=max(min(mean_0_cis - 2*std0_cis, mean_1_cis - 2*std1_cis),-0.05))
    plt.grid(alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, model_name, f"promptIdx{prompt_idx}_minNgram{min_ngram}_{doc_string}_{aggregate}_numSent{num_sent}_CI{LOW_CI_BOUND}-{HIGH_CI_BOUND}.png"), dpi=200, bbox_inches="tight")

    ### PLOT: Plotting for Coverages
    # Create a violin plot with matplotlib
    plt.figure(figsize=(6, 5))
    violin_parts = plt.violinplot([data_0_covs, data_1_covs], showmeans=True)

    # Set the color and transparency
    for partname in ('bodies', 'cmeans', 'cbars', 'cmins', 'cmaxes'):
        if partname == 'bodies':
            for body in violin_parts[partname]:
                body.set_facecolor('black')
                body.set_edgecolor('black')
                body.set_alpha(0.1)
        else:
            violin_parts[partname].set_edgecolor('black')
            violin_parts[partname].set_alpha(0.1)
    plt.scatter(x_positions_0, data_0_covs, color='black', alpha=0.2)
    plt.scatter(x_positions_1, data_1_covs, color='black', alpha=0.2)
    plt.text(1.25, mean_0_covs, f'Mean: {mean_0_covs:.2f}', ha='center', va='bottom', color='blue', fontsize=10)
    plt.text(1.25, median_0_covs, f'Median: {median_0_covs:.2f}', ha='center', va='top', color='green', fontsize=10)
    plt.text(1.75, mean_1_covs, f'Mean: {mean_1_covs:.2f}', ha='center', va='bottom', color='blue', fontsize=10)
    plt.text(1.75, median_1_covs, f'Median: {median_1_covs:.2f}', ha='center', va='top', color='green', fontsize=10)
    plt.xticks([1, 2], ["Unseen Data", "Seen Data"])
    plt.ylabel(f'{aggregate} Coverage')

    plt.title(f'Cov for BookMIA, {model_name}, min_ngram {min_ngram}, {doc_string}, prompt {prompt_idx}, agg {aggregate}')
    plt.ylim(top=min(1.05, max(mean_0_covs + 2*std0_covs, mean_1_covs + 2*std1_covs)), bottom=-0.05)
    plt.grid(alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, model_name, f"promptIdx{prompt_idx}_minNgram{min_ngram}_{doc_string}_{aggregate}_numSent{num_sent}_cov.png"), dpi=200, bbox_inches="tight")

    ### NEW PLOT: Book ID color-coded violin plot (new) (CI)
    plt.figure(figsize=(6, 5))
    violin_parts = plt.violinplot([data_0_cis, data_1_cis], showmeans=True)
    # Set the color and transparency
    for partname in ('bodies', 'cmeans', 'cbars', 'cmins', 'cmaxes'):
        if partname == 'bodies':
            for body in violin_parts[partname]:
                body.set_facecolor('black')
                body.set_edgecolor('black')
                body.set_alpha(0.1)
        else:
            violin_parts[partname].set_edgecolor('black')
            violin_parts[partname].set_alpha(0.1)
    # Add scatter points for each category, color-coded by book_id
    for i, book_id in enumerate(book_ids):
        jitter = book_id_to_jitter[book_id]  # Use the pre-computed jitter for the book_id
        color = book_id_to_color[book_id]
        if gen_labels[i] == 0:
            x_position = 1 + jitter
            plt.scatter(x_position, cis[i], color=color, alpha=0.5, label=f'Book ID {book_id}' if i == book_ids.index(book_id) else "")
        else:
            x_position = 2 + jitter
            plt.scatter(x_position, cis[i], color=color, alpha=0.5, label=f'Book ID {book_id}' if i == book_ids.index(book_id) else "")
    # Adding mean and median text 
    plt.text(1.25, mean_0_cis, f'Mean: {mean_0_cis:.2f}', ha='center', va='bottom', color='blue', fontsize=10)
    plt.text(1.25, median_0_cis, f'Median: {median_0_cis:.2f}', ha='center', va='top', color='green', fontsize=10)
    plt.text(1.75, mean_1_cis, f'Mean: {mean_1_cis:.2f}', ha='center', va='bottom', color='blue', fontsize=10)
    plt.text(1.75, median_1_cis, f'Median: {median_1_cis:.2f}', ha='center', va='top', color='green', fontsize=10)
    # Adding labels and title
    plt.xticks([1, 2], ["Unseen Data", "Seen Data"])
    plt.ylabel(f'{aggregate} Coverage')
    plt.title(f'CI BookMIA, {model_name}, min_ngram {min_ngram}, {doc_string}, prompt {prompt_idx}, agg {aggregate}')
    plt.ylim(top=min(HIGH_CI_BOUND-LOW_CI_BOUND + 0.25, max(mean_0_cis + 2*std0_cis, mean_1_cis + 2*std1_cis)), 
             bottom=max(min(mean_0_cis - 2*std0_cis, mean_1_cis - 2*std1_cis),-0.05))
    plt.grid(alpha=0.2, axis='y')
    plt.tight_layout()
    # plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1), title="Book IDs")
    plt.savefig(os.path.join(folder_path, model_name, f"promptIdx{prompt_idx}_minNgram{min_ngram}_{doc_string}_{aggregate}_numSent{num_sent}_CI{LOW_CI_BOUND}-{HIGH_CI_BOUND}_bookid.png"), dpi=200, bbox_inches="tight")

    ### NEW PLOT: Book ID color-coded violin plot (new) (coverage)
    plt.figure(figsize=(6, 5))
    violin_parts = plt.violinplot([data_0_covs, data_1_covs], showmeans=True)
    # Set the color and transparency
    for partname in ('bodies', 'cmeans', 'cbars', 'cmins', 'cmaxes'):
        if partname == 'bodies':
            for body in violin_parts[partname]:
                body.set_facecolor('black')
                body.set_edgecolor('black')
                body.set_alpha(0.1)
        else:
            violin_parts[partname].set_edgecolor('black')
            violin_parts[partname].set_alpha(0.1)
    # Add scatter points for each category, color-coded by book_id
    for i, book_id in enumerate(book_ids):
        jitter = book_id_to_jitter[book_id]  # Use the pre-computed jitter for the book_id
        color = book_id_to_color[book_id]
        if gen_labels[i] == 0:
            x_position = 1 + jitter
            plt.scatter(x_position, covs[i], color=color, alpha=0.5, label=f'Book ID {book_id}' if i == book_ids.index(book_id) else "")
        else:
            x_position = 2 + jitter
            plt.scatter(x_position, covs[i], color=color, alpha=0.5, label=f'Book ID {book_id}' if i == book_ids.index(book_id) else "")
    # Adding mean and median text 
    plt.text(1.25, mean_0_covs, f'Mean: {mean_0_covs:.2f}', ha='center', va='bottom', color='blue', fontsize=10)
    plt.text(1.25, median_0_covs, f'Median: {median_0_covs:.2f}', ha='center', va='top', color='green', fontsize=10)
    plt.text(1.75, mean_1_covs, f'Mean: {mean_1_covs:.2f}', ha='center', va='bottom', color='blue', fontsize=10)
    plt.text(1.75, median_1_covs, f'Median: {median_1_covs:.2f}', ha='center', va='top', color='green', fontsize=10)
    # Adding labels and title
    plt.xticks([1, 2], ["Unseen Data", "Seen Data"])
    plt.ylabel(f'{aggregate} Coverage')
    plt.title(f'Cov BookMIA, {model_name}, min_ngram {min_ngram}, {doc_string}, prompt {prompt_idx}, agg {aggregate}')
    plt.ylim(top=min(1.05, max(mean_0_covs + 2*std0_covs, mean_1_covs + 2*std1_covs)), bottom=-0.05)
    plt.grid(alpha=0.2, axis='y')
    plt.tight_layout()
    # plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1), title="Book IDs")
    plt.savefig(os.path.join(folder_path, model_name, f"promptIdx{prompt_idx}_minNgram{min_ngram}_{doc_string}_{aggregate}_numSent{num_sent}_cov_bookid.png"), dpi=200, bbox_inches="tight")

    ### NEW PLOT: ROC AUC curves for cov
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

    return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--parallel', action="store_true")
    args = parser.parse_args()

    min_ngrams=[4,5]
    prompt_idxs = list(range(10))
    start_sents = list(range(10))
    num_sents = list(range(10))
    models = ["gpt-3.5-turbo-0125", "gpt-4o-mini-2024-07-18"]
    models = ["gpt-3.5-turbo-0125"]
    all_docs = [True, False]
    aggregates = ["mean", "max"]

    combinations = list(itertools.product(min_ngrams, prompt_idxs, start_sents, num_sents, models, all_docs, aggregates))

    if args.parallel:
        num_workers = min(cpu_count(), len(combinations))  # Limit to available CPU cores or number of tasks
        with Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(process_combination, combinations), total=len(combinations), desc="Processing in parallel"))
    else:
        total = 0
        for combo in tqdm(combinations):
            func_output = process_combination(combo)
            total += 1
        print(total, len(combinations))