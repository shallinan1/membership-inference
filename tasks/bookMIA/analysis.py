# TODO: something like max_docs which says only to use the first max_docs coverages/cis

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
CREATIVITY_CONSTANT = HIGH_CI_BOUND - LOW_CI_BOUND + 1
NORM_FACTOR = 1e-2

def combine_lists(list1, list2):
    output_list = []
    for l1, l2 in zip(list1, list2):
        output_list.append(l1 + l2)
    return output_list

def combine_dicts(dict1, dict2):
    for cur_key in ['logprobs', 'model']:
        if cur_key in dict1:
            dict1.pop(cur_key)
        if cur_key in dict2:
            dict2.pop(cur_key)

    # Assert that both dictionaries have the same keys
    assert dict1.keys() == dict2.keys(), f"Keys do not match: {dict1.keys()} != {dict2.keys()}"
    
    combined_dict = {}
    
    for key in dict1.keys():
        if isinstance(dict1[key], list):
            # Combine list values by concatenation
            combined_dict[key] = dict1[key] + dict2[key]
        else:
            # Ensure non-list values are equal
            assert dict1[key] == dict2[key], f"Values for '{key}' do not match: {dict1[key]} != {dict2[key]}"
            combined_dict[key] = dict1[key]
    
    return combined_dict
    
def combine_list_of_dicts(list1, list2):
    # Ensure both lists have the same length
    assert len(list1) == len(list2), f"Lists must have the same length: {len(list1)} != {len(list2)}"
    
    combined_list = []

    # Iterate through both lists simultaneously and combine each pair of dictionaries
    for dict1, dict2 in zip(list1, list2):
        combined_dict = combine_dicts(dict1, dict2)
        combined_list.append(combined_dict)
    
    return combined_list

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

ref_models_dict = {
    "Llama-3.1-70B-Instruct": "Llama-3.1-8B-Instruct"
}

def process_combination(params):
    min_ngram, prompt_idx, start_sent, num_sent, model, all_doc, aggregate, ref_prompt_idx, min_tokens, temp = params

    if "gpt-3.5-turbo" in model:
        model_name = "ChatGPT"
    elif "gpt-4o-mini" in model:
        model_name = "GPT-4o-mini"
    elif "gpt-4o" in model:
        model_name = "GPT-4o"
    elif "Llama-3.1-70B-Instruct" in model:
        model_name = "Llama-3.1-70B-Instruct"
    elif "Llama-3.1-8B-Instruct" in model:
        model_name = "Llama-3.1-8B-Instruct"
    else:
        model_name = "Default"

    # Base path to search
    gen_path_base_start = f"/gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/generations/{model}_maxTok512_minTok{min_tokens}_numSeq"
    gen_path_base_end = f"_topP0.95_temp{temp}_numSent{num_sent}_startSent{start_sent}_promptIdx{prompt_idx}_len"

    # Directory where the files are located
    gen_dir = os.path.dirname(gen_path_base_start)
    base_filename = os.path.basename(gen_path_base_start)

    # List all generation files in the directory
    gen_file_paths = [os.path.join(gen_dir, f) for f in os.listdir(gen_dir) if 
                      (f.startswith(base_filename) and f.endswith(".jsonl") and gen_path_base_end in f)]

    # Initialize empty lists for combined generation data and coverage data
    combined_gen_data = []
    combined_coverage_data = []
    
    if not gen_file_paths:
        return None
    
    doc_string = "alldoc" if all_doc else "onedoc"

    # Make save folders
    folder_path = "/gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/plots"
    ref_model = ref_models_dict.get(model, "gpt2-largeE")

    # Save path
    save_folder = os.path.join(folder_path, model_name, f"promptIdx{prompt_idx}_minNgram{min_ngram}_{doc_string}_numSent{num_sent}_{aggregate}_refModel{ref_model}_refPrompt{ref_prompt_idx}")

    # Iterate through the found generation JSONL files and load their data
    for gen_path in gen_file_paths:
        # Load the jsonl for generation data
        gen_data = load_jsonl(gen_path)

        # Get corresponding coverage file paths (assuming similar structure with date appended)
        coverage_path = gen_path.replace(".jsonl", f"_{min_ngram}_{doc_string}.jsonl").replace("generations", "coverages")

        if not os.path.exists(coverage_path):
            continue
        # Load and combine coverage data
        with open(coverage_path, 'r') as file:
            coverage_data = json.load(file)

        # Combine the generation data
        if not combined_gen_data:
            combined_gen_data = gen_data
        else:
            combined_gen_data = combine_list_of_dicts(combined_gen_data, gen_data)

        if not combined_coverage_data:
            combined_coverage_data = coverage_data
        else:
            combined_coverage_data = combine_lists(combined_coverage_data, coverage_data)

    # TODO some check here to make sure the lenghth aligns in reference model?
    # TODO do we need to check the generations? or just load coverages
    # Load in reference model path 
    ref_coverage_path_start = f"/gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/coverages/{ref_model}_maxTok512_minTok{min_tokens}_numSeq"
    ref_coverage_middle = f"_topP0.95_temp{temp}_numSent{num_sent}_startSent{start_sent}_promptIdx{ref_prompt_idx}_len"
    ref_coverage_path_end = f"_{min_ngram}_{doc_string}.jsonl"

    ref_dir = os.path.dirname(ref_coverage_path_start)
    base_filename = os.path.basename(ref_coverage_path_start)
    matching_file_paths = [os.path.join(ref_dir, f) for f in os.listdir(ref_dir) if 
                        (f.startswith(base_filename) and ref_coverage_middle in f and f.endswith(ref_coverage_path_end))]

    # TODO make folder for different reference models and save them

    # Print out matching file paths
    if matching_file_paths:
        if len(matching_file_paths) > 1:
            print("Multiple matching file paths")
        ref_coverage_path = matching_file_paths[0] # TODO use multiple reference coverages?
        with open(ref_coverage_path, 'r') as file:
            ref_coverage_data = json.load(file)

        assert len(ref_coverage_data) == len(coverage_data)
    else:
        print("No matching files found.")
        ref_coverage_data = None

    no_empty_gen_data = []
    no_empty_coverage_data = []
    no_empty_ref_coverage_data = []
    omitted = 0
    for i, c in enumerate(combined_coverage_data):
        if len(c) > 0:
            no_empty_gen_data.append(combined_gen_data[i])
            no_empty_coverage_data.append(combined_coverage_data[i])
            if ref_coverage_data is not None:
                cur_ref_coverage_data = ref_coverage_data[i]
                if len(cur_ref_coverage_data) == 0:
                    cur_ref_coverage_data = [{'text': "text", 'matched_spans': [], 'coverage': 0, 'avg_span_len': 0}]
                no_empty_ref_coverage_data.append(cur_ref_coverage_data)
        else:
            omitted += 1
    if omitted > 0:
#        pass
        print(f"omitted something {save_folder}; Old Length: {len(combined_gen_data)}, new length: {len(no_empty_gen_data)}. Number omitted {omitted}")

    gen_data = no_empty_gen_data
    coverage_data = no_empty_coverage_data
    ref_coverage_data = no_empty_ref_coverage_data

    if len(gen_data) == 0 or len(gen_data) != len(coverage_data):
        print(f"Invalid length for {gen_path}. Len gen_data {len(gen_data)}. Len coverage data {len(coverage_data)}")
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

    cis = [compute_ci_statistic(cur_data, LOW_CI_BOUND, HIGH_CI_BOUND) for cur_data in tqdm(coverage_data, leave=False, desc = "Iterating through original data", position=1)]
    covs = [[c["coverage"] for c in cur_data] for cur_data in coverage_data]

    if aggregate == "mean":
        covs = np.array([np.mean(inner_list) for inner_list in covs])
        cis = np.array([np.mean(inner_list) for inner_list in cis])
    elif aggregate == "max":
        covs = np.array([np.max(inner_list) for inner_list in covs])
        cis = np.array([np.max(inner_list) for inner_list in cis])
    else:
        pass # TODO code here to plot the distributions themself

    if len(ref_coverage_data) == len(coverage_data):
        ref_cis = [compute_ci_statistic(cur_data, LOW_CI_BOUND, HIGH_CI_BOUND) for cur_data in tqdm(ref_coverage_data, leave=False, desc = "Iterating through ref data", position=1)]
        ref_covs =  [[c["coverage"] for c in cur_data] for cur_data in ref_coverage_data]
        if aggregate == "mean": # TODO separate param here?
            ref_covs = np.array([np.mean(inner_list) for inner_list in ref_covs])
            ref_cis = np.array([np.mean(inner_list) for inner_list in ref_cis])
        elif aggregate == "max":
            ref_covs = np.array([np.max(inner_list) for inner_list in ref_covs])
            ref_cis = np.array([np.max(inner_list) for inner_list in ref_cis])
    
        # Now, make a normalized version - subtract?
        normalized_covs_subtract = np.maximum(covs - ref_covs, 0)
        normalized_cis_subtract = CREATIVITY_CONSTANT - np.maximum(cis - ref_cis, 0)
        
        normalized_covs_divide = covs / (ref_covs + NORM_FACTOR)
        normalized_cis_divide = np.minimum(np.array([CREATIVITY_CONSTANT-c for c in cis]) / (np.array([CREATIVITY_CONSTANT-c for c in ref_cis]) + NORM_FACTOR), 1)

        normalied_covs_multiply = covs * (1 - ref_covs)
        normalized_cis_multiply = cis * (np.array([CREATIVITY_CONSTANT-c for c in cis])/CREATIVITY_CONSTANT)

        ref_cis = np.array([CREATIVITY_CONSTANT-c for c in ref_cis]) # Convert

    cis = np.array([CREATIVITY_CONSTANT-c for c in cis]) # Convert 

    data_0_cis = [cis[i] for i in range(len(cis)) if gen_labels[i] == 0]
    data_1_cis = [cis[i] for i in range(len(cis)) if gen_labels[i] == 1]
    data_0_covs = [covs[i] for i in range(len(covs)) if gen_labels[i] == 0]
    data_1_covs = [covs[i] for i in range(len(covs)) if gen_labels[i] == 1]

    # TODO look into why it sometimes has a nan in it??
    mean_0_cis, median_0_cis, std0_cis = np.nanmean(data_0_cis), np.nanmedian(data_0_cis), np.nanstd(data_0_cis)
    mean_1_cis, median_1_cis, std1_cis = np.nanmean(data_1_cis), np.nanmedian(data_1_cis), np.nanstd(data_1_cis)
    mean_0_covs, median_0_covs, std0_covs = np.nanmean(data_0_covs), np.nanmedian(data_0_covs), np.nanstd(data_0_covs)
    mean_1_covs, median_1_covs, std1_covs = np.nanmean(data_1_covs), np.nanmedian(data_1_covs), np.nanstd(data_1_covs)
    
    if len(ref_coverage_data) == len(coverage_data):
        # Now, do same for reference ones
        data_0_ref_cis = [ref_cis[i] for i in range(len(ref_cis)) if gen_labels[i] == 0]
        data_1_ref_cis = [ref_cis[i] for i in range(len(ref_cis)) if gen_labels[i] == 1]
        data_0_ref_covs = [ref_covs[i] for i in range(len(ref_covs)) if gen_labels[i] == 0]
        data_1_ref_covs = [ref_covs[i] for i in range(len(ref_covs)) if gen_labels[i] == 1]

        mean_0_ref_cis, median_0_ref_cis, std0_ref_cis = np.nanmean(data_0_ref_cis), np.nanmedian(data_0_ref_cis), np.nanstd(data_0_ref_cis)
        mean_1_ref_cis, median_1_ref_cis, std1_ref_cis = np.nanmean(data_1_ref_cis), np.nanmedian(data_1_ref_cis), np.nanstd(data_1_ref_cis)
        mean_0_ref_covs, median_0_ref_covs, std0_ref_covs = np.nanmean(data_0_ref_covs), np.nanmedian(data_0_ref_covs), np.nanstd(data_0_ref_covs)
        mean_1_ref_covs, median_1_ref_covs, std1_ref_covs = np.nanmean(data_1_ref_covs), np.nanmedian(data_1_ref_covs), np.nanstd(data_1_ref_covs)

        # Same for normalized ones
        data_0_normalized_cis_subtract = [normalized_cis_subtract[i] for i in range(len(normalized_cis_subtract)) if gen_labels[i] == 0]
        data_1_normalized_cis_subtract = [normalized_cis_subtract[i] for i in range(len(normalized_cis_subtract)) if gen_labels[i] == 1]
        data_0_normalized_covs_subtract = [normalized_covs_subtract[i] for i in range(len(normalized_covs_subtract)) if gen_labels[i] == 0]
        data_1_normalized_covs_subtract = [normalized_covs_subtract[i] for i in range(len(normalized_covs_subtract)) if gen_labels[i] == 1]

        mean_0_normalized_cis_subtract, median_0_normalized_cis_subtract, std0_normalized_cis_subtract = np.nanmean(data_0_normalized_cis_subtract), np.nanmedian(data_0_normalized_cis_subtract), np.nanstd(data_0_normalized_cis_subtract)
        mean_1_normalized_cis_subtract, median_1_normalized_cis_subtract, std1_normalized_cis_subtract = np.nanmean(data_1_normalized_cis_subtract), np.nanmedian(data_1_normalized_cis_subtract), np.nanstd(data_1_normalized_cis_subtract)
        mean_0_normalized_covs_subtract, median_0_normalized_covs_subtract, std0_normalized_covs_subtract = np.nanmean(data_0_normalized_covs_subtract), np.nanmedian(data_0_normalized_covs_subtract), np.nanstd(data_0_normalized_covs_subtract)
        mean_1_normalized_covs_subtract, median_1_normalized_covs_subtract, std1_normalized_covs_subtract = np.nanmean(data_1_normalized_covs_subtract), np.nanmedian(data_1_normalized_covs_subtract), np.nanstd(data_1_normalized_covs_subtract)
        
        data_0_normalized_cis_divide = [normalized_cis_divide[i] for i in range(len(normalized_cis_divide)) if gen_labels[i] == 0]
        data_1_normalized_cis_divide = [normalized_cis_divide[i] for i in range(len(normalized_cis_divide)) if gen_labels[i] == 1]
        data_0_normalized_covs_divide = [normalized_covs_divide[i] for i in range(len(normalized_covs_divide)) if gen_labels[i] == 0]
        data_1_normalized_covs_divide = [normalized_covs_divide[i] for i in range(len(normalized_covs_divide)) if gen_labels[i] == 1]

        mean_0_normalized_cis_divide, median_0_normalized_cis_divide, std0_normalized_cis_divide = np.nanmean(data_0_normalized_cis_divide), np.nanmedian(data_0_normalized_cis_divide), np.nanstd(data_0_normalized_cis_divide)
        mean_1_normalized_cis_divide, median_1_normalized_cis_divide, std1_normalized_cis_divide = np.nanmean(data_1_normalized_cis_divide), np.nanmedian(data_1_normalized_cis_divide), np.nanstd(data_1_normalized_cis_divide)
        mean_0_normalized_covs_divide, median_0_normalized_covs_divide, std0_normalized_covs_divide = np.nanmean(data_0_normalized_covs_divide), np.nanmedian(data_0_normalized_covs_divide), np.nanstd(data_0_normalized_covs_divide)
        mean_1_normalized_covs_divide, median_1_normalized_covs_divide, std1_normalized_covs_divide = np.nanmean(data_1_normalized_covs_divide), np.nanmedian(data_1_normalized_covs_divide), np.nanstd(data_1_normalized_covs_divide)
        
    if np.nan in data_0_cis or np.nan in data_1_cis:
        print(f"NAN FOUND {save_folder}") # This should not happen

    # Make save folder if we finished everything succesfully
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

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
    plt.title(f'CI, BookMIA, {model_name}, min_ngram {min_ngram}, {doc_string}, prompt {prompt_idx}, agg {aggregate}')
    plt.ylim(top=min(CREATIVITY_CONSTANT + 0.25, max(mean_0_cis + 2*std0_cis, mean_1_cis + 2*std1_cis)), 
             bottom=max(min(mean_0_cis - 2*std0_cis, mean_1_cis - 2*std1_cis),-0.05))
    plt.grid(alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"CI{LOW_CI_BOUND}-{HIGH_CI_BOUND}_orig.png"), dpi=200, bbox_inches="tight")

    if len(ref_coverage_data) == len(coverage_data):
        plt.figure(figsize=(6, 5))
        violin_parts = plt.violinplot([data_0_ref_cis, data_1_ref_cis], showmeans=True)
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
        plt.scatter(x_positions_0, data_0_ref_cis, color='black', alpha=0.2)
        plt.scatter(x_positions_1, data_1_ref_cis, color='black', alpha=0.2)
        # Adding mean and median text
        plt.text(1.25, mean_0_ref_cis, f'Mean: {mean_0_ref_cis:.2f}', ha='center', va='bottom', color='blue', fontsize=10)
        plt.text(1.25, median_0_ref_cis, f'Median: {median_0_ref_cis:.2f}', ha='center', va='top', color='green', fontsize=10)
        plt.text(1.75, mean_1_ref_cis, f'Mean: {mean_1_ref_cis:.2f}', ha='center', va='bottom', color='blue', fontsize=10)
        plt.text(1.75, median_1_ref_cis, f'Median: {median_1_ref_cis:.2f}', ha='center', va='top', color='green', fontsize=10)
        # Adding labels and title
        plt.xticks([1, 2], ["Unseen Data", "Seen Data"])
        plt.ylabel(f'{aggregate} Creativity Index')
        plt.title(f'CI, BookMIA, {model_name}, min_ngram {min_ngram}, {doc_string}, prompt {prompt_idx}, agg {aggregate}')
        plt.ylim(top=min(CREATIVITY_CONSTANT + 0.25, max(mean_0_ref_cis + 2*std0_ref_cis, mean_1_ref_cis + 2*std1_ref_cis)), 
                bottom=max(min(mean_0_ref_cis - 2*std0_ref_cis, mean_1_ref_cis - 2*std1_ref_cis),-0.05))
        plt.grid(alpha=0.2, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"CI{LOW_CI_BOUND}-{HIGH_CI_BOUND}_ref.png"), dpi=200, bbox_inches="tight")

        plt.figure(figsize=(6, 5))
        violin_parts = plt.violinplot([data_0_normalized_cis_subtract, data_1_normalized_cis_subtract], showmeans=True)
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
        plt.scatter(x_positions_0, data_0_normalized_cis_subtract, color='black', alpha=0.2)
        plt.scatter(x_positions_1, data_1_normalized_cis_subtract, color='black', alpha=0.2)
        # Adding mean and median text
        plt.text(1.25, mean_0_normalized_cis_subtract, f'Mean: {mean_0_normalized_cis_subtract:.2f}', ha='center', va='bottom', color='blue', fontsize=10)
        plt.text(1.25, median_0_normalized_cis_subtract, f'Median: {median_0_normalized_cis_subtract:.2f}', ha='center', va='top', color='green', fontsize=10)
        plt.text(1.75, mean_1_normalized_cis_subtract, f'Mean: {mean_1_normalized_cis_subtract:.2f}', ha='center', va='bottom', color='blue', fontsize=10)
        plt.text(1.75, median_1_normalized_cis_subtract, f'Median: {median_1_normalized_cis_subtract:.2f}', ha='center', va='top', color='green', fontsize=10)
        # Adding labels and title
        plt.xticks([1, 2], ["Unseen Data", "Seen Data"])
        plt.ylabel(f'{aggregate} Creativity Index')
        plt.title(f'Norm CI, BookMIA, {model_name}, min_ngram {min_ngram}, {doc_string}, prompt {prompt_idx}, agg {aggregate}')
        plt.ylim(top=min(CREATIVITY_CONSTANT + 0.25, max(mean_0_normalized_cis_subtract + 2*std0_normalized_cis_subtract, mean_1_normalized_cis_subtract + 2*std1_normalized_cis_subtract)), 
                bottom=max(min(mean_0_normalized_cis_subtract - 2*std0_normalized_cis_subtract, mean_1_normalized_cis_subtract - 2*std1_normalized_cis_subtract),-0.05))
        plt.grid(alpha=0.2, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"CI{LOW_CI_BOUND}-{HIGH_CI_BOUND}_norm_subtract.png"), dpi=200, bbox_inches="tight")

        plt.figure(figsize=(6, 5))
        violin_parts = plt.violinplot([data_0_normalized_cis_divide, data_1_normalized_cis_divide], showmeans=True)
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

        plt.scatter(x_positions_0, data_0_normalized_cis_divide, color='black', alpha=0.2)
        plt.scatter(x_positions_1, data_1_normalized_cis_divide, color='black', alpha=0.2)
        # Adding mean and median text
        plt.text(1.25, mean_0_normalized_cis_divide, f'Mean: {mean_0_normalized_cis_divide:.2f}', ha='center', va='bottom', color='blue', fontsize=10)
        plt.text(1.25, median_0_normalized_cis_divide, f'Median: {median_0_normalized_cis_divide:.2f}', ha='center', va='top', color='green', fontsize=10)
        plt.text(1.75, mean_1_normalized_cis_divide, f'Mean: {mean_1_normalized_cis_divide:.2f}', ha='center', va='bottom', color='blue', fontsize=10)
        plt.text(1.75, median_1_normalized_cis_divide, f'Median: {median_1_normalized_cis_divide:.2f}', ha='center', va='top', color='green', fontsize=10)
        # Adding labels and title
        plt.xticks([1, 2], ["Unseen Data", "Seen Data"])
        plt.ylabel(f'{aggregate} Creativity Index')
        plt.title(f'Norm CI, BookMIA, {model_name}, min_ngram {min_ngram}, {doc_string}, prompt {prompt_idx}, agg {aggregate}')
        # plt.ylim(top=min(CREATIVITY_CONSTANT + 0.25, max(mean_0_normalized_cis_divide + 2*std0_normalized_cis_divide, mean_1_normalized_cis_divide + 2*std1_normalized_cis_divide)), 
        #          bottom=max(min(mean_0_normalized_cis_divide - 2*std0_normalized_cis_divide, mean_1_normalized_cis_divide - 2*std1_normalized_cis_divide),-0.05))
        plt.yscale('log')
        plt.grid(alpha=0.2, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"CI{LOW_CI_BOUND}-{HIGH_CI_BOUND}_norm_divide.png"), dpi=200, bbox_inches="tight")

    ### PLOT: Plotting for Coverages
    plt.figure(figsize=(6, 5))
    violin_parts = plt.violinplot([data_0_covs, data_1_covs], showmeans=True)
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
    plt.title(f'Cov, BookMIA, {model_name}, min_ngram {min_ngram}, {doc_string}, prompt {prompt_idx}, agg {aggregate}')
    plt.ylim(top=min(1.05, max(mean_0_covs + 2*std0_covs, mean_1_covs + 2*std1_covs)), bottom=-0.05)
    plt.grid(alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"cov_orig.png"), dpi=200, bbox_inches="tight")

    if len(ref_coverage_data) == len(coverage_data):
        plt.figure(figsize=(6, 5))
        violin_parts = plt.violinplot([data_0_ref_covs, data_1_ref_covs], showmeans=True)
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
        plt.scatter(x_positions_0, data_0_ref_covs, color='black', alpha=0.2)
        plt.scatter(x_positions_1, data_1_ref_covs, color='black', alpha=0.2)
        # Adding mean and median text
        plt.text(1.25, mean_0_ref_covs, f'Mean: {mean_0_ref_covs:.2f}', ha='center', va='bottom', color='blue', fontsize=10)
        plt.text(1.25, median_0_ref_covs, f'Median: {median_0_ref_covs:.2f}', ha='center', va='top', color='green', fontsize=10)
        plt.text(1.75, mean_1_ref_covs, f'Mean: {mean_1_ref_covs:.2f}', ha='center', va='bottom', color='blue', fontsize=10)
        plt.text(1.75, median_1_ref_covs, f'Median: {median_1_ref_covs:.2f}', ha='center', va='top', color='green', fontsize=10)
        # Adding labels and title
        plt.xticks([1, 2], ["Unseen Data", "Seen Data"])
        plt.ylabel(f'{aggregate} Creativity Index')
        plt.title(f'CI, BookMIA, {model_name}, min_ngram {min_ngram}, {doc_string}, prompt {prompt_idx}, agg {aggregate}')
        plt.ylim(top=min(CREATIVITY_CONSTANT + 0.25, max(mean_0_ref_covs + 2*std0_ref_covs, mean_1_ref_covs + 2*std1_ref_covs)), 
                bottom=max(min(mean_0_ref_covs - 2*std0_ref_covs, mean_1_ref_covs - 2*std1_ref_covs),-0.05))
        plt.grid(alpha=0.2, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"cov_ref.png"), dpi=200, bbox_inches="tight")

        ### PLOT: Plotting for CIs - Reference Model
        plt.figure(figsize=(6, 5))
        violin_parts = plt.violinplot([data_0_normalized_covs_subtract, data_1_normalized_covs_subtract], showmeans=True)
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
        plt.scatter(x_positions_0, data_0_normalized_covs_subtract, color='black', alpha=0.2)
        plt.scatter(x_positions_1, data_1_normalized_covs_subtract, color='black', alpha=0.2)
        # Adding mean and median text
        plt.text(1.25, mean_0_normalized_covs_subtract, f'Mean: {mean_0_normalized_covs_subtract:.2f}', ha='center', va='bottom', color='blue', fontsize=10)
        plt.text(1.25, median_0_normalized_covs_subtract, f'Median: {median_0_normalized_covs_subtract:.2f}', ha='center', va='top', color='green', fontsize=10)
        plt.text(1.75, mean_1_normalized_covs_subtract, f'Mean: {mean_1_normalized_covs_subtract:.2f}', ha='center', va='bottom', color='blue', fontsize=10)
        plt.text(1.75, median_1_normalized_covs_subtract, f'Median: {median_1_normalized_covs_subtract:.2f}', ha='center', va='top', color='green', fontsize=10)
        # Adding labels and title
        plt.xticks([1, 2], ["Unseen Data", "Seen Data"])
        plt.ylabel(f'{aggregate} Creativity Index')
        plt.title(f'CI, BookMIA, {model_name}, min_ngram {min_ngram}, {doc_string}, prompt {prompt_idx}, agg {aggregate}')
        plt.ylim(top=min(CREATIVITY_CONSTANT + 0.25, max(mean_0_normalized_covs_subtract + 2*std0_normalized_covs_subtract, mean_1_normalized_covs_subtract + 2*std1_normalized_covs_subtract)), 
                bottom=max(min(mean_0_normalized_covs_subtract - 2*std0_normalized_covs_subtract, mean_1_normalized_covs_subtract - 2*std1_normalized_covs_subtract),-0.05))
        plt.grid(alpha=0.2, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"cov_norm_subtract.png"), dpi=200, bbox_inches="tight")

        plt.figure(figsize=(6, 5))
        violin_parts = plt.violinplot([data_0_normalized_covs_divide, data_1_normalized_covs_divide], showmeans=True)
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
        plt.scatter(x_positions_0, data_0_normalized_covs_divide, color='black', alpha=0.2)
        plt.scatter(x_positions_1, data_1_normalized_covs_divide, color='black', alpha=0.2)
        # Adding mean and median text
        plt.text(1.25, mean_0_normalized_covs_divide, f'Mean: {mean_0_normalized_covs_divide:.2f}', ha='center', va='bottom', color='blue', fontsize=10)
        plt.text(1.25, median_0_normalized_covs_divide, f'Median: {median_0_normalized_covs_divide:.2f}', ha='center', va='top', color='green', fontsize=10)
        plt.text(1.75, mean_1_normalized_covs_divide, f'Mean: {mean_1_normalized_covs_divide:.2f}', ha='center', va='bottom', color='blue', fontsize=10)
        plt.text(1.75, median_1_normalized_covs_divide, f'Median: {median_1_normalized_covs_divide:.2f}', ha='center', va='top', color='green', fontsize=10)
        # Adding labels and title
        plt.xticks([1, 2], ["Unseen Data", "Seen Data"])
        plt.ylabel(f'{aggregate} Coverage')
        plt.title(f'Cov, BookMIA, {model_name}, min_ngram {min_ngram}, {doc_string}, prompt {prompt_idx}, agg {aggregate}')
        # plt.ylim(top=min(CREATIVITY_CONSTANT + 0.25, max(mean_0_normalized_covs_divide + 2*std0_normalized_covs_divide, mean_1_normalized_covs_divide + 2*std1_normalized_covs_divide)), 
        #          bottom=max(min(mean_0_normalized_covs_divide - 2*std0_normalized_covs_divide, mean_1_normalized_covs_divide - 2*std1_normalized_covs_divide),-0.05))
        plt.yscale('symlog')
        plt.ylim(bottom=0)
        plt.grid(alpha=0.2, axis='y')
        plt.tight_layout()

        plt.savefig(os.path.join(save_folder, f"cov_norm_divide.png"), dpi=200, bbox_inches="tight") # *

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
    plt.ylim(top=min(CREATIVITY_CONSTANT + 0.25, max(mean_0_cis + 2*std0_cis, mean_1_cis + 2*std1_cis)), 
             bottom=max(min(mean_0_cis - 2*std0_cis, mean_1_cis - 2*std1_cis),-0.05))
    plt.grid(alpha=0.2, axis='y')
    plt.tight_layout()
    # plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1), title="Book IDs")
    plt.savefig(os.path.join(save_folder, f"CI{LOW_CI_BOUND}-{HIGH_CI_BOUND}_bookid.png"), dpi=200, bbox_inches="tight")

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
    plt.savefig(os.path.join(save_folder, f"cov_bookid.png"), dpi=200, bbox_inches="tight")

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
    plt.title(f'Cov ROC Curve, BookMIA, {model_name}, min_ngram {min_ngram}, {doc_string}, prompt {prompt_idx}, agg {aggregate}')
    plt.grid(alpha=0.15)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"cov_rocauc.png"), dpi=200, bbox_inches="tight")

    # ROC AUC curves for CI
    fpr, tpr, thresholds = roc_curve(gen_labels, -1 * cis)
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
    plt.title(f'CI ROC Curve, BookMIA, {model_name}, min_ngram {min_ngram}, {doc_string}, prompt {prompt_idx}, agg {aggregate}')
    plt.grid(alpha=0.15)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"CI{LOW_CI_BOUND}-{HIGH_CI_BOUND}_rocauc.png"), dpi=200, bbox_inches="tight")
    

    if len(ref_coverage_data) == len(coverage_data):
        fpr, tpr, thresholds = roc_curve(gen_labels, normalized_covs_subtract)
        roc_auc = auc(fpr, tpr)
        # Calculate accuracy for each threshold
        accuracy_scores = []
        for threshold in thresholds:
            y_pred = np.where(normalized_covs_subtract >= threshold, 1, 0)
            accuracy_scores.append(accuracy_score(gen_labels, y_pred))
        plt.figure()
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # Diagonal line for random guess
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Norm Cov ROC Curve, BookMIA, {model_name}, min_ngram {min_ngram}, {doc_string}, prompt {prompt_idx}, agg {aggregate}')
        plt.grid(alpha=0.15)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"cov_rocauc_norm_subtract.png"), dpi=200, bbox_inches="tight")

        # ROC AUC curves for CI
        fpr, tpr, thresholds = roc_curve(gen_labels, -1 * normalized_cis_subtract)
        roc_auc = auc(fpr, tpr)

        # Calculate accuracy for each threshold
        accuracy_scores = []
        for threshold in thresholds:
            y_pred = np.where(normalized_cis_subtract >= threshold, 1, 0)
            accuracy_scores.append(accuracy_score(gen_labels, y_pred))

        # Plotting the ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # Diagonal line for random guess
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Norm CI ROC Curve, BookMIA, {model_name}, min_ngram {min_ngram}, {doc_string}, prompt {prompt_idx}, agg {aggregate}')
        plt.grid(alpha=0.15)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"CI{LOW_CI_BOUND}-{HIGH_CI_BOUND}_rocauc_norm_subtract.png"), dpi=200, bbox_inches="tight")

        fpr, tpr, thresholds = roc_curve(gen_labels, normalized_covs_divide)
        roc_auc = auc(fpr, tpr)
        # Calculate accuracy for each threshold
        accuracy_scores = []
        for threshold in thresholds:
            y_pred = np.where(normalized_covs_divide >= threshold, 1, 0)
            accuracy_scores.append(accuracy_score(gen_labels, y_pred))
        plt.figure()
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # Diagonal line for random guess
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Norm Cov ROC Curve, BookMIA, {model_name}, min_ngram {min_ngram}, {doc_string}, prompt {prompt_idx}, agg {aggregate}')
        plt.grid(alpha=0.15)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"cov_rocauc_norm_divide.png"), dpi=200, bbox_inches="tight")

        # ROC AUC curves for CI
        fpr, tpr, thresholds = roc_curve(gen_labels, -1 * normalized_cis_divide)
        roc_auc = auc(fpr, tpr)

        # Calculate accuracy for each threshold
        accuracy_scores = []
        for threshold in thresholds:
            y_pred = np.where(normalized_cis_divide >= threshold, 1, 0)
            accuracy_scores.append(accuracy_score(gen_labels, y_pred))

        # Plotting the ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # Diagonal line for random guess
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Norm CI ROC Curve, BookMIA, {model_name}, min_ngram {min_ngram}, {doc_string}, prompt {prompt_idx}, agg {aggregate}')
        plt.grid(alpha=0.15)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"CI{LOW_CI_BOUND}-{HIGH_CI_BOUND}_rocauc_norm_divide.png"), dpi=200, bbox_inches="tight")

    plt.close('all')
    return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--parallel', action="store_true")
    args = parser.parse_args()

    min_ngrams=[4,5]
    prompt_idxs = list(range(10))[::-1]
    start_sents = list(range(10))
    num_sents = list(range(1, 10))
    models = ["gpt-3.5-turbo-0125", "gpt-4o-mini-2024-07-18", "gpt-4o-2024-05-13"][::-1]

    # # models = ["gpt-3.5-turbo-0125"]
    # models = ["Llama-3.1-8B-Instruct", "Llama-3.1-70B-Instruct"]
    models = ["Llama-3.1-70B-Instruct", "Llama-3.1-8B-Instruct",]
    all_docs = [False, True]
    aggregates = ["max", "mean"]
    ref_prompt_idxs = [5]
    min_tokens = [0, 10]
    temps = [1.0, 0]

    combinations = list(itertools.product(min_ngrams, prompt_idxs, start_sents, num_sents, models, all_docs, aggregates, ref_prompt_idxs, min_tokens, temps))

    if args.parallel:
        num_workers = min(cpu_count(), len(combinations))  # Limit to available CPU cores or number of tasks
        num_workers=8
        with Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(process_combination, combinations), total=len(combinations), desc="Processing in parallel", position=0))
    else:
        total = 0
        for combo in tqdm(combinations, total=len(combinations), position=0):
            func_output = process_combination(combo)
            total += 1
        print(total, len(combinations))

    """
    python3 -m tasks.bookMIA.analysis
    python3 -m tasks.bookMIA.analysis --parallel
    """