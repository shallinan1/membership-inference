# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
import os
from tasks.bookMIA.analysis import get_ngram_coverage, compute_ci_statistic
from IPython import embed
from utils import load_jsonl, load_json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score


LOW_CI_BOUND=3
HIGH_CI_BOUND=12
CREATIVITY_CONSTANT = HIGH_CI_BOUND - LOW_CI_BOUND + 1
NORM_FACTOR = 1e-2

save_folder = "tasks/logistic/" 
gen_path = "tasks/bookMIA/generations/train/Llama-2-7b-hf_maxTok512_minTok0_numSeq20_topP0.95_temp1.0_numSent10_startSent1_numWord-1_startWord-1_useSentF_promptIdx1_len494_2024-10-30-23:41:07.jsonl"
gen_path = "tasks/bookMIA/generations/val/Llama-2-70b-hf_maxTok512_minTok0_numSeq20_topP0.95_temp1.0_numSent10_startSent1_numWord-1_startWord-1_useSentF_promptIdx1_len493_2024-10-31-06:03:33.jsonl"
min_ngram = 5
doc_string = "onedoc"
coverage_path = gen_path.replace(".jsonl", f"_{min_ngram}_{doc_string}.jsonl").replace("generations", "coverages")
save_folder = os.path.join(save_folder, coverage_path.split(os.sep)[-1][:-6])
os.makedirs(save_folder,exist_ok=True)

gen_data = load_jsonl(gen_path)
gen_labels = [g["label"] for g in gen_data]

coverage_data = load_json(coverage_path)
cis = [compute_ci_statistic(cur_data, LOW_CI_BOUND, HIGH_CI_BOUND) for cur_data in tqdm(coverage_data, leave=False, desc = "Iterating through original data", position=1)]
#cis = np.array([[[CREATIVITY_CONSTANT-c] for c in d] for d in cis]) # Convert  # TODO fix this

covs = [[c["coverage"] for c in cur_data] for cur_data in coverage_data]

# Load a dataset (using the iris dataset as an example)
data = load_iris()
X = data.data  # Features

for X_train, name in [(cis, "CI"), (covs, "Cov")]:
    if name == "CI":
        MIN = 1
        num_bins = 15
        # Define the bin edges
        bins = (CREATIVITY_CONSTANT + MIN) - np.geomspace(MIN, CREATIVITY_CONSTANT, num_bins)
        # bins = np.append(bins, [0])
        bins = bins[::-1]
        bins = np.linspace(0, CREATIVITY_CONSTANT, num_bins)

        # Count the occurrences in each bin
        X_train = np.array([np.histogram(row, bins=bins)[0] for row in X_train])

    embed()
    log_reg = LogisticRegression(max_iter=1000)  # Initialize the Logistic Regression model

    # Train the model
    log_reg.fit(X_train, gen_labels)

    # Make predictions on the test set
    y_pred = log_reg.predict(X_train)

    # Evaluate the model
    accuracy = accuracy_score(gen_labels, y_pred)
    report = classification_report(gen_labels, y_pred)

    print("Accuracy:", accuracy)
    print("\nClassification Report:\n", report)
embed()
"""
python3 -m tasks.train_logistic_regression.py
"""

fpr, tpr, thresholds = roc_curve(gen_labels, covs)
roc_auc = auc(fpr, tpr)