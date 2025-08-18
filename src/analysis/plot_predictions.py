from IPython import embed
from code.utils import load_json
import numpy as np 
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr

agg_strategy = "max"

file1= "/gscratch/xlab/hallisky/membership-inference/outputs/ours/bookMIA/coverages/train/gpt-4o-2024-05-13_maxTok512_minTok0_numSeq20_topP0.95_temp1.0_numSent5_startSent1_numWord-1_startWord-1_useSentF_promptIdx5_len494_2025-01-16-11:48:33_4_onedoc.jsonl"
file1_data = load_json(file1)
file1_model = file1.split("/")[-1].split("_")[0]

file2 = "/gscratch/xlab/hallisky/membership-inference/outputs/ours/bookMIA/coverages/train/gpt-4o-mini-2024-07-18_maxTok512_minTok0_numSeq20_topP0.95_temp1.0_numSent5_startSent1_numWord-1_startWord-1_useSentF_promptIdx5_len494_2025-01-13-10:28:35_4_onedoc.jsonl"
file2_data = load_json(file2)
file2_model = file2.split("/")[-1].split("_")[0]

save_path = "/gscratch/xlab/hallisky/membership-inference/code/analysis/bookMIA/plots/train" # TODO unhardcode this

def aggregate_data(file_data, aggregation = "max"):
    book_ids, snippet_ids, covs, labels, calibrated_covs = [], [], [], [], []
    for entry in file_data:
        book_id = entry['book_id']
        snippet_id = entry['snippet_id']
        label = entry['label']
        
        # Extract coverage values from all generations
        coverage_values = [gen['coverage'] for gen in entry['coverage']]
        
        if aggregation == "max":
            agg_coverage = max(coverage_values)
        elif aggregation == "mean":
            agg_coverage = np.mean(coverage_values)
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")
        
        # Create coverage-calibrated field
        if label == 1:
            calibrated_cov = agg_coverage
        elif label == 0:
            calibrated_cov = 1 - agg_coverage
        else:
            raise ValueError(f"Unexpected label value: {label}")


        book_ids.append(book_id)
        snippet_ids.append(snippet_id)
        covs.append(agg_coverage)
        labels.append(label)
        calibrated_covs.append(calibrated_cov)

    return {"book_ids": book_ids, 
            "snippet_ids": snippet_ids, 
            "coverages": np.array(covs), 
            "labels": labels, 
            "calibrated_coverages": np.array(calibrated_covs)}

all_data = {}
all_data["file1"] = aggregate_data(file1_data, aggregation=agg_strategy)
all_data["file2"] = aggregate_data(file2_data, aggregation=agg_strategy)

assert all_data["file1"]["book_ids"] == all_data["file2"]["book_ids"]
assert all_data["file1"]["snippet_ids"] == all_data["file2"]["snippet_ids"]

colors = ['black', 'red']
labels = ['Non-member', 'Member']

plt.figure()
plt.tight_layout()
for label_value in [0, 1]:
    mask = np.array(all_data["file1"]["labels"]) == label_value
    plt.scatter(
        all_data["file1"]["calibrated_coverages"][mask],
        all_data["file2"]["calibrated_coverages"][mask],
        c=colors[label_value],
        label=labels[label_value],
        alpha=0.4,
        zorder=3
        )
plt.grid(alpha=0.1, zorder=-1)
plt.xlabel(file1_model)
plt.ylabel(file2_model)

plt.legend(bbox_to_anchor=(0.5, -0.12), loc='upper center', ncol=2)
r_stat = pearsonr(all_data["file1"]["calibrated_coverages"] , all_data["file2"]["calibrated_coverages"])
plt.text(0.05, 0.95, f'R = {r_stat[0]:.2f}', transform=plt.gca().transAxes, 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
plt.savefig(os.path.join(save_path, f"calibrated_{file1_model}_{file2_model}_{agg_strategy}.png"), dpi=200, bbox_inches="tight")

# Save log version
plt.xscale('log')
plt.yscale('log')

plt.figure()
plt.tight_layout()
for label_value in [0, 1]:
    mask = np.array(all_data["file1"]["labels"]) == label_value
    plt.scatter(
        all_data["file1"]["coverages"][mask],
        all_data["file2"]["coverages"][mask],
        c=colors[label_value],
        label=labels[label_value],
        alpha=0.4,
        zorder=3
        )
plt.grid(alpha=0.1, zorder=-1)
plt.xlabel(file1_model)
plt.ylabel(file2_model)

plt.legend(bbox_to_anchor=(0.5, -0.12), loc='upper center', ncol=2)
r_stat = pearsonr(all_data["file1"]["coverages"] , all_data["file2"]["coverages"])
plt.text(0.05, 0.95, f'R = {r_stat[0]:.2f}', transform=plt.gca().transAxes, 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
plt.savefig(os.path.join(save_path, f"{file1_model}_{file2_model}_{agg_strategy}.png"), dpi=200, bbox_inches="tight")

# Save log version
plt.xscale('log')
plt.yscale('log')


# embed()

"""
python3 -m code.analysis.plot_predictions
"""