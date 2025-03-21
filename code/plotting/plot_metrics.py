import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import embed

df= pd.read_csv("code/plotting/temp_data.tsv", sep='\t')

exclude_cols = ['num_sequences', 'min_ngram']
metrics = [col for col in df.columns if col not in exclude_cols]

# Columns to exclude from metrics
exclude_cols = ['num_sequences', 'min_ngram']
metrics = [col for col in df.columns if col not in exclude_cols]
num_metrics = len(metrics)

# Setup subplot grid
fig, axes = plt.subplots(1, num_metrics, figsize=(3 * num_metrics, 3), sharey=True)

# Make sure axes is iterable even with one subplot
if num_metrics == 1:
    axes = [axes]

# Plot each metric in its own subplot
for i, (ax, metric) in enumerate(zip(axes, metrics)):
    print(metric)
    for ngram in sorted(df['min_ngram'].unique()):
        df_subset = df[df['min_ngram'] == ngram]
        if "Longest" in metric:
            ax.plot(df_subset['num_sequences'], df_subset[metric], 'o--', alpha=0.7,  color="black")
        else:
            ax.plot(df_subset['num_sequences'], df_subset[metric], 'o--', label=f'min_ngram={ngram}', alpha=0.7)

    ax.set_xlabel('Number of Sequences')
    # ax.set_ylabel(metric)
    ax.set_title(f'{metric.split("_")[1]}')
    ax.grid(alpha=0.3)
    if i == num_metrics-3:
        ax.legend(ncols=num_metrics+1, loc='upper center', bbox_to_anchor=(0.95, -0.18))
    
plt.tight_layout()
plt.savefig("code/plotting/plots/metrics_subplot.pdf", bbox_inches="tight")
plt.show()

# embed()

# # Make subplots

# # OLD CODE
# # Plot (for 1)
# plt.figure(figsize=(4,4))

# for metric, scores in performance.items():
#     plt.plot(generation_lengths, scores, 'o--', label=metric)

# plt.xlabel('Generation Length')
# plt.ylabel('Performance')
# plt.yticks(np.arange(0.5, 1.0, 0.1))
# plt.title('Performance vs. Generation Length')
# plt.grid(alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.savefig("code/plotting/plots/metrics.png", dpi=200, bbox_inches="tight")

"""
python3 -m code.plotting.plot_metrics
"""