import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

font_path = "../fonts/TeX-Gyre-Pagella/texgyrepagella-regular.otf"
fm.fontManager.addfont(font_path)
custom_font = fm.FontProperties(fname=font_path).get_name()

bold_path = "../fonts/TeX-Gyre-Pagella/texgyrepagella-bold.otf"
fm.fontManager.addfont(bold_path)
bold_font = fm.FontProperties(fname=bold_path)

plt.rcParams['font.family'] = custom_font

import numpy as np
import pandas as pd
from IPython import embed

name_map = {
    "LongestSubstringChar": 'LCS (character)',
    "LongestSublistWord": "LCS (word)"
}

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
            ax.plot(df_subset['num_sequences'], df_subset[metric], 'o--', label=f'Min N-Gram={ngram}', alpha=0.7)

    # ax.set_xlabel('Number of Sequences')
    # ax.set_ylabel(metric)
    cur_title = f'{metric.split("_")[1]}'
    ax.set_title(name_map.get(cur_title, cur_title))
    ax.set_xticks([10, 20, 50, 100])
    ax.grid(alpha=0.3)
    if i == num_metrics-3:
        ax.legend(ncols=num_metrics+1, loc='upper center', bbox_to_anchor=(0.95, -0.19))

plt.tight_layout()
fig.supxlabel('Number of Sequences', fontsize=14, font=bold_font, y=-0.04)
plt.savefig("code/plotting/plots/metrics_subplot.pdf", bbox_inches="tight")
plt.show()

# embed()
"""
python3 -m code.plotting.plot_metrics
"""