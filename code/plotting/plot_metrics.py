import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import argparse
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
def main(args):
    df= pd.read_csv(args.data_path, sep='\t')

    # Columns to exclude from metrics
    exclude_cols = [args.save_name, 'min_ngram']
    metrics = [col for col in df.columns if col not in exclude_cols]
    num_metrics = len(metrics)

    # Setup subplot grid
    fig, axes = plt.subplots(1, num_metrics, figsize=(3 * num_metrics, 2), sharey=True)

    # Make sure axes is iterable even with one subplot
    if num_metrics == 1:
        axes = [axes]

    # Plot each metric in its own subplot
    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        print(metric)
        for ngram in sorted(df['min_ngram'].unique()):
            df_subset = df[df['min_ngram'] == ngram]
            if "Longest" in metric:
                ax.plot(df_subset[args.save_name], df_subset[metric], 'o--', alpha=0.7,  color="black")
            else:
                ax.plot(df_subset[args.save_name], df_subset[metric], 'o--', label=f'Min N-Gram={ngram}', alpha=0.7)

            if "Longest" in metric:
                break

        # ax.set_xlabel('Number of Sequences')
        # ax.set_ylabel(metric)
        cur_title = f'{metric.split("_")[1]}'
        ax.set_title(name_map.get(cur_title, cur_title))
        if "prop" in args.save_name:
            ax.set_xticks(np.array([10, 25, 50, 75, 90])/100)
        elif "temp" in args.save_name:
            ax.set_xticks(np.array([20, 50, 80, 100, 120, 150, 180])/100)
        else:
            ax.set_xticks([10, 20, 50, 100])
        ax.grid(alpha=0.3)
        if i == num_metrics-3:
            ax.legend(ncols=num_metrics+1, loc='upper center', bbox_to_anchor=(0.95, 1.5))

    fig.subplots_adjust(wspace=0.1) 
    fig.supxlabel(args.x_label, fontsize=16, font=bold_font, y=-0.18)
    plt.savefig(f"code/plotting/plots/{args.save_name}.pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple script to parse data path.")
    parser.add_argument('--data_path', type=str, default="code/plotting/temp_data.tsv", help="Path to the data directory.")
    parser.add_argument('--x_label', type=str, help="Path to the data directory.")

    parser.add_argument("--save_name")
    main(parser.parse_args())

# embed()
"""
python3 -m code.plotting.plot_metrics \
    --data_path code/plotting/temp_data_min_num_sequences.tsv \
    --save_name num_sequences \
    --x_label "Number of Sequences Generated"
    
python3 -m code.plotting.plot_metrics \
    --data_path code/plotting/temp_data_min_proportion.tsv \
    --save_name num_proportion_from_end \
    --x_label "Proportion of Text as Prompt"

python3 -m code.plotting.plot_metrics \
    --data_path code/plotting/temp_data_temp.tsv \
    --save_name temperature  \
    --x_label "Temperature"
"""