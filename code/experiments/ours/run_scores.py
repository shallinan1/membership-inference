
from code.user_secrets import CACHE_PATH
import os
# Set up environment variables
os.environ["HF_HOME"] = CACHE_PATH
os.environ["HF_DATASETS_PATH"] = CACHE_PATH
import argparse
from sklearn.metrics import roc_curve, auc, accuracy_score
from code.utils import load_json

import json
from code.experiments.utils import plot_roc_curve
from IPython import embed
import numpy as np

strategies = {
    "Min_Coverage_Gen_Length": {"func": lambda x: np.min(x["coverages_gen_length"])},
    "Max_Coverage_Gen_Length": {"func": lambda x: np.max(x["coverages_gen_length"])},
    "Median_Coverage_Gen_Length": {"func": lambda x: np.median(x["coverages_gen_length"])},
    "Mean_Coverage_Gen_Length": {"func": lambda x: np.mean(x["coverages_gen_length"])},
    
    "Min_Coverage_Ref_Length": {"func": lambda x: np.min(x["coverages_ref_length"])},
    "Max_Coverage_Ref_Length": {"func": lambda x: np.max(x["coverages_ref_length"])},
    "Median_Coverage_Ref_Length": {"func": lambda x: np.median(x["coverages_ref_length"])},
    "Mean_Coverage_Ref_Length": {"func": lambda x: np.mean(x["coverages_ref_length"])},
    
    "Min_Coverage_Total_Length": {"func": lambda x: np.min(x["coverages_total_length"])},
    "Max_Coverage_Total_Length": {"func": lambda x: np.max(x["coverages_total_length"])},
    "Median_Coverage_Total_Length": {"func": lambda x: np.median(x["coverages_total_length"])},
    "Mean_Coverage_Total_Length": {"func": lambda x: np.mean(x["coverages_total_length"])},

    "Min_Coverage_Gen_Length_Unique": {"func": lambda x: np.min(x["coverages_gen_length_unique"])},
    "Max_Coverage_Gen_Length_Unique": {"func": lambda x: np.max(x["coverages_gen_length_unique"])},
    "Median_Coverage_Gen_Length_Unique": {"func": lambda x: np.median(x["coverages_gen_length_unique"])},
    "Mean_Coverage_Gen_Length_Unique": {"func": lambda x: np.mean(x["coverages_gen_length_unique"])},
    
    "Min_Coverage_Ref_Length_Unique": {"func": lambda x: np.min(x["coverages_ref_length_unique"])},
    "Max_Coverage_Ref_Length_Unique": {"func": lambda x: np.max(x["coverages_ref_length_unique"])},
    "Median_Coverage_Ref_Length_Unique": {"func": lambda x: np.median(x["coverages_ref_length_unique"])},
    "Mean_Coverage_Ref_Length_Unique": {"func": lambda x: np.mean(x["coverages_ref_length_unique"])},
    
    "Min_Coverage_Total_Length_Unique": {"func": lambda x: np.min(x["coverages_total_length_unique"])},
    "Max_Coverage_Total_Length_Unique": {"func": lambda x: np.max(x["coverages_total_length_unique"])},
    "Median_Coverage_Total_Length_Unique": {"func": lambda x: np.median(x["coverages_total_length_unique"])},
    "Mean_Coverage_Total_Length_Unique": {"func": lambda x: np.mean(x["coverages_total_length_unique"])},

    "Min_Creativity_Gen_Length": {"func": lambda x: np.min(x["creativities_gen_length"])},
    "Max_Creativity_Gen_Length": {"func": lambda x: np.max(x["creativities_gen_length"])},
    "Median_Creativity_Gen_Length": {"func": lambda x: np.median(x["creativities_gen_length"])},
    "Mean_Creativity_Gen_Length": {"func": lambda x: np.mean(x["creativities_gen_length"])},

    "Min_Creativity_Ref_Length": {"func": lambda x: np.min(x["creativities_ref_length"])},
    "Max_Creativity_Ref_Length": {"func": lambda x: np.max(x["creativities_ref_length"])},
    "Median_Creativity_Ref_Length": {"func": lambda x: np.median(x["creativities_ref_length"])},
    "Mean_Creativity_Ref_Length": {"func": lambda x: np.mean(x["creativities_ref_length"])},
    
    "Min_Creativity_Total_Length": {"func": lambda x: np.min(x["creativities_total_length"])},
    "Max_Creativity_Total_Length": {"func": lambda x: np.max(x["creativities_total_length"])},
    "Median_Creativity_Total_Length": {"func": lambda x: np.median(x["creativities_total_length"])},
    "Mean_Creativity_Total_Length": {"func": lambda x: np.mean(x["creativities_total_length"])},

    "Min_Creativity_Gen_Length_Unique": {"func": lambda x: np.min(x["creativities_gen_length_unique"])},
    "Max_Creativity_Gen_Length_Unique": {"func": lambda x: np.max(x["creativities_gen_length_unique"])},
    "Median_Creativity_Gen_Length_Unique": {"func": lambda x: np.median(x["creativities_gen_length_unique"])},
    "Mean_Creativity_Gen_Length_Unique": {"func": lambda x: np.mean(x["creativities_gen_length_unique"])},

    "Min_Creativity_Ref_Length_Unique": {"func": lambda x: np.min(x["creativities_ref_length_unique"])},
    "Max_Creativity_Ref_Length_Unique": {"func": lambda x: np.max(x["creativities_ref_length_unique"])},
    "Median_Creativity_Ref_Length_Unique": {"func": lambda x: np.median(x["creativities_ref_length_unique"])},
    "Mean_Creativity_Ref_Length_Unique": {"func": lambda x: np.mean(x["creativities_ref_length_unique"])},
    
    "Min_Creativity_Total_Length_Unique": {"func": lambda x: np.min(x["creativities_total_length_unique"])},
    "Max_Creativity_Total_Length_Unique": {"func": lambda x: np.max(x["creativities_total_length_unique"])},
    "Median_Creativity_Total_Length_Unique": {"func": lambda x: np.median(x["creativities_total_length_unique"])},
    "Mean_Creativity_Total_Length_Unique": {"func": lambda x: np.mean(x["creativities_total_length_unique"])},

    "Min_GenTextLengthChar": {"func": lambda x: np.min(x["gen_text_length_char"])},
    "Max_GenTextLengthChar": {"func": lambda x: np.max(x["gen_text_length_char"])},
    "Median_GenTextLengthChar": {"func": lambda x: np.median(x["gen_text_length_char"])},
    "Mean_GenTextLengthChar": {"func": lambda x: np.mean(x["gen_text_length_char"])},
    
    "Min_GenTextLengthWord": {"func": lambda x: np.min(x["gen_text_length_word"])},
    "Max_GenTextLengthWord": {"func": lambda x: np.max(x["gen_text_length_word"])},
    "Median_GenTextLengthWord": {"func": lambda x: np.median(x["gen_text_length_word"])},
    "Mean_GenTextLengthWord": {"func": lambda x: np.mean(x["gen_text_length_word"])},
    
    "Min_LongestSubstringChar": {"func": lambda x: np.min(x["longest_substring_char"])},
    "Max_LongestSubstringChar": {"func": lambda x: np.max(x["longest_substring_char"])},
    "Median_LongestSubstringChar": {"func": lambda x: np.median(x["longest_substring_char"])},
    "Mean_LongestSubstringChar": {"func": lambda x: np.mean(x["longest_substring_char"])},
    
    "Min_LongestSublistWord": {"func": lambda x: np.min(x["longest_sublist_word"])},
    "Max_LongestSublistWord": {"func": lambda x: np.max(x["longest_sublist_word"])},
    "Median_LongestSublistWord": {"func": lambda x: np.median(x["longest_sublist_word"])},
    "Mean_LongestSublistWord": {"func": lambda x: np.mean(x["longest_sublist_word"])},
}

def main(args):
    target_model_name = args.outputs_file.split(os.sep)[-1][:-6]

    base_dir = os.path.dirname(args.outputs_file).replace("creativities", "scores")  # Up one level from 'probs'
    output_dir = os.path.join(base_dir, target_model_name)
    plot_dir = os.path.join(output_dir, 'plots')
    print(f"Saving to {output_dir}")

    results = load_json(args.outputs_file) # Load in the probs from file
    gen_labels = [g["label"] for g in results]

    all_scores = {}
    for strategy in strategies:
        strategy_values = strategies[strategy]

        if strategy == "ReferenceLoss":
            if args.ref_model_probs is not None:
                # TODO implement this
                pass
        else:
            scores = [strategy_values["func"](r) for r in results]

            fpr, tpr, thresholds = roc_curve(gen_labels, scores)
            roc_auc = auc(fpr, tpr)
            all_scores[strategy] = {}
            all_scores[strategy]["roc_auc"] = roc_auc

            # plot_title=f"{dataset} ({split}): {strategy}, {target_model_name}"
            # plot_roc_curve(fpr, tpr, roc_auc, plot_title, os.path.join(plot_dir, f"{strategy}.png"))

    output_file_path = os.path.join(output_dir, f"scores.json")
    os.makedirs(plot_dir, exist_ok=True)
    with open(output_file_path, 'w') as f:
        json.dump(all_scores, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--outputs_file', type=str, default=None)
    main(parser.parse_args())

    """
    python3 -m code.experiments.ours.run_scores \
        --outputs_file /gscratch/xlab/hallisky/membership-inference/outputs/ours/pile_external/coverages/val/pythia-1.4b_maxTok512_minTok0_numSeq20_topP0.95_temp1.0_numSent5_startSent0_numWord-1_startWord-1_useSentF_promptIdx0_len100_2025-01-22-23:15:24_3_onedoc.jsonl \
    """

