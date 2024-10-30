
from user_secrets import CACHE_PATH
import os
# Set up environment variables
os.environ["HF_HOME"] = CACHE_PATH
os.environ["HF_DATASETS_PATH"] = CACHE_PATH
import argparse
from sklearn.metrics import roc_curve, auc, accuracy_score
import zlib
from tqdm import tqdm
from utils import load_jsonl
import numpy as np
from IPython import embed


# def inference(model1, tokenizer1,text, ex, modelname1):

#     perplexity, probabilities, loss = getPerplexityProbLoss(text, model1, tokenizer1, gpu=model1.device)
#     p_lower, _, p_lower_likelihood = getPerplexityProbLoss(text.lower(), model1, tokenizer1, gpu=model1.device)

#     # ppl
#     pred["ppl"] = p1
#     # Ratio of log ppl of large and small models
#     pred["ppl/Ref_ppl (calibrate PPL to the reference model)"] = p1_likelihood-p_ref_likelihood
 
#     # Ratio of log ppl of lower-case and normal-case
#     pred["ppl/lowercase_ppl"] = -(np.log(p_lower) / np.log(p1)).item()

#     # min-k prob
#     for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
#         k_length = int(len(all_prob)*ratio)
#         topk_prob = np.sort(all_prob)[:k_length]
#         pred[f"Min_{ratio*100}% Prob"] = -np.mean(topk_prob).item()

#     return ex



def zlib_attack(loss, text):
    return loss/len(zlib.compress(bytes(text, 'utf-8')))

strategies = {"Perplexity": { "func": lambda x: -x["loss"]},
              "Loss": {"func": lambda x: -x["loss"]},
              "Zlib": {"func": lambda x: zlib_attack(x["loss"], x["snippet"])},
              "Reference Loss": {"func": lambda x, y: y - x}}

def main(args):
    # Load in the probs from file
    results = load_jsonl(args.target_model_probs)
    gen_labels = [g["label"] for g in results]

    for strategy in strategies:
        if strategy == "Reference Loss":
            if args.ref_model_probs is not None:
                for ref_model_path in args.ref_model_probs:
                    results_ref = load_jsonl(args.target_model_probs)
                    assert len(results_ref) == len(results)
                    assert [g["label"] for g in results] == gen_labels

                    strategy_values = strategies["Reference Loss"]
                    scores = [strategy_values["func"](orig["loss"], ref["loss"]) for orig, ref in zip(results, results_ref)]

                    fpr, tpr, thresholds = roc_curve(gen_labels, scores)
                    roc_auc = auc(fpr, tpr)
                    print(strategy, roc_auc)
        else:
            strategy_values = strategies[strategy]
            scores = [strategy_values["func"](r) for r in results]

            fpr, tpr, thresholds = roc_curve(gen_labels, scores)
            roc_auc = auc(fpr, tpr)
            print(strategy, roc_auc)

        # # Calculate accuracy for each threshold
        # accuracy_scores = []
        # for threshold in thresholds:
        #     y_pred = np.where(covs >= threshold, 1, 0)
        #     accuracy_scores.append(accuracy_score(gen_labels, y_pred))

    


    # all_output = evaluate_data(final_subset, model1, model2, tokenizer1, tokenizer2, args.key_name, args.target_model, args.ref_model)
    
    # fig_fpr_tpr(all_output, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--target_model_probs', type=str, default=None)
    parser.add_argument('--ref_model_probs', type=str, nargs="+", default=None)
    main(parser.parse_args())


    """
    python3 -m baselines.run_baselines \
        --target_model_probs baselines/bookMIA/val/probs/Llama-2-7b-hf.jsonl \

    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m baselines.run_baselines \
        --target_model meta-llama/Llama-2-70b-hf \
        --key_name snippet;
    """

