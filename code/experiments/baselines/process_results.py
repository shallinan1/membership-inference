import os
import json
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
from pathlib import Path
from code.utils import load_jsonl

def calculate_metrics(y_true, y_pred):
    """Calculate AUROC and TPR at different FPR thresholds."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_pred)
    
    # Calculate TPR at different FPR thresholds
    tpr_at_fpr = {}
    for target_fpr in [0.001, 0.005, 0.01, 0.05]:  # 0.1%, 0.5%, 1%, 5%
        valid_indices = np.where(fpr <= target_fpr)[0]
        if len(valid_indices) > 0:
            idx = valid_indices[-1]  # Get the last (highest) FPR that's <= target
            tpr_at_fpr[f"TPR@{target_fpr*100}%"] = tpr[idx]
        else:
            tpr_at_fpr[f"TPR@{target_fpr*100}%"] = 0.0  # If no FPR <= target, set TPR to 0
    
    return {"AUROC": auroc, **tpr_at_fpr}

def process_overlaps_file(file_path):
    """Process a single overlaps file and return metrics."""
    data = load_jsonl(file_path)
    y_true = [obj['label'] for obj in data]
    y_pred = [obj['score'] for obj in data]
    
    return calculate_metrics(np.array(y_true), np.array(y_pred))

def process_scores_file(file_path):
    """Process a scores.json file and return metrics."""
    with open(file_path, 'r') as f:
        scores = json.load(f)
    
    # Extract metrics from scores.json
    metrics = {}
    for method, method_scores in scores.items():
        if isinstance(method_scores, dict) and 'auroc' in method_scores:
            metrics[method] = {
                "AUROC": method_scores['auroc'],
                "TPR@0.1%": method_scores.get('tpr_at_0.1_fpr', None),
                "TPR@0.5%": method_scores.get('tpr_at_0.5_fpr', None),
                "TPR@1%": method_scores.get('tpr_at_1_fpr', None),
                "TPR@5%": method_scores.get('tpr_at_5_fpr', None)
            }
    
    return metrics

def main():
    base_dir = Path("outputs/baselines")
    results = {
        "AUROC": [],
        "TPR@0.1%": [],
        "TPR@0.5%": [],
        "TPR@1%": [],
        "TPR@5%": []
    }
    
    # Iterate through all dataset folders
    for dataset_dir in base_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
            
        # Determine whether to use train or test folder
        if dataset_dir.name == "bookMIA":
            eval_dir = dataset_dir / "train"
        else:
            eval_dir = dataset_dir / "test"
            
        if not eval_dir.exists():
            continue
            
        # Check for required folders
        if not (eval_dir / "results").exists() and not (eval_dir / "decop_probs").exists():
            continue
            
        # Process results folder if it exists
        results_dir = eval_dir / "results"
        if results_dir.exists():
            for model_dir in results_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                    
                scores_file = model_dir / "scores.json"
                if scores_file.exists():
                    metrics = process_scores_file(scores_file)
                    for method, method_metrics in metrics.items():
                        for metric_name, value in method_metrics.items():
                            if value is not None:
                                results[metric_name].append({
                                    "Dataset": dataset_dir.name,
                                    "Model": model_dir.name,
                                    "Method": method,
                                    "Value": value
                                })
        
        # Process overlaps folders
        for overlap_type in ["PIP_overlaps", "SPL_overlaps", "VMA_overlaps"]:
            overlaps_dir = eval_dir / overlap_type
            if overlaps_dir.exists():
                for model_file in overlaps_dir.glob("*.jsonl"):
                    metrics = process_overlaps_file(model_file)
                    for metric_name, value in metrics.items():
                        results[metric_name].append({
                            "Dataset": dataset_dir.name,
                            "Model": model_file.stem,
                            "Method": overlap_type.replace("_overlaps", ""),
                            "Value": value
                        })
    
    from IPython import embed; embed()
    # Create and save tables
    for metric_name, data in results.items():
        if data:  # Only create table if we have data
            df = pd.DataFrame(data)
            # Pivot the table to have datasets as rows and models as columns
            pivot_df = df.pivot_table(
                index=["Dataset", "Method"],
                columns="Model",
                values="Value"
            ).round(4)
            
            # Save to CSV
            output_file = f"tables/{metric_name}_table.csv"
            os.makedirs("tables", exist_ok=True)
            pivot_df.to_csv(output_file)
            print(f"Saved {metric_name} table to {output_file}")

if __name__ == "__main__":
    main() 

    """
    python3 -m code.experiments.baselines.process_results
    """