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
            tpr_at_fpr[f"TPR@{target_fpr*100:.1f}%"] = tpr[idx]
        else:
            tpr_at_fpr[f"TPR@{target_fpr*100:.1f}%"] = 0.0
    
    return {"AUROC": auroc, **tpr_at_fpr}

def process_overlaps_file(file_path):
    """Process a single overlaps file and return metrics."""
    data = load_jsonl(file_path)
    y_true = [obj['label'] for obj in data]
    y_pred = [obj['final_score'] for obj in data]
    
    return calculate_metrics(np.array(y_true), np.array(y_pred))

def process_scores_file(file_path):
    """Process a scores.json file and return metrics."""
    with open(file_path, 'r') as f:
        scores = json.load(f)
    
    # Extract metrics from scores.json
    metrics = {}
    for method, method_scores in scores.items():
        if isinstance(method_scores, dict) and 'roc_auc' in method_scores:
            metrics[method] = {
                "AUROC": method_scores['roc_auc'],
                "TPR@0.1%": method_scores.get('tpr_at_0.1_fpr', None),
                "TPR@0.5%": method_scores.get('tpr_at_0.5_fpr', None),
                "TPR@1.0%": method_scores.get('tpr_at_1.0_fpr', None),
                "TPR@5.0%": method_scores.get('tpr_at_5.0_fpr', None)
            }
    
    return metrics

def main():
    base_dir = Path("outputs/baselines")
    results = {}  # Will be organized by dataset
    
    # Iterate through all dataset folders
    for dataset_dir in base_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
            
        # Initialize dataset results
        dataset_name = dataset_dir.name
        results[dataset_name] = {
            "AUROC": {},
            "TPR@0.1%": {},
            "TPR@0.5%": {},
            "TPR@1.0%": {},
            "TPR@5.0%": {}
        }
            
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
                    
                model_name = model_dir.name
                scores_file = model_dir / "scores.json"
                if scores_file.exists():
                    metrics = process_scores_file(scores_file)
                    for method, method_metrics in metrics.items():
                        for metric_name, value in method_metrics.items():
                            if value is not None:
                                if model_name not in results[dataset_name][metric_name]:
                                    results[dataset_name][metric_name][model_name] = {}
                                results[dataset_name][metric_name][model_name][method] = value
        
        # Process decop_probs folder if it exists
        decop_dir = eval_dir / "decop_probs"
        if decop_dir.exists():
            for model_file in decop_dir.glob("*.jsonl"):
                model_name = model_file.stem
                # TODO: Process decop_probs files
                pass
        
        # Process overlaps folders
        for overlap_type in ["PIP_overlaps", "SPL_overlaps", "VMA_overlaps"]:
            overlaps_dir = eval_dir / overlap_type
            if overlaps_dir.exists():
                for model_file in overlaps_dir.glob("*.jsonl"):
                    # Remove _zlib from model name if present
                    model_name = model_file.stem.replace("_zlib", "")
                    metrics = process_overlaps_file(model_file)
                    # Determine method name based on file name
                    if overlap_type == "SPL_overlaps":
                        if "_zlib" in model_file.stem:
                            method = "SPL_zlib"
                        else:
                            method = "SPL"
                    else:
                        method = overlap_type.replace("_overlaps", "")
                    for metric_name, value in metrics.items():
                        if model_name not in results[dataset_name][metric_name]:
                            results[dataset_name][metric_name][model_name] = {}
                        results[dataset_name][metric_name][model_name][method] = value
    
    from IPython import embed; embed()
    # Create and save tables
    for dataset_name, dataset_results in results.items():
        for metric_name, model_results in dataset_results.items():
            if model_results:  # Only create table if we have data
                # Convert to DataFrame
                df = pd.DataFrame.from_dict(model_results, orient='index').round(4)
                
                # Save to CSV
                output_file = f"tables/{dataset_name}_{metric_name}_table.csv"
                os.makedirs("tables", exist_ok=True)
                df.to_csv(output_file)
                print(f"Saved {dataset_name} {metric_name} table to {output_file}")

if __name__ == "__main__":
    main() 

    """
    python3 -m code.experiments.baselines.process_results
    """