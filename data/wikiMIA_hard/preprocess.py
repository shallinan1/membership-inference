"""
Preprocesses the WikiMIA-2024 Hard dataset.
Downloads the pre-split dataset from HuggingFace and saves train/val/test splits locally.
This dataset contains temporal Wikipedia articles with different versions based on date cutoffs.
"""
from dotenv import load_dotenv
load_dotenv()
import os
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

from datasets import load_dataset
from code.utils import save_to_jsonl

def main():
    # Load the dataset from HuggingFace
    dataset = load_dataset("hallisky/wikiMIA-2024-hard")
    
    # Create output directory
    save_folder = os.path.join("data", "wikiMIA_hard", "split-random-overall")
    os.makedirs(save_folder, exist_ok=True)
    
    # Convert and save each split
    train_data = [dict(item) for item in dataset["train"]]
    val_data = [dict(item) for item in dataset["validation"]]
    test_data = [dict(item) for item in dataset["test"]]
    
    # Save train, val, and test splits
    save_to_jsonl(train_data, os.path.join(save_folder, "train.jsonl"))
    save_to_jsonl(test_data, os.path.join(save_folder, "test.jsonl"))
    save_to_jsonl(val_data, os.path.join(save_folder, "val.jsonl"))
    
    print("Data splits saved in folder:", save_folder)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}, Val: {len(val_data)}")

if __name__ == "__main__":
    main()

    """
    # To run this file, use:
    python3 -m data.wikiMIA_hard.preprocess
    """