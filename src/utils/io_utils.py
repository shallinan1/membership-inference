import json
from tqdm import tqdm
# Load gen_path as jsonl
import re

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in tqdm(f, "Loading jsonl")]
    return data

# Load coverage_path as a json
def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_to_jsonl(data, file_path):
    """
    Saves a list of dictionaries to a JSONL file.

    Parameters:
    - data (list): A list of dictionaries to save.
    - file_path (str): Path to the JSONL file.
    """
    with open(file_path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    print(f"Data saved to {file_path} in JSONL format.")