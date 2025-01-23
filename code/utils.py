import json
from tqdm import tqdm
# Load gen_path as jsonl

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in tqdm(f, "Loading jsonl")]
    return data

# Load coverage_path as a json
def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def combine_lists(list1, list2):
    output_list = []
    for l1, l2 in zip(list1, list2):
        output_list.append(l1 + l2)
    return output_list

def combine_dicts(dict1, dict2):
    for cur_key in ['logprobs', 'model']:
        if cur_key in dict1:
            dict1.pop(cur_key)
        if cur_key in dict2:
            dict2.pop(cur_key)

    # Assert that both dictionaries have the same keys
    assert dict1.keys() == dict2.keys(), f"Keys do not match: {dict1.keys()} != {dict2.keys()}"
    
    combined_dict = {}
    
    for key in dict1.keys():
        if isinstance(dict1[key], list):
            # Combine list values by concatenation
            combined_dict[key] = dict1[key] + dict2[key]
        else:
            # Ensure non-list values are equal
            assert dict1[key] == dict2[key], f"Values for '{key}' do not match: {dict1[key]} != {dict2[key]}"
            combined_dict[key] = dict1[key]
    
    return combined_dict
    
def combine_list_of_dicts(list1, list2):
    # Ensure both lists have the same length
    assert len(list1) == len(list2), f"Lists must have the same length: {len(list1)} != {len(list2)}"
    
    combined_list = []

    # Iterate through both lists simultaneously and combine each pair of dictionaries
    for dict1, dict2 in zip(list1, list2):
        combined_dict = combine_dicts(dict1, dict2)
        combined_list.append(combined_dict)
    
    return combined_list

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