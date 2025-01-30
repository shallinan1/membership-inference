from code.user_secrets import CACHE_PATH
import os
# Set up environment variables
os.environ["HF_HOME"] = CACHE_PATH
os.environ["HF_DATASETS_PATH"] = CACHE_PATH

import random
import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split
from code.utils import save_to_jsonl, load_jsonl
from IPython import embed
from datasets import load_dataset

# This code is to re-use the Pile data from the "Do MIA work" paper
def main(args):   
    random.seed(args.seed)
    np.random.seed(args.seed)

    data_member = load_jsonl("data/tulu_v1/processed/tulu_v1/tulu_v1_data.jsonl")
    data_nonmember = load_jsonl("data/tulu_v1/processed/tulu_v1/inverse_tulu_v1_data.jsonl")
    
    # Sample data
    data_member = random.sample(data_member, 1000)
    data_nonmember = random.sample(data_nonmember, 1000)
    for d in data_member:
        d["label"] = 1
    for d in data_nonmember:
        d["label"] = 0
        
    member_train, member_temp = train_test_split(data_member, test_size=args.val_split + args.test_split, random_state=args.seed)
    val_split_adjusted = args.val_split / (args.val_split + args.test_split)
    member_val, member_test = train_test_split(member_temp, test_size=1 - val_split_adjusted, random_state=args.seed)

    # Split nonmember and member data
    nonmember_train, nonmember_temp = train_test_split(data_nonmember, test_size=args.val_split + args.test_split, random_state=args.seed)
    val_split_adjusted = args.val_split / (args.val_split + args.test_split)
    nonmember_val, nonmember_test = train_test_split(nonmember_temp, test_size=1 - val_split_adjusted, random_state=args.seed)

    # Recombine and shuffle
    train_data = nonmember_train + member_train
    val_data = nonmember_val + member_val
    test_data = nonmember_test + member_test

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    save_folder = os.path.join("data", "tulu_v1", "split-random-overall")
    os.makedirs(save_folder, exist_ok=True)

    # Save train, val, and test splits
    save_to_jsonl(train_data, os.path.join(save_folder, "test.jsonl"))
    save_to_jsonl(test_data, os.path.join(save_folder, "train.jsonl")) # Make the "train set" the test set, since we want it to be the majority
    save_to_jsonl(val_data, os.path.join(save_folder, "val.jsonl"))

    print("Data splits saved in folder:", save_folder)        
    print(train_data[0])
    print(len(train_data), len(test_data), len(val_data))
    embed()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make data splits")
    parser.add_argument("--data_points", type=int, default=1000) # Others: wikiMIA, Pile?
    parser.add_argument("--val_split", type=float, default=0.05) # Others: wikiMIA, Pile?
    parser.add_argument("--test_split", type=float, default=0.05) # Others: wikiMIA, Pile?
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args()

    main(args)

    """
    # To run this file, use:
    python3 -m data.tulu_v1.preprocess
    """

