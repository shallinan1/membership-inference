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
from code.utils import load_jsonl, save_to_jsonl
from IPython import embed
from datasets import load_dataset

# This code is to re-use the Pile data from the "Do MIA work" paper
def main(args):   
    dataset = load_dataset("iamgroot42/mimir", "pile_cc", split="ngram_7_0.2", cache_dir=CACHE_PATH, trust_remote_code=True)
    
    data_member = [{"snippet": d, "label": 1} for d in dataset["member"]]
    data_nonmember = [{"snippet": d, "label": 0} for d in dataset["nonmember"]]

    # Split nonmember and member data
    nonmember_train, nonmember_temp = train_test_split(data_nonmember, test_size=args.val_split + args.test_split, random_state=args.seed)
    val_split_adjusted = args.val_split / (args.val_split + args.test_split)
    nonmember_val, nonmember_test = train_test_split(nonmember_temp, test_size=1 - val_split_adjusted, random_state=args.seed)

    member_train, member_temp = train_test_split(data_member, test_size=args.val_split + args.test_split, random_state=args.seed)
    val_split_adjusted = args.val_split / (args.val_split + args.test_split)
    member_val, member_test = train_test_split(member_temp, test_size=1 - val_split_adjusted, random_state=args.seed)

    # Recombine and shuffle
    train_data = nonmember_train + member_train
    val_data = nonmember_val + member_val
    test_data = nonmember_test + member_test

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    save_folder = os.path.join("data", "pile_external", "split-random-overall")
    os.makedirs(save_folder, exist_ok=True)

    # Save train, val, and test splits
    save_to_jsonl(train_data, os.path.join(save_folder, "test.jsonl"))
    save_to_jsonl(test_data, os.path.join(save_folder, "train.jsonl")) # Make the "train set" the test set, since we want it to be the majority
    save_to_jsonl(val_data, os.path.join(save_folder, "val.jsonl"))

    print("Data splits saved in folder:", save_folder)        
    print(train_data[0])
    print(len(train_data), len(test_data), len(val_data))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make data splits")

    parser.add_argument("--val_split", type=float, default=0.05) # Others: wikiMIA, Pile?
    parser.add_argument("--test_split", type=float, default=0.05) # Others: wikiMIA, Pile?
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    main(args)

    """
    # To run this file, use:
    python3 -m data.pile_external.preprocess
    """
