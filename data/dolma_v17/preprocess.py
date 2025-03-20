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
    dataset_member = load_dataset("emozilla/dolma-v1_7-3B")["train"] # DOLMA 1.7 0.01% subsample, so we don't need to process ourselves
    dataset_nonmember = load_dataset("allenai/paloma", "dolma-v1_5")["test"] # Paloma evaluation set
    random.seed(args.seed)

    data_member = dataset_member.shuffle(seed=args.seed).select(range(500))
    data_member = data_member.add_column("label", [1] * len(data_member))
    data_member = data_member.rename_column("text", "snippet")    
    
    data_nonmember = dataset_nonmember.shuffle(seed=args.seed).select(range(500))
    data_nonmember = data_nonmember.add_column("label", [0] * len(data_nonmember))
    data_nonmember = data_nonmember.rename_column("text", "snippet")

    data_member = [d for d in data_member]
    data_nonmember = [d for d in data_nonmember]

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

    save_folder = os.path.join("data", "dolma_v17", "split-random-overall")
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
    python3 -m data.dolma_v17.preprocess
    """
