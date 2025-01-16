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

def main(args):
    subset = "books3" # TODO change this
   
    base_data_path = "/data/pile"
    data_nonmember = load_jsonl(os.path.join(base_data_path, "test.jsonl"))
    # Filter the data into the subsets
    subset_data_nonmember = []
    for d in data_nonmember:
        if d['meta']['pile_set_name'] == "Books3":
            subset_data_nonmember.append({"snippet": d["text"], "label": 0})
            if len(subset_data_nonmember) == 100:
                break
    del data_nonmember # Memory saving

    # subset_data_member = []
    # data_member = load_jsonl(os.path.join(base_data_path, "train", "00.jsonl")) # TODO iterate over all tarining files
    # for d in data_member:
    #     if d['meta']['pile_set_name'] == "Books3":
    #         subset_data_member.append({"snippet": d["text"], "label": 1})
    #         if len(subset_data_member) == 100:
    #             break
    # del data_member
    subset_data_member = subset_data_nonmember.copy()

    nonmember_train, nonmember_temp = train_test_split(subset_data_nonmember, test_size=args.val_split + args.test_split, random_state=args.seed)
    val_split_adjusted = args.val_split / (args.val_split + args.test_split)
    nonmember_val, nonmember_test = train_test_split(nonmember_temp, test_size=1 - val_split_adjusted, random_state=args.seed)

    # Create splits for member data
    member_train, member_temp = train_test_split(subset_data_member, test_size=args.val_split + args.test_split, random_state=args.seed)
    val_split_adjusted = args.val_split / (args.val_split + args.test_split)
    member_val, member_test = train_test_split(member_temp, test_size=1 - val_split_adjusted, random_state=args.seed)

    save_folder = os.path.join("data", "pile", "split-random-overall")
    os.makedirs(save_folder, exist_ok=True)
    
    train_data = nonmember_train + member_train
    val_data = nonmember_val + member_val
    test_data = nonmember_test + member_test

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # Save train, val, and test splits
    save_to_jsonl(train_data, os.path.join(save_folder, "test.jsonl"))
    save_to_jsonl(test_data, os.path.join(save_folder, "train.jsonl")) # Make the "train set" the test set, since we want it to be the majority
    save_to_jsonl(val_data, os.path.join(save_folder, "val.jsonl"))

    print("Data splits saved in folder:", save_folder)        
    print(train_data[0])
    print(len(train_data), len(test_data), len(val_data))
    # TODO saving for data level inference?

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make data splits")

    parser.add_argument("--datasets", type=str, nargs='+', help="List of dataset names to load", default=["bookMIA"]) # Others: wikiMIA, Pile?
    parser.add_argument("--val_split", type=float, default=0.05) # Others: wikiMIA, Pile?
    parser.add_argument("--test_split", type=float, default=0.05) # Others: wikiMIA, Pile?
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    main(args)

    """
    # To run this file, use:
    python3 -m data.pile.preprocess
    """
