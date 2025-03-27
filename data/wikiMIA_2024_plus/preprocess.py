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
from code.utils import load_jsonl
import re

bad_words = ["is a surname", "is the name of", "surname", "include:\n\n", ":\n\n", "refer to:", "can mean:", "This is a list", "refers to"]


def main(args):  
    raw_data = load_jsonl("data/wikiMIA_2024_plus/scraped/scraped.jsonl")
    random.shuffle(raw_data)
    limit = 112
    data_member = []
    data_nonmember = []
    for r in raw_data:
        bad=False
        for b in bad_words:
            if b in r["summary"]:
                bad=True
                break
        if bad:
            continue
        r["input"] = r["summary"]
        r.pop("summary")
        r["words"] = len(r["input"].split())
        r["label"] = 1 - r["label"] # Flip the label due to error

        if r["words"] < 25:
            continue

        r_label = r["label"]

        if r_label == 1 and len(data_member) < limit:
            data_member.append(r)

        elif r_label == 0 and len(data_nonmember) < limit:
            data_nonmember.append(r)
    
    min_length = min(len(data_member), len(data_nonmember))
    data_member = data_member[:min_length]
    data_nonmember = data_nonmember[:min_length]

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


    save_folder = os.path.join("data", "wikiMIA_2024_plus", "split-random-overall")
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

    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--test_split", type=float, default=0.05)
    parser.add_argument("--min_percentile", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    main(args)

    """
    # To run this file, use:
    python3 -m data.wikiMIA_2024_plus.preprocess
    """
