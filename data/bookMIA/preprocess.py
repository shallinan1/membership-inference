import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
CACHE_PATH = os.getenv("CACHE_PATH")

# Set up environment variables
os.environ["HF_HOME"] = CACHE_PATH
os.environ["HF_DATASETS_PATH"] = CACHE_PATH

from datasets import load_dataset
import random
import numpy as np
import argparse
import os
import pandas as pd
from data.bookMIA.utils import clean_snippet
from sklearn.model_selection import train_test_split

def main(args):
    if "bookMIA" in args.datasets:
        # Load dataset
        ds = load_dataset("swj0419/BookMIA")
        df = ds["train"].to_pandas()

        # Jane Eyre is loaded incorrectly??
        problematic_rows = df[df['snippet'].str.contains('\x00', regex=False)].index
        for idx in problematic_rows: # Replace '\x00' with regex in problematic rows
            df.at[idx, 'snippet'] = clean_snippet(df.at[idx, 'snippet'])

        # # Filter data based on snippet counts
        # snippet_value_counts = df.snippet_id.value_counts()
        # valid_snippet_ids = snippet_value_counts[snippet_value_counts >= 20].index
        # filtered_df = df[df.snippet_id.isin(valid_snippet_ids)]

        # subsets = []
        # # Extract k=20 snippets with the same "book_id" for 20 different "book_ids" (total 400).
        # # Half of book_ids should have label 0, the other half should have label 1
        # for label in [0,1]:
        #     valid_ids = list(np.unique(filtered_df[filtered_df.label == label].book_id.tolist()))
        #     cur_ids = random.sample(valid_ids, 20)

        #     print(cur_ids)
        #     # Filter the dataframe to include only selected book_ids for this label
        #     selected_books_df = filtered_df[filtered_df.book_id.isin(cur_ids)]

        #     # Take the same snippets
        #     snippets_ids = list(np.arange(0, 99, 5))

        #     subset_df = selected_books_df[selected_books_df.snippet_id.isin(snippets_ids)]
        #     subsets.append(subset_df)

        # final_subset = pd.concat(subsets)

        # Make train, val, and test split
        train_df, temp_df = train_test_split(df, test_size=args.val_split + args.test_split, random_state=args.seed)
        val_split_adjusted = args.val_split / (args.val_split + args.test_split)
        val_df, test_df = train_test_split(temp_df, test_size=1 - val_split_adjusted, random_state=args.seed)

        save_folder = os.path.join("data", "bookMIA", "split-random-overall")
        os.makedirs(save_folder, exist_ok=True)
        
        # Save train, val, and test splits
        train_df.to_json(os.path.join(save_folder, "test.jsonl"), lines=True, orient='records')
        test_df.to_json(os.path.join(save_folder, "train.jsonl"), lines=True, orient='records') # Make the "train set" the test set, since we want it to be the majority
        val_df.to_json(os.path.join(save_folder, "val.jsonl"), lines=True, orient='records')

        print("Data splits saved in folder:", save_folder)        
        print(train_df.loc[0])
        print(len(train_df), len(test_df), len(val_df))
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
    python3 -m data.bookMIA.preprocess
    """
