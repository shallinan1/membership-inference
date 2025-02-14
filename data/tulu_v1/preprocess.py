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
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_word_counts(sample):
    user_message = sample["messages"][0]["content"]
    response_message = sample["messages"][1]["content"]

    user_length, response_length = len(user_message.split()), len(response_message.split())

    return {
        "user_length": user_length, 
        "response_length": response_length, 
        "total_length": user_length+response_length
    }

def calculate_percentiles(data, 
                          fields=["user_length", "response_length", "total_length"], 
                          percentiles=[0,1, 5, 50, 95, 99, 100]):
    """
    Calculate percentiles for specified fields in the dataset.
    """
    percentile_dict = {}
    for field in fields:
        values = [sample[field] for sample in data]
        percentile_dict[field] = [int(x) for x in np.percentile(values, percentiles)]
    return percentile_dict

def plot_length_histogram(data_member, data_nonmember, field, title):
    """
    Plot histogram of lengths for a given field (e.g., "user_length", "response_length", "total_length").
    The upper and lower ends are cut off at the 95th percentile.
    Histograms for Member and Nonmember are displayed in separate subplots.
    """
    # Extract lengths from both datasets
    member_lengths = [sample[field] for sample in data_member]
    nonmember_lengths = [sample[field] for sample in data_nonmember]
    
    # Combine lengths to calculate percentiles
    all_lengths = member_lengths + nonmember_lengths
    
    # Calculate 2.5th and 97.5th percentiles (to cut off 5% from both ends)
    lower_cutoff = np.percentile(all_lengths, 5)
    upper_cutoff = np.percentile(all_lengths, 95)
    
    # Filter lengths to include only values within the cutoffs
    member_lengths_filtered = [x for x in member_lengths if lower_cutoff <= x <= upper_cutoff]
    nonmember_lengths_filtered = [x for x in nonmember_lengths if lower_cutoff <= x <= upper_cutoff]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot Member histogram
    ax1.hist(member_lengths_filtered, bins=4, alpha=0.5, label="Member", color="blue")
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"Member: {title} (95th Percentile Cutoff)")
    ax1.legend()
    
    # Plot Nonmember histogram
    ax2.hist(nonmember_lengths_filtered, bins=4, alpha=0.5, label="Nonmember", color="orange")
    ax2.set_xlabel(field)
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"Nonmember: {title} (95th Percentile Cutoff)")
    ax2.legend()
    
    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(f"{title}.png", bbox_inches="tight", dpi=200)
    
def main(args):   
    random.seed(args.seed)
    np.random.seed(args.seed)

    data_member = load_jsonl("data/tulu_v1/processed/tulu_v1/tulu_v1_data.jsonl")
    data_nonmember = load_jsonl("data/tulu_v1/processed/tulu_v1/inverse_tulu_v1_data.jsonl")
    random.shuffle(data_member)
    random.shuffle(data_nonmember)

    data_member_classes, data_nonmember_classes = set(), set()
    for d in data_member:
        d["label"] = 1
        data_member_classes.add(d["dataset"])
    for d in data_nonmember:
        d["label"] = 0
        data_nonmember_classes.add(d["dataset"])

    for sample in tqdm(data_member):
        sample.update(calculate_word_counts(sample))
    for sample in tqdm(data_nonmember):
        sample.update(calculate_word_counts(sample))

    # Calculate percentiles for non-merged data
    percentiles_member = calculate_percentiles(data_member)
    percentiles_nonmember = calculate_percentiles(data_nonmember)
    percentiles_merged = calculate_percentiles(data_member + data_nonmember)
    print("Percentiles for data_member:", percentiles_member)
    print("Percentiles for data_nonmember:", percentiles_nonmember)
    print("Percentiles for merged data:", percentiles_merged)

    # Example usage
    plot_length_histogram(data_member, data_nonmember, "user_length", "Histogram of User Lengths")
    plot_length_histogram(data_member, data_nonmember, "response_length", "Histogram of Response Lengths")
    plot_length_histogram(data_member, data_nonmember, "total_length", "Histogram of Total Lengths")

    n_bins = 4

    # Remove data that is too short or too long
    all_user_lengths = [sample["user_length"] for sample in data_member + data_nonmember]
    all_response_lengths = [sample["response_length"] for sample in data_member + data_nonmember]
    
    # Prune extreme length responses
    user_lower_cutoff = np.percentile(all_user_lengths, 5)
    user_upper_cutoff = np.percentile(all_user_lengths, 95)
    response_lower_cutoff = np.percentile(all_response_lengths, 5)
    response_upper_cutoff = np.percentile(all_response_lengths, 95)

    data_member_new, data_nonmember_new = [], []
    for d in data_member:
        if (
            user_lower_cutoff < d["user_length"] < user_upper_cutoff
            and response_lower_cutoff < d["response_length"] < response_upper_cutoff
        ):
            data_member_new.append(d)
    for d in data_nonmember:
        if (
            user_lower_cutoff < d["user_length"] < user_upper_cutoff
            and response_lower_cutoff < d["response_length"] < response_upper_cutoff
        ):
            data_nonmember_new.append(d)

    print("Old/new member lengths:", len(data_member), len(data_member_new))
    print("Old/new nonmember lengths:", len(data_nonmember), len(data_nonmember_new))
    data_member = data_member_new
    data_nonmember = data_nonmember_new

    # TODO Sample equally from data to get uniformly distributed both user length and response length
    def sample_uniformly_by_length(data_member, data_nonmember, n_bins=4):
        """
        Sample data uniformly based on user_length and response_length bins.
        """
        # Combine data to calculate percentiles for bin edges
        all_user_lengths = [sample["user_length"] for sample in data_member + data_nonmember]
        all_response_lengths = [sample["response_length"] for sample in data_member + data_nonmember]

        plt.figure()
        plt.hist(all_user_lengths,bins=100)
        plt.tight_layout()
        plt.savefig("all_user_lengths.png", dpi=200,bbox_inches="tight")


        plt.figure()
        plt.hist(all_response_lengths,bins=100)
        plt.tight_layout()
        plt.savefig("all_response_lengths.png", dpi=200,bbox_inches="tight")


        # Create bins for user_length and response_length
        user_bin_edges = np.percentile(all_user_lengths, np.linspace(0, 100, n_bins + 1))
        response_bin_edges = np.percentile(all_response_lengths, np.linspace(0, 100, n_bins + 1))
        
        user_bin_edges[-1] += 1 # Fix bug for longest sequence in assign_to_bins
        response_bin_edges[-1] += 1

        print(user_bin_edges)
        print(response_bin_edges)

        # Assign samples to bins
        def assign_to_bins(data, user_bin_edges, response_bin_edges):
            binned_data = [[[] for _ in range(n_bins)] for _ in range(n_bins)]
            for sample in data:
                user_bin = np.digitize(sample["user_length"], user_bin_edges) - 1
                response_bin = np.digitize(sample["response_length"], response_bin_edges) - 1
                assert user_bin >= 0 and user_bin < n_bins and response_bin >= 0 and response_bin < n_bins
                binned_data[user_bin][response_bin].append(sample)
            return binned_data
        
        binned_member = assign_to_bins(data_member, user_bin_edges, response_bin_edges)
        binned_nonmember = assign_to_bins(data_nonmember, user_bin_edges, response_bin_edges)
        
        bin_counts_dict_total = {i: {j: 0 for j in range(n_bins)} for i in range(n_bins)}
        for i in range(n_bins):
            for j in range(n_bins):
                bin_counts_dict_total[i][j] += len(binned_member[i][j]) + len(binned_nonmember[i][j])
        total_sum = sum(sum(inner.values()) for inner in bin_counts_dict_total.values())
        for i in bin_counts_dict_total:
            for j in bin_counts_dict_total[i]:
                bin_counts_dict_total[i][j] /= total_sum

        # Sample from each bin according to bin_counts_dict_total
        sampled_member, sampled_nonmember = [], []
        for i in range(n_bins):
            for j in range(n_bins):
                member_samples = binned_member[i][j]
                nonmember_samples = binned_nonmember[i][j]

                sample_amount = int(500 * bin_counts_dict_total[i][j])

                sampled_member.extend(random.sample(member_samples, sample_amount))
                sampled_nonmember.extend(random.sample(nonmember_samples, sample_amount))
        
        return sampled_member, sampled_nonmember


    sampled_member, sampled_nonmember = sample_uniformly_by_length(data_member, data_nonmember, n_bins)
    embed()
        
    # member_train, member_temp = train_test_split(data_member, test_size=args.val_split + args.test_split, random_state=args.seed)
    # val_split_adjusted = args.val_split / (args.val_split + args.test_split)
    # member_val, member_test = train_test_split(member_temp, test_size=1 - val_split_adjusted, random_state=args.seed)

    # # Split nonmember and member data
    # nonmember_train, nonmember_temp = train_test_split(data_nonmember, test_size=args.val_split + args.test_split, random_state=args.seed)
    # val_split_adjusted = args.val_split / (args.val_split + args.test_split)
    # nonmember_val, nonmember_test = train_test_split(nonmember_temp, test_size=1 - val_split_adjusted, random_state=args.seed)

    # # Recombine and shuffle
    # train_data = nonmember_train + member_train
    # val_data = nonmember_val + member_val
    # test_data = nonmember_test + member_test

    # random.shuffle(train_data)
    # random.shuffle(val_data)
    # random.shuffle(test_data)

    # save_folder = os.path.join("data", "tulu_v1", "split-random-overall")
    # os.makedirs(save_folder, exist_ok=True)

    # # Save train, val, and test splits
    # save_to_jsonl(train_data, os.path.join(save_folder, "test.jsonl"))
    # save_to_jsonl(test_data, os.path.join(save_folder, "train.jsonl")) # Make the "train set" the test set, since we want it to be the majority
    # save_to_jsonl(val_data, os.path.join(save_folder, "val.jsonl"))

    # print("Data splits saved in folder:", save_folder)        
    # print(train_data[0])
    # print(len(train_data), len(test_data), len(val_data))
    # embed()

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

