from code.utils import load_jsonl, save_to_jsonl, convert_to_tulu_v1_format
from IPython import embed
import numpy as np
import argparse

def main(args):
    data = []
    for split in args.split: # Iterate over all splits of the data
        data_path = f"data/{args.task}/split-random-overall/{split}.jsonl"
        cur_data = load_jsonl(data_path)
        data.extend(cur_data)

    if args.task == "tulu_v1":
        member_lengths_prompt = []
        member_lengths_output = []
        nonmember_lengths_prompt = []
        nonmember_lengths_output = []

        for d in data:
            if d["label"] == 0: # Nonmember
                nonmember_lengths_prompt.append(len(d["messages"][0]["content"].split()))
                nonmember_lengths_output.append(len(d["messages"][1]["content"].split()))
            
            else: # Member
                member_lengths_prompt.append(len(d["messages"][0]["content"].split()))
                member_lengths_output.append(len(d["messages"][1]["content"].split()))

        # Compute statistics
        # Compute statistics
        stats = {
            "member_prompt_avg": np.mean(member_lengths_prompt),
            "member_prompt_std": np.std(member_lengths_prompt),
            "member_output_avg": np.mean(member_lengths_output),
            "member_output_std": np.std(member_lengths_output),
            "member_total_avg": np.mean(np.array(member_lengths_prompt) + np.array(member_lengths_output)),
            "member_total_std": np.std(np.array(member_lengths_prompt) + np.array(member_lengths_output)),

            "nonmember_prompt_avg": np.mean(nonmember_lengths_prompt),
            "nonmember_prompt_std": np.std(nonmember_lengths_prompt),
            "nonmember_output_avg": np.mean(nonmember_lengths_output),
            "nonmember_output_std": np.std(nonmember_lengths_output),
            "nonmember_total_avg": np.mean(np.array(nonmember_lengths_prompt) + np.array(nonmember_lengths_output)),
            "nonmember_total_std": np.std(np.array(nonmember_lengths_prompt) + np.array(nonmember_lengths_output)),
        }

        print(f"Stats for {args.split}: stats")

        # Print in CSV-friendly format
        print("category,prompt_avg,prompt_std,output_avg,output_std,total_avg,total_std")
        print(f"member,{stats['member_prompt_avg']:.2f},{stats['member_prompt_std']:.2f},{stats['member_output_avg']:.2f},{stats['member_output_std']:.2f},{stats['member_total_avg']:.2f},{stats['member_total_std']:.2f}")
        print(f"nonmember,{stats['nonmember_prompt_avg']:.2f},{stats['nonmember_prompt_std']:.2f},{stats['nonmember_output_avg']:.2f},{stats['nonmember_output_std']:.2f},{stats['nonmember_total_avg']:.2f},{stats['nonmember_total_std']:.2f}")
    
    else:
        member_lengths = []
        nonmember_lengths = []

        for d in data:
            snippet_length = len(d[args.key_name].split())
            if d["label"] == 0:  # Nonmember
                nonmember_lengths.append(snippet_length)
            else:  # Member
                member_lengths.append(snippet_length)

        # Compute statistics
        stats = {
            "member_avg": np.mean(member_lengths),
            "member_std": np.std(member_lengths),
            "nonmember_avg": np.mean(nonmember_lengths),
            "nonmember_std": np.std(nonmember_lengths),
        }

        print(f"Stats for {args.split}: stats")

        # Print in CSV-friendly format
        print("category,avg,std")
        print(f"member,{stats['member_avg']:.2f},{stats['member_std']:.2f}")
        print(f"nonmember,{stats['nonmember_avg']:.2f},{stats['nonmember_std']:.2f}")


    embed()
    # TODO save?
    output_dir = "code/analysis/{args.task}"
    os.path.makedirs(output_dir)
    embed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--task', type=str, default="pile_external", help="the task (dataset)")
    parser.add_argument('--split', type=str, nargs="+", default=None)
    parser.add_argument('--key_name', type=str, default="snippet")

    main(parser.parse_args())

"""
python3 -m code.analysis.avg_data_lengths --task tulu_v1 --split train

python3 -m code.analysis.avg_data_lengths --task tulu_v1 --split val
"""