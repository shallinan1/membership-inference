import argparse
from user_secrets import CACHE_PATH
import os
from datasets import load_dataset
from generate.openai_generate import get_gpt_output
from nltk import sent_tokenize
import pandas as pd
from IPython import embed
from tqdm import tqdm

# Set up environment variables
os.environ["HF_HOME"] = CACHE_PATH
os.environ["HF_DATASETS_PATH"] = CACHE_PATH

# Function to define the main process
def main(args):
    # Task prompts
    task_prompts = [
        "Given the following sentence from a novel, continue the narrative while keeping the style and context consistent: ",
        "Write a novel: ",
        "Continue the story: ",
        ""
    ]

    # Load dataset
    ds = load_dataset("swj0419/BookMIA")
    df = ds["train"].to_pandas()

    # Filter data based on snippet counts
    snippet_value_counts = df.snippet_id.value_counts()
    valid_snippet_ids = snippet_value_counts[snippet_value_counts == 100].index
    filtered_df = df[df.snippet_id.isin(valid_snippet_ids)]

    # Select 5 rows with label 0 and 1
    label_0_subset = filtered_df[filtered_df.label == 0].head(5)
    label_1_subset = filtered_df[filtered_df.label == 1].head(5)

    # Combine the two subsets
    final_subset = pd.concat([label_0_subset, label_1_subset])

    # Prepare to save generations
    save_folder = "tasks/bookMIA/generations"
    os.makedirs(save_folder, exist_ok=True)

    # Add a column to save generations
    final_subset["generation"] = ""

    # Generate text and save to DataFrame
    task_prompt = task_prompts[args.task_prompt_idx]
    for index, row in tqdm(final_subset.iterrows(), total=len(final_subset)):
        snippet_sentences = sent_tokenize(row.snippet)
        # Ignore first one, since it is often partial
        if args.num_sentences == 1:
            prompt_text = snippet_sentences[1]
        else:
            prompt_text = " ".join(snippet_sentences[1:1 + args.num_sentences])

        prompt = task_prompt + prompt_text

        generation = get_gpt_output(prompt, model=args.model, max_tokens=args.max_tokens, n=args.num_sequences, top_p=args.top_p)

        # Save the generation in the DataFrame
        final_subset.at[index, "generation"] = generation

    # Save DataFrame to CSV with detailed info in the filename
    file_name = f"{args.model}_maxTokens{args.max_tokens}_numSeq{args.num_sequences}_topP{args.top_p}_numSent{args.num_sentences}_promptIdx{args.task_prompt_idx}.jsonl"
    file_path = os.path.join(save_folder, file_name)
    final_subset.to_json(file_path, index=False, lines=True, orient='records')

# Argument parser setup
def parse_args():
    parser = argparse.ArgumentParser(description="Generate text using GPT models.")
    parser.add_argument('--max_tokens', type=int, default=512, help='Maximum number of tokens to generate.')
    parser.add_argument('--num_sequences', type=int, default=1, help='Number of sequences to generate.')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p sampling value.')
    parser.add_argument('--num_sentences', type=int, default=1, help='Number of sentences to use from the snippet.')
    parser.add_argument('--task_prompt_idx', type=int, default=1, help='Index of the task prompt to use.')
    parser.add_argument('--model', type=str, default="davinci-002", help='Model to use for text generation.')
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments and call main function
    args = parse_args()
    main(args)
