from user_secrets import CACHE_PATH

import os
os.environ["HF_HOME"] = CACHE_PATH
os.environ["HF_DATASETS_PATH"] = CACHE_PATH
from datasets import load_dataset
from generate.openai_generate import get_gpt_output
from nltk import sent_tokenize
import pandas as pd
from IPython import embed
from tqdm import tqdm


task_prompts = ["Given the following sentence from a novel, continue the narrative while keeping the style and context consistent: ",
                "Write a novel: ",
                "Continue the story: "
                ""
                ]
max_tokens=512
num_sequences=1
top_p=0.95
num_sentences = 1
task_prompt_idx=1
model="davinci-003"

ds = load_dataset("swj0419/BookMIA")
df = ds["train"].to_pandas()

snippet_value_counts = df.snippet_id.value_counts()
valid_snippet_ids = snippet_value_counts[snippet_value_counts == 100].index

# Filter the DataFrame to include only those rows with snippet_id in 'valid_snippet_ids'
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
task_prompt = task_prompts[task_prompt_idx]
for index, row in tqdm(final_subset.iterrows()):
    snippet_sentences = sent_tokenize(row.snippet)
    # Ignore first one, since it is often partial
    if num_sentences == 1:
        prompt_text = snippet_sentences[1]
    else:
        prompt_text = " ".join(snippet_sentences[1:1+num_sentences])

    prompt = task_prompt + prompt_text

    generation = get_gpt_output(prompt, model=model, max_tokens=max_tokens, n=num_sequences, top_p=0.95)

    # Save the generation in the DataFrame
    final_subset.at[index, "generation"] = generation
    embed()

# Save DataFrame to CSV with detailed info in the filename
file_name = f"{model}_maxTokens{max_tokens}_numSeq{num_sequences}_topP{top_p}_numSent{num_sentences}_promptIdx{task_prompt_idx}.jsonl"
file_path = os.path.join(save_folder, file_name)
final_subset.to_json(file_path, index=False, lines=True)