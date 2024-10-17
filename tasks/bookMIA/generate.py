import argparse
from user_secrets import CACHE_PATH
import os
# Set up environment variables
os.environ["HF_HOME"] = CACHE_PATH
os.environ["HF_DATASETS_PATH"] = CACHE_PATH

from datasets import load_dataset
from generate.openai_generate import get_gpt_output
from nltk import sent_tokenize
import pandas as pd
from IPython import embed
from tqdm import tqdm
import re
from generate.vllm_generate import ModelGenerator 
from generate.generate_utils import task_prompts_dict_book, make_prompts
import numpy as np
import random
from datetime import datetime

def extract_sentence_chunk(text, start_sentence, num_sentences):
    text_sentences = sent_tokenize(text)
    # Ignore first one, since it is often partial
    if num_sentences == 1:
        try:
            prompt_text = text_sentences[args.start_sentence]
        except:
            print("Something wrong")
            return None
    else:
        prompt_text = " ".join(text_sentences[start_sentence:start_sentence + num_sentences])
        rest_of_text =  " ".join(text_sentences[start_sentence + num_sentences:])
    return prompt_text, rest_of_text

# Function to define the main process
def main(args):
    random.seed(0)

    model_str = args.model.split("/")[-1] # Splits to get actual model name
    if model_str not in task_prompts_dict_book:
        print("Valid model not passed in. Try again")
    cur_task_prompts = task_prompts_dict_book[model_str][args.task_prompt_idx]

    # Load dataset
    ds = load_dataset("swj0419/BookMIA")
    df = ds["train"].to_pandas()

    # Jane Eyre is loaded incorrectly??
    problematic_rows = df[df['snippet'].str.contains('\x00', regex=False)].index

    # Function to clean the 'snippet' by replacing '\x00' with an empty string
    def clean_snippet(snippet):
        if isinstance(snippet, bytes):  # Decode if snippet is in bytes
            snippet = snippet.decode('utf-8', errors='ignore')
        return re.sub(r'\x14', '',re.sub(r'\x00', '', snippet))

    # Replace '\x00' with regex in problematic rows
    for idx in problematic_rows:
        df.at[idx, 'snippet'] = clean_snippet(df.at[idx, 'snippet'])

    # Filter data based on snippet counts
    snippet_value_counts = df.snippet_id.value_counts()
    valid_snippet_ids = snippet_value_counts[snippet_value_counts >= 20].index
    filtered_df = df[df.snippet_id.isin(valid_snippet_ids)]

    subsets = []
    # Extract k=20 snippets with the same "book_id" for 20 different "book_ids" (total 400).
    # Half of book_ids should have label 0, the other half should have label 1
    for label in [0,1]:
        valid_ids = list(np.unique(filtered_df[filtered_df.label == label].book_id.tolist()))
        cur_ids = random.sample(valid_ids, 20)

        print(cur_ids)
        # Filter the dataframe to include only selected book_ids for this label
        selected_books_df = filtered_df[filtered_df.book_id.isin(cur_ids)]

        # Take the same snippets
        snippets_ids = list(np.arange(0, 99, 5))

        subset_df = selected_books_df[selected_books_df.snippet_id.isin(snippets_ids)]
        subsets.append(subset_df)

    final_subset = pd.concat(subsets)

    print(f"Length: {len(final_subset)}")
    # Prepare to save generations
    save_folder = "tasks/bookMIA/generations"
    os.makedirs(save_folder, exist_ok=True)

    # Add a column to save generations
    final_subset["generation"] = ""
    final_subset["model"] = ""
    final_subset["logprobs"] = ""
    final_subset["snippet_"] = ""

    if not args.openai:
        # Initialize ModelGenerator
        generator = ModelGenerator(
            model=args.model,
            tokenizer=args.model if not args.tokenizer else args.tokenizer,
            seed=args.seed,
            hf_token=args.hf_token,
            cache_dir=CACHE_PATH
        )

    if args.openai:
        first_gen = True
        for index, row in tqdm(final_subset.iterrows(), total=len(final_subset), desc="Generating"):
            prompt_text, rest_of_text = extract_sentence_chunk(row.snippet, args.start_sentence, args.num_sentences)
            assert prompt_text is not None
                
            prompt = make_prompts(
                prompt_text, 
                cur_task_prompts["task_prompt"], 
                cur_task_prompts["task_preprompt"],
                cur_task_prompts["task_postprompt"]
                )[0]

            if first_gen:
                print(prompt)
                first_gen=False

            full_generations = get_gpt_output(prompt, 
                                              model=args.model, 
                                              max_tokens=args.max_tokens, 
                                              n=args.num_sequences, 
                                              top_p=args.top_p)
            
            if args.model == 'gpt-3.5-turbo-instruct' or any([args.model.startswith(x) for x in ['babbage', 'davinci']]):
                generations = [r['text'] for r in full_generations['choices']]

                # Store model too
                models = [full_generations["model"]] * len(generations)
                
                # Can also store the logprobs
                logprobs =  [r['logprobs'] for r in full_generations['choices']]
            else:
                generations = [r['message']['content'] for r in full_generations['choices']]
                # Store model too
                models = [full_generations["model"]] * len(generations)
                # Can also store the logprobs
                logprobs =  [r['logprobs'] for r in full_generations['choices']]

            # Save the generation in the DataFrame
            final_subset.at[index, "generation"] = generations
            final_subset.at[index, "model"] = models
            final_subset.at[index, "logprobs"] = logprobs
            final_subset.at[index, "snippet_no_prompt"] = rest_of_text

    else:
        passages = final_subset.snippet.tolist()
        prompt_outputs = [extract_sentence_chunk(text, args.start_sentence, args.num_sentences) for text in passages]
        
        prompt_texts, rest_of_texts = zip(*prompt_outputs)
        prompt_texts= list(prompt_texts)
        rest_of_texts = list(rest_of_texts)

        assert None not in prompt_texts

        prompts = make_prompts(
            prompt_texts, 
            cur_task_prompts["task_prompt"], 
            cur_task_prompts["task_preprompt"],
            cur_task_prompts["task_postprompt"]
            )
      
        # Generate texts
        final_prompts, all_text_outputs, all_prompt_logprobs, all_output_logprobs = generator.generate_vllm(
            prompts=prompts,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_tokens,
            max_length=args.max_length,
            n=args.num_sequences
        )

        final_subset["generation"] =all_text_outputs
        final_subset["model"] = [model_str] * len(final_subset)
        final_subset["logprobs"] = all_output_logprobs
        final_subset["logprobs_prompt"] = all_prompt_logprobs
        final_subset["snippet_no_prompt"] = rest_of_texts

    # Convert current datetime to string in 'YYYY-MM-DD HH:MM:SS' format
    date_str = datetime.now().strftime("%Y-%m-%d-%H:%M:%S").strip()

    # Save DataFrame to CSV with detailed info in the filename
    file_name = f"{model_str}_maxTokens{args.max_tokens}_numSeq{args.num_sequences}_topP{args.top_p}_numSent{args.num_sentences}_startSent{args.start_sentence}_promptIdx{args.task_prompt_idx}_len{len(final_subset)}_{date_str}.jsonl"
    file_path = os.path.join(save_folder, file_name)
    columns = [col for col in final_subset.columns if col != 'snippet'] + ['snippet']
    final_subset = final_subset[columns]
    final_subset.to_json(file_path, index=False, lines=True, orient='records')
    

# Argument parser setup
def parse_args():
    parser = argparse.ArgumentParser(description="Generate text using GPT models.")
    parser.add_argument('--max_tokens', type=int, default=512, help='Maximum number of tokens to generate.')
    parser.add_argument('--max_length', type=int, default=2048, help='Maximum length')
    parser.add_argument('--num_sequences', type=int, default=1, help='Number of sequences to generate.')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p sampling value.')
    parser.add_argument('--num_sentences', type=int, default=1, help='Number of sentences to use from the snippet.')
    parser.add_argument('--start_sentence', type=int, default=1, help='Number of sentences to use from the snippet.')
    parser.add_argument('--task_prompt_idx', type=int, default=1, help='Index of the task prompt to use.')
    parser.add_argument('--model', type=str, default="davinci-002", help='Model to use for text generation.')
    parser.add_argument('--tokenizer', type=str, default=None, help='Pass in tokenizer manually. Optional.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Model sampling temperature')
    parser.add_argument('--hf_token', type=str, default=None, help='Pass in tokenizer manually. Optional.')
    parser.add_argument("--openai", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments and call main function
    args = parse_args()
    main(args)


"""
python3 -m tasks.bookMIA.generate \
    --openai \
    --model gpt-4o-mini-2024-07-18 \
    --start_sentence 1 \
    --num_sentences 5 \
    --num_sequences 10 \
    --max_tokens 512 \
    --task_prompt_idx 5;

python3 -m tasks.bookMIA.generate \
    --openai \
    --model gpt-4o-mini-2024-07-18 \
    --start_sentence 1 \
    --num_sentences 3 \
    --num_sequences 10 \
    --max_tokens 512 \
    --task_prompt_idx 5;

python3 -m tasks.bookMIA.generate \
    --openai \
    --model gpt-4o-2024-05-13 \
    --start_sentence 1 \
    --num_sentences 5 \
    --num_sequences 10 \
    --max_tokens 512 \
    --task_prompt_idx 5;

python3 -m tasks.bookMIA.generate \
    --openai \
    --model gpt-4o-2024-05-13 \
    --start_sentence 1 \
    --num_sentences 3 \
    --num_sequences 10 \
    --max_tokens 512 \
    --task_prompt_idx 5;

python3 -m tasks.bookMIA.generate \
    --openai \
    --model gpt-3.5-turbo-0125 \
    --start_sentence 1 \
    --num_sentences 5 \
    --num_sequences 10 \
    --max_tokens 512 \
    --task_prompt_idx 5;

python3 -m tasks.bookMIA.generate \
    --openai \
    --model gpt-3.5-turbo-0125 \
    --start_sentence 1 \
    --num_sentences 3 \
    --num_sequences 10 \
    --max_tokens 512 \
    --task_prompt_idx 5;
    
python3 -m tasks.bookMIA.generate \
    --openai \
    --model gpt-4o-mini-2024-07-18 \
    --start_sentence 1 \
    --num_sentences 5 \
    --num_sequences 10 \
    --max_tokens 512 \
    --task_prompt_idx 0;

python3 -m tasks.bookMIA.generate \
    --openai \
    --model gpt-4o-mini-2024-07-18 \
    --start_sentence 1 \
    --num_sentences 3 \
    --num_sequences 10 \
    --max_tokens 512 \
    --task_prompt_idx 0;

python3 -m tasks.bookMIA.generate \
    --openai \
    --model gpt-3.5-turbo-0125 \
    --start_sentence 1 \
    --num_sentences 5 \
    --num_sequences 10 \
    --max_tokens 512 \
    --task_prompt_idx 0;

python3 -m tasks.bookMIA.generate \
    --openai \
    --model gpt-3.5-turbo-0125 \
    --start_sentence 1 \
    --num_sentences 3 \
    --num_sequences 10 \
    --max_tokens 512 \
    --task_prompt_idx 0;
    
python3 -m tasks.bookMIA.generate \
    --openai \
    --model gpt-4o-2024-05-13 \
    --start_sentence 1 \
    --num_sentences 5 \
    --num_sequences 10 \
    --max_tokens 512 \
    --task_prompt_idx 0;

python3 -m tasks.bookMIA.generate \
    --openai \
    --model gpt-4o-2024-05-13 \
    --start_sentence 1 \
    --num_sentences 3 \
    --num_sequences 10 \
    --max_tokens 512 \
    --task_prompt_idx 0;


python3 -m tasks.bookMIA.generate \
    --openai \
    --model gpt-4o-mini-2024-07-18 \
    --start_sentence 1 \
    --num_sentences 5 \
    --num_sequences 10 \
    --max_tokens 512 \
    --task_prompt_idx 4;

python3 -m tasks.bookMIA.generate \
    --openai \
    --model gpt-4o-mini-2024-07-18 \
    --start_sentence 1 \
    --num_sentences 3 \
    --num_sequences 10 \
    --max_tokens 512 \
    --task_prompt_idx 4;

python3 -m tasks.bookMIA.generate \
    --openai \
    --model gpt-3.5-turbo-0125 \
    --start_sentence 1 \
    --num_sentences 5 \
    --num_sequences 10 \
    --max_tokens 512 \
    --task_prompt_idx 4;

python3 -m tasks.bookMIA.generate \
    --openai \
    --model gpt-3.5-turbo-0125 \
    --start_sentence 1 \
    --num_sentences 3 \
    --num_sequences 10 \
    --max_tokens 512 \
    --task_prompt_idx 4;
    
python3 -m tasks.bookMIA.generate \
    --openai \
    --model gpt-4o-2024-05-13 \
    --start_sentence 1 \
    --num_sentences 5 \
    --num_sequences 10 \
    --max_tokens 512 \
    --task_prompt_idx 4;

python3 -m tasks.bookMIA.generate \
    --openai \
    --model gpt-4o-2024-05-13 \
    --start_sentence 1 \
    --num_sentences 3 \
    --num_sequences 10 \
    --max_tokens 512 \
    --task_prompt_idx 4;    

# GPT2-large

CUDA_VISIBLE_DEVICES=0 python3 -m tasks.bookMIA.generate \
    --model openai-community/gpt2-large \
    --start_sentence 1 \
    --num_sentences 3 \
    --num_sequences 20 \
    --max_tokens 512 \
    --task_prompt_idx 0;    

CUDA_VISIBLE_DEVICES=1 python3 -m tasks.bookMIA.generate \
    --model openai-community/gpt2-large \
    --start_sentence 1 \
    --num_sentences 3 \
    --num_sequences 20 \
    --max_tokens 512 \
    --task_prompt_idx 1;

CUDA_VISIBLE_DEVICES=2 python3 -m tasks.bookMIA.generate \
    --model openai-community/gpt2-large \
    --start_sentence 1 \
    --num_sentences 5 \
    --num_sequences 20 \
    --max_tokens 512 \
    --task_prompt_idx 0;   

CUDA_VISIBLE_DEVICES=0 python3 -m tasks.bookMIA.generate \
    --model openai-community/gpt2-large \
    --start_sentence 1 \
    --num_sentences 5 \
    --num_sequences 20 \
    --max_tokens 512 \
    --task_prompt_idx 1;    
    
"""