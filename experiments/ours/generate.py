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
from generate.vllm_generate import ModelGenerator 
from generate.generate_utils import task_prompts_dict_book, make_prompts
import numpy as np
import random
from datetime import datetime

def extract_chunk_words(text, start_word, num_words):
    pass

def extract_chunk_sentence(text, start_sentence, num_sentences):
    text_sentences = sent_tokenize(text)

    # Make it so we at least have the last sentence to generate
    num_sentences = min(len(text_sentences) - start_sentence - 1, num_sentences)

    # Ignore first one, since it is often partial
    if len(text_sentences) == 1:
        embed()
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

    if args.task == "bookMIA":
        data_path = f"data/bookMIA/split-random-overall/{args.data_split}.jsonl"
    final_subset = pd.read_json(data_path, lines=True)
    print(f"Length: {len(final_subset)}")

    save_folder = os.path.join(f"experiments/ours/{args.task}/generations/{args.data_split}")
    os.makedirs(save_folder, exist_ok=True)

    if args.prompt_with_words_not_sent:
        args.start_sent = -1
        args.num_sents = -1
        prompt_with_sent_str = "T"
    else:
        args.start_word = -1
        args.num_words = -1
        prompt_with_sent_str = "F"

    # Add a column to save generations
    final_subset["generation"] = ""
    final_subset["model"] = ""
    final_subset["logprobs"] = ""
    final_subset["snippet_"] = ""

    # Reduce num_sequences if using greedy decoding
    if args.temperature == 0:
        print("GREEDY decoding - setting num_sequences to 1")
        args.num_sequences = 1

    if not args.openai:
        # Initialize ModelGenerator
        generator = ModelGenerator(
            model=args.model,
            tokenizer=args.model if not args.tokenizer else args.tokenizer,
            seed=args.seed,
            hf_token=args.hf_token,
            cache_dir=CACHE_PATH,
        )

    if args.openai:
        first_gen = True
        for index, row in tqdm(final_subset.iterrows(), total=len(final_subset), desc="Generating"):
            prompt_text, rest_of_text = extract_chunk_sentence(row.snippet, args.start_sentence, args.num_sentences)
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
                # TODO fix
                generations = [r['text'] for r in full_generations['choices']]

                # Store model too
                models = [full_generations["model"]] * len(generations)
                
                # Can also store the logprobs
                logprobs =  [r['logprobs'] for r in full_generations['choices']]
            else:
                generations = [r.message.content for r in full_generations.choices]

                # Store model too
                models = [full_generations.model] * len(generations)
                # Can also store the logprobs
                logprobs =  [r.logprobs for r in full_generations.choices]

            # Save the generation in the DataFrame
            final_subset.at[index, "generation"] = generations
            final_subset.at[index, "model"] = models
            final_subset.at[index, "logprobs"] = logprobs
            final_subset.at[index, "snippet_no_prompt"] = rest_of_text

    else:
        passages = final_subset.snippet.tolist()
        
        if not args.prompt_with_words_not_sent:
            prompt_outputs = [extract_chunk_sentence(text, args.start_sentence, args.num_sentences) for text in passages]        
        else:
            # TODO implement this
            import sys; sys.exit()
    
        prompt_texts, rest_of_texts = zip(*prompt_outputs)
        prompt_texts= list(prompt_texts)
        rest_of_texts = list(rest_of_texts)

        assert None not in prompt_texts

        prompts = make_prompts(
            prompt_texts, 
            cur_task_prompts["task_prompt"], 
            cur_task_prompts["task_preprompt"],
            cur_task_prompts["task_postprompt"],
            model_name=model_str,
            prompt_key="lightest"
            )
              
        # Generate texts
        final_prompts, all_text_outputs, all_prompt_logprobs, all_output_logprobs = generator.generate_vllm(
            prompts=prompts,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_tokens,
            min_tokens=args.min_tokens,
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

    minTokStr = "minTok" + str(args.min_tokens) + "_"
    
    # Save DataFrame to CSV with detailed info in the filename
    file_name = f"{model_str}_maxTok{args.max_tokens}_{minTokStr}numSeq{args.num_sequences}_topP{args.top_p}_temp{args.temperature}_numSent{args.num_sentences}_startSent{args.start_sentence}_numWord{args.num_words}_startWord{args.num_words}_useSent{prompt_with_sent_str}_promptIdx{args.task_prompt_idx}_len{len(final_subset)}_{date_str}.jsonl"
    file_path = os.path.join(save_folder, file_name)
    columns = [col for col in final_subset.columns if col != 'snippet'] + ['snippet']
    final_subset = final_subset[columns]
    final_subset.to_json(file_path, index=False, lines=True, orient='records')
    

# Argument parser setup
def parse_args():
    parser = argparse.ArgumentParser(description="Generate text using GPT models.")
    parser.add_argument('--max_tokens', type=int, default=512, help='Maximum number of tokens to generate.')
    parser.add_argument('--min_tokens', type=int, default=0, help='Maximum number of tokens to generate.')
    parser.add_argument('--max_length', type=int, default=2048, help='Maximum length')
    parser.add_argument('--num_sequences', type=int, default=1, help='Number of sequences to generate.')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p sampling value.')

    # What part of the samples to prompt with
    parser.add_argument('--num_sentences', type=int, default=1, help='Number of sentences to use from the snippet.')
    parser.add_argument('--start_sentence', type=int, default=1, help='Starting sentence to use from the snippet.')
    parser.add_argument('--num_words', type=int, default=1, help='Number of words to use from the snippet.')
    parser.add_argument('--start_word', type=int, default=1, help='Starting word to use from the snippet.')
    parser.add_argument("--prompt_with_words_not_sent", action="store_true", help="whether or not to use words vs sentences")

    parser.add_argument('--task_prompt_idx', type=int, default=1, help='Index of the task prompt to use.')
    parser.add_argument('--model', type=str, default="davinci-002", help='Model to use for text generation.')
    parser.add_argument('--tokenizer', type=str, default=None, help='Pass in tokenizer manually. Optional.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Model sampling temperature')
    parser.add_argument('--hf_token', type=str, default=None, help='Pass in tokenizer manually. Optional.')
    parser.add_argument("--openai", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", type=str, default="bookMIA")
    parser.add_argument("--data_split", type=str, default="train")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments and call main function
    args = parse_args()
    main(args)


"""
CUDA_VISIBLE_DEVICES=0 python3 -m experiments.ours.generate \
    --model openai-community/gpt2-large \
    --start_sentence 1 \
    --num_sentences 3 \
    --num_sequences 20 \
    --max_tokens 512 \
    --task_prompt_idx 0;      

CUDA_VISIBLE_DEVICES=0 python3 -m experiments.ours.generate \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --start_sentence 1 \
    --num_sentences 5 \
    --num_sequences 20 \
    --max_tokens 512 \
    --min_tokens 10 \
    --task_prompt_idx 5;    

CUDA_VISIBLE_DEVICES=0 python3 -m experiments.ours.generate \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --start_sentence 1 \
    --num_sentences 5 \
    --num_sequences 20 \
    --max_tokens 512 \
    --min_tokens 10 \
    --task_prompt_idx 5;      

CUDA_VISIBLE_DEVICES=0,1 python3 -m experiments.ours.generate \
    --model meta-llama/Llama-2-70b-chat-hf \
    --start_sentence 1 \
    --num_sentences 5 \
    --num_sequences 20 \
    --max_tokens 512 \
    --task_prompt_idx 5; 

CUDA_VISIBLE_DEVICES=0 python3 -m experiments.ours.generate \
    --model meta-llama/Llama-2-7b-hf \
    --start_sentence 1 \
    --num_sentences 1 \
    --num_sequences 20 \
    --max_tokens 512 \
    --task_prompt_idx 1 \
    --task bookMIA \
    --data_split val; 

 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m experiments.ours.generate \
    --model meta-llama/Llama-2-70b-hf \
    --start_sentence 1 \
    --num_sentences 1 \
    --num_sequences 20 \
    --max_tokens 512 \
    --task_prompt_idx 1 \
    --task bookMIA \
    --data_split train;   
"""