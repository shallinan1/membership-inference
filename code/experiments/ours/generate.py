import argparse
from code.user_secrets import CACHE_PATH
import os
# Set up environment variables
os.environ["HF_HOME"] = CACHE_PATH
os.environ["HF_DATASETS_PATH"] = CACHE_PATH

from code.helper.generation.openai_generate import get_gpt_output
from nltk import sent_tokenize
import pandas as pd
from IPython import embed
from tqdm import tqdm
from code.helper.generation.vllm_generate import ModelGenerator 
from code.helper.generation.generate_utils import task_prompts_dict_book, make_prompts
import random
from datetime import datetime
from code.experiments.ours.utils import extract_chunk_sentence
import asyncio
from code.helper.generation.openai_parallel_generate import openai_parallel_generate, requests_limits_dict, requests_url_dict
from code.utils import remove_first_sentence_if_needed

# Function to define the main process
def main(args):
    random.seed(0)

    model_str = args.model.split("/")[-1] # Splits to get actual model name
    if model_str not in task_prompts_dict_book[args.task]:
        print("Valid model not passed in. Try again")
    cur_task_prompts = task_prompts_dict_book[args.task][model_str][args.task_prompt_idx]

    data_path = f"data/{args.task}/split-random-overall/{args.data_split}.jsonl"
    final_subset = pd.read_json(data_path, lines=True)
    print(f"Length: {len(final_subset)}")
    if args.key_name is not None:
        final_subset["snippet"] = final_subset[args.key_name]

    save_folder = os.path.join(f"outputs/ours/{args.task}/generations/{args.data_split}")
    os.makedirs(save_folder, exist_ok=True)

    if args.prompt_with_words_not_sent:
        args.start_sent, args.num_sents = -1, -1
        prompt_with_sent_str = "T"
    else:
        args.start_word, args.num_words = -1, -1
        prompt_with_sent_str = "F"

    # Reduce num_sequences if using greedy decoding
    if args.temperature == 0:
        print("GREEDY decoding - setting num_sequences to 1")
        args.num_sequences = 1

    if args.openai:        
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
        )

        requests = []
        for i, prompt in enumerate(prompts):
            request_id = i
            requests.append({
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "seed": args.seed,
                "top_p": args.top_p,
                "n": args.num_sequences, # Hardcode this
#                "top_logprobs": 10,
                "metadata": {"request_id": request_id},
            })
        
        max_requests_per_minute = requests_limits_dict[args.model]["max_requests_per_minute"]
        max_tokens_per_minute = requests_limits_dict[args.model]["max_tokens_per_minute"]
        request_url = requests_url_dict[args.model]

        print(prompts[0])
        # embed()
        full_generations = asyncio.run(openai_parallel_generate(
                requests, 
                args, 
                max_requests_per_minute=max_requests_per_minute, 
                max_tokens_per_minute=max_tokens_per_minute,
                request_url=request_url
                ))
            
        indexed_results = {}
        for result in full_generations:
            request_id = result[2]["request_id"] # Extract request_id from metadata
            indexed_results[request_id] = result[1]  # API response is the second element

        all_text_outputs = []
        for i in range(len(full_generations)):
            cur_results = indexed_results[i]
            all_text_outputs.append([cur["message"]["content"] for cur in cur_results["choices"]])
        
        # TODO gpt-3.5-turbo-instruct is it different?

        final_subset["prompt"] = prompts
        final_subset["generation"] = all_text_outputs
        final_subset["model"] = [model_str] * len(final_subset)
        final_subset["snippet_no_prompt"] = rest_of_texts
    else:
        generator = ModelGenerator(
            model=args.model,
            tokenizer=args.model if not args.tokenizer else args.tokenizer,
            seed=args.seed,
            hf_token=args.hf_token,
            cache_dir=CACHE_PATH,
        )

        if args.task == "tulu_v1": # Special processing for tulu dataset
            # Num turns = 1 by default for now
            prompt_texts = final_subset.messages.apply(lambda x: x[0]['content']).tolist()
            rest_of_texts = final_subset.messages.apply(lambda x: x[1]['content']).tolist()
            # TODO some assertion to make sure these roles are correct

            final_subset["snippet"] = [p + "\n" + r for p, r in zip(prompt_texts, rest_of_texts)]
            args.start_sentence, args.num_sentence = -1, -1
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

        # Generation
        final_prompts, all_text_outputs, all_prompt_logprobs, all_output_logprobs = generator.generate_vllm(
            prompts=prompts,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_tokens,
            min_tokens=args.min_tokens,
            max_length=args.max_length,
            n=args.num_sequences
        )
        final_subset["prompt"] = final_prompts
        final_subset["generation"] = all_text_outputs
        final_subset["model"] = [model_str] * len(final_subset)
        final_subset["snippet_no_prompt"] = rest_of_texts

        # final_subset["logprobs"] = all_output_logprobs
        # final_subset["logprobs_prompt"] = all_prompt_logprobs

    # Convert current datetime to string in 'YYYY-MM-DD HH:MM:SS' format
    date_str = datetime.now().strftime("%Y-%m-%d-%H:%M:%S").strip()
    minTokStr = "minTok" + str(args.min_tokens) + "_"
    
    # Save DataFrame to CSV with detailed info in the filename
    file_name = f"{model_str}_maxTok{args.max_tokens}_{minTokStr}numSeq{args.num_sequences}_topP{args.top_p}_temp{args.temperature}_numSent{args.num_sentences}_startSent{args.start_sentence}_numWord{args.num_words}_startWord{args.num_words}_useSent{prompt_with_sent_str}_promptIdx{args.task_prompt_idx}_len{len(final_subset)}_{date_str}.jsonl"
    file_path = os.path.join(save_folder, file_name)
    columns = [col for col in final_subset.columns if col != 'snippet'] + ['snippet']
    final_subset = final_subset[columns]
    final_subset.to_json(file_path, index=False, lines=True, orient='records')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text")

    parser.add_argument('--max_tokens', type=int, default=512, help='Maximum number of tokens to generation.')
    parser.add_argument('--min_tokens', type=int, default=0, help='Maximum number of tokens to generation.')
    parser.add_argument('--max_length', type=int, default=2048, help='Maximum length')
    parser.add_argument('--num_sequences', type=int, default=1, help='Number of sequences to generation.')
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
    
    parser.add_argument("--remove_bad_first", action="store_true")
    parser.add_argument("--keep_n_sentences", type=int, default=-1)

    parser.add_argument("--key_name", type=str, default=None, help="text key name")


    main(parser.parse_args())


    """
    CUDA_VISIBLE_DEVICES=0 python3 -m code.experiments.ours.generate \
        --model openai-community/gpt2-large \
        --start_sentence 1 \
        --num_sentences 3 \
        --num_sequences 20 \
        --max_tokens 512 \
        --task_prompt_idx 0;      

    CUDA_VISIBLE_DEVICES=0 python3 -m code.experiments.ours.generate \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --start_sentence 1 \
        --num_sentences 5 \
        --num_sequences 20 \
        --max_tokens 512 \
        --min_tokens 10 \
        --task_prompt_idx 5;    

    CUDA_VISIBLE_DEVICES=0,1 python3 -m code.experiments.ours.generate \
        --model meta-llama/Llama-2-70b-chat-hf \
        --start_sentence 1 \
        --num_sentences 5 \
        --num_sequences 20 \
        --max_tokens 512 \
        --task_prompt_idx 5; 
    """