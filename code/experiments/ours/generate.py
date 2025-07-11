import argparse
from code.user_secrets import CACHE_PATH
import os
# Set up environment variables
os.environ["HF_HOME"] = CACHE_PATH
os.environ["HF_DATASETS_PATH"] = CACHE_PATH

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
from code.experiments.utils import zigzag_append, chunk_list, remove_last_n_words, bool_to_first_upper

def main(args):
    # Seed model and get the model string
    random.seed(args.seed)
    model_str = args.model.split("/")[-1] # Splits to get actual model name
    if model_str not in task_prompts_dict_book[args.task]:
        print("Valid model not passed in. Try again")

    # Set up the task prompts
    args.task_prompt_idx = sorted(args.task_prompt_idx)
    cur_task_prompts = []
    for cur_prompt_idx in args.task_prompt_idx:
        cur_task_prompts.append(task_prompts_dict_book[args.task][model_str][cur_prompt_idx])
    chunk_size = len(cur_task_prompts) # TODO different chunk sizes

    # Load the data
    data_path = f"data/{args.task}/split-random-overall/{args.data_split}.jsonl"
    final_subset = pd.read_json(data_path, lines=True)
    print(f"Length: {len(final_subset)}")
    if args.key_name is not None:
        final_subset["snippet"] = final_subset[args.key_name]
    if args.remove_bad_first: # Remove ill-formatted first sentence
        final_subset["snippet"] = final_subset["snippet"].apply(remove_first_sentence_if_needed)

    save_folder = os.path.join(f"outputs/ours/{args.task}/generations/{args.data_split}")
    os.makedirs(save_folder, exist_ok=True)

    if args.prompt_with_words_not_sent:
        args.start_sentence, args.num_sentences = -1, -1
    else:
        args.start_word, args.num_words_from_end, args.num_proportion_from_end = -1, -1, -1

    if args.temperature == 0: # Reduce num_sequences if using greedy decoding
        print("GREEDY decoding - setting num_sequences to 1")
        args.num_sequences = 1
    
    if args.openai: # OpenAI models
        passages = final_subset.snippet.tolist()
        if not args.prompt_with_words_not_sent: # Use sentences
            prompt_outputs = [extract_chunk_sentence(text, args.start_sentence, args.num_sentences) for text in passages]        
            prompt_texts, rest_of_texts = zip(*prompt_outputs)
            prompt_texts, rest_of_texts = list(prompt_texts), list(rest_of_texts)
        else: # Use words
            import tiktoken
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            if args.num_words_from_end >= 1:
                if args.num_proportion_from_end != 0:
                    print("Overriding num_proportion_from_end")
                    args.num_proportion_from_end = -1

                data = [remove_last_n_words(encoding, text, args.num_words_from_end, openai=True) for text in passages]
                prompt_texts, rest_of_texts, token_lengths = map(list, zip(*data))
            else:
                assert 0 < args.num_proportion_from_end < 1, "No remove tokens set"
                # Get length of words
                text_lengths = [len(text.split()) for text in passages]
                remove_lengths = [max(1, min(int(l * args.num_proportion_from_end), l-1)) for l in text_lengths]
                data = [remove_last_n_words(encoding, text, remove_length, openai=True) for text, remove_length in zip(passages, remove_lengths)]
                prompt_texts, rest_of_texts, token_lengths = map(list, zip(*data))

        assert None not in prompt_texts
        unmerged_prompts = []
        for cur_task_prompt in cur_task_prompts:
            unmerged_prompts.append(make_prompts(
                prompt_texts, 
                cur_task_prompt["task_prompt"], 
                cur_task_prompt["task_preprompt"],
                cur_task_prompt["task_postprompt"],
            ))
        prompts = zigzag_append(unmerged_prompts) # Make indices match up

        requests = []
        for i, prompt in enumerate(prompts):
            if args.max_length_to_sequence_length:
                cur_max_tokens = token_lengths[i // chunk_size]
            else:
                assert args.max_tokens >= 1
                cur_max_tokens = args.max_tokens
            request_id = i
            cur_request = {
                "model": args.model,
                "max_tokens": cur_max_tokens,
                "temperature": args.temperature,
                "seed": args.seed,
                "top_p": args.top_p,
                "n": args.num_sequences, # Hardcode this
                "metadata": {"request_id": request_id},
            }
            if "instruct" in args.model or "davinci" in args.model:
                cur_request = cur_request | {
                    "prompt": prompt,
                    # "logprobs": 10
                }
            else:
                cur_request = cur_request | {
                    "messages": [{"role": "user", "content": prompt}],
                    # "logprobs": True, "top_logprobs": 10
                }
            requests.append(cur_request)
        if "instruct" not in args.model:
            print(f"Example prompt\n\n{requests[0]['messages'][0]['content']}")
        else:
            print(f"Example prompt\n\n{requests[0]['prompt']}")

        max_requests_per_minute = requests_limits_dict[args.model]["max_requests_per_minute"]
        max_tokens_per_minute = requests_limits_dict[args.model]["max_tokens_per_minute"]
        request_url = requests_url_dict[args.model]
        print(f"Using rate limits\n------\nMax requests per minute: {max_requests_per_minute}\nMax tokens per minute: {max_tokens_per_minute}")

        full_generations = asyncio.run(openai_parallel_generate(
                requests, 
                args, 
                max_requests_per_minute=max_requests_per_minute, 
                max_tokens_per_minute=max_tokens_per_minute,
                request_url=request_url
                ))
            
        indexed_results = {}
        unknown_id_generations = [] # Special case where the request_id is not returned
        for result in full_generations:
            try:
                request_id = result[2]["request_id"] # Extract request_id from metadata
                indexed_results[request_id] = result[1]  # API response is the second element
            except:
                unknown_id_generations.append(result[1])

        if len(unknown_id_generations) != 0:
            len_unknown = len(unknown_id_generations)
            print("Error on ids of ", len_unknown)
            for i in range(len(requests)):
                if i not in indexed_results:
                    indexed_results[i] = unknown_id_generations.pop()

        all_text_outputs = []
        for i in range(len(full_generations)):
            cur_results = indexed_results[i]
            if "instruct" in args.model or "davinci" in args.model:
                all_text_outputs.append([cur["text"] for cur in cur_results["choices"]])
            else:
                all_text_outputs.append([cur["message"]["content"] for cur in cur_results["choices"]])

        # TODO gpt-3.5-turbo-instruct is it different?
        final_subset["prompt"] = chunk_list(prompts, chunk_size)
        final_subset["generation"] = chunk_list(all_text_outputs, chunk_size)
        final_subset["model"] = [model_str] * len(final_subset)
        final_subset["snippet_no_prompt"] = rest_of_texts

    else: # vLLM models
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

            tokenizer = generator.llm.get_tokenizer()
            token_lengths = [len(tokenizer.encode(text, add_special_tokens=False)) for text in rest_of_texts]
        else:
            passages = final_subset.snippet.tolist()

            if not args.prompt_with_words_not_sent:
                prompt_outputs = [extract_chunk_sentence(text, args.start_sentence, args.num_sentences) for text in passages]
                prompt_texts, rest_of_texts = zip(*prompt_outputs)
                prompt_texts= list(prompt_texts)
                rest_of_texts = list(rest_of_texts)
            else:
                if args.num_words_from_end >= 1:
                    if args.num_proportion_from_end != 0:
                        print("Overriding num_proportion_from_end")
                        args.num_proportion_from_end = -1

                    data = [remove_last_n_words(generator.llm.get_tokenizer(), text, args.num_words_from_end, openai=False) for text in passages]
                    prompt_texts, rest_of_texts, token_lengths = map(list, zip(*data))                    
                else:
                    assert 0 < args.num_proportion_from_end < 1, "No remove tokens set"
                    # Get length of words
                    text_lengths = [len(text.split()) for text in passages]
                    remove_lengths = [max(1, min(int(l * args.num_proportion_from_end), l-1)) for l in text_lengths]
                    data = [remove_last_n_words(generator.llm.get_tokenizer(), text, remove_length, openai=False) for text, remove_length in zip(passages, remove_lengths)]
                    prompt_texts, rest_of_texts, token_lengths = map(list, zip(*data))

        assert None not in prompt_texts
        unmerged_prompts = []
        for cur_task_prompt in cur_task_prompts:
            unmerged_prompts.append(make_prompts(
                prompt_texts, 
                cur_task_prompt["task_prompt"], 
                cur_task_prompt["task_preprompt"],
                cur_task_prompt["task_postprompt"],
                model_name=model_str,
                prompt_key="lightest"
            ))
        prompts = zigzag_append(unmerged_prompts) # Make indices match up

        if args.max_length_to_sequence_length:
            cur_max_tokens = [element for element in token_lengths for _ in range(chunk_size)]
        else:
            assert args.max_tokens >= 1
            cur_max_tokens = args.max_tokens

        # Generation
        final_prompts, all_text_outputs, all_prompt_logprobs, all_output_logprobs = generator.generate_vllm(
            prompts=prompts,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=cur_max_tokens,
            min_tokens=args.min_tokens,
            max_length=args.max_length,
            n=args.num_sequences
        )

        final_subset["prompt"] = chunk_list(final_prompts, chunk_size)
        final_subset["generation"] = chunk_list(all_text_outputs, chunk_size)
        final_subset["model"] = [model_str] * len(final_subset)
        final_subset["snippet_no_prompt"] = rest_of_texts

        # final_subset["logprobs"] = all_output_logprobs
        # final_subset["logprobs_prompt"] = all_prompt_logprobs

    # Convert current datetime to string in 'YYYY-MM-DD HH:MM:SS' format
    date_str = datetime.now().strftime("%Y-%m-%d-%H:%M:%S").strip()
    minTokStr = "minTok" + str(args.min_tokens) + "_"
    
    # Save DataFrame to CSV with detailed info in the filename
    file_name = f"""{model_str}_maxTok{args.max_tokens}_{minTokStr}numSeq{args.num_sequences}\
_topP{args.top_p}_temp{args.temperature}_numSent{args.num_sentences}_startSent{args.start_sentence}\
_numWordFromEnd{args.num_words_from_end}-{args.num_proportion_from_end}_maxLenSeq{bool_to_first_upper(args.max_length_to_sequence_length)}\
_useSent{bool_to_first_upper(not args.prompt_with_words_not_sent)}-rmvBad{bool_to_first_upper(args.remove_bad_first)}\
_promptIdx{'-'.join(map(str, args.task_prompt_idx))}_len{len(final_subset)}_{date_str}.jsonl"""
    
    file_path = os.path.join(save_folder, file_name)
    columns = [col for col in final_subset.columns if col != 'snippet'] + ['snippet']
    final_subset = final_subset[columns]
    final_subset.to_json(file_path, index=False, lines=True, orient='records')
    print(f"Saved to {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text")

    # Generation parameters
    parser.add_argument('--max_tokens', type=int, default=512, help='Maximum number of tokens to generation.')
    parser.add_argument('--min_tokens', type=int, default=0, help='Maximum number of tokens to generation.')
    parser.add_argument('--max_length', type=int, default=2048, help='Maximum length')
    parser.add_argument('--num_sequences', type=int, default=1, help='Number of sequences to generation.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Model sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p sampling value.')
    parser.add_argument('--max_length_to_sequence_length', action="store_true", help="Make the generation length equal to the length of the rest of the sequence")

    # What part of the samples to prompt with
    parser.add_argument('--num_sentences', type=int, default=1, help='Number of sentences to use from the snippet.')
    parser.add_argument('--start_sentence', type=int, default=1, help='Starting sentence to use from the snippet.')

    parser.add_argument("--prompt_with_words_not_sent", action="store_true", help="whether or not to use words vs sentences")
    parser.add_argument('--num_words_from_end', type=int, default=-1, help='Number of words to use from the snippet.')
    parser.add_argument('--num_proportion_from_end', type=float, default=-1, help='Number of words to use from the snippet.')
    parser.add_argument('--task_prompt_idx', type=int, nargs='+', default=[1], help='Index or list of indices of the task prompts to use.')
    # parser.add_argument('--start_word', type=int, default=1, help='Starting word to use from the snippet.')

    # Model details
    parser.add_argument('--model', type=str, default="davinci-002", help='Model to use for text generation.')
    parser.add_argument('--tokenizer', type=str, default=None, help='Pass in tokenizer manually. Optional.')
    parser.add_argument('--hf_token', type=str, default=None, help='Pass in huggingface token.')
    parser.add_argument("--openai", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    # Data details
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
    """