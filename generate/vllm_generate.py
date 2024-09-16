from huggingface_hub import login
import argparse
from transformers import AutoTokenizer, set_seed
import torch
import os
import json
from IPython import embed
from generate.generate_utils import *
from vllm import LLM, SamplingParams
from typing import List


def load_prompts(input_file, model_name="default", prompt_col='prompt', prompt_key=None, task=None):
    print("\nLoading prompts\n")
    lower_model_name = model_name.lower()

    task_prompt = 'Please write a few paragraphs for a novel starting with the following prompt: '
    task_postprompt = ''
    if "new_book_LM" in task: # Training data for new book task
        task_prompt = 'Please write a few paragraphs for a novel starting with the following prompt: '
    elif "letter" in task:
        task_prompt = ""
    elif "fake_news" in task:
        if "chat" not in lower_model_name and "it" not in lower_model_name and "inst" not in lower_model_name and "tulu" not in lower_model_name:
            task_prompt = ""
        else:
            task_prompt = "Please write a news article based on the given headline:\n\n"
    elif "poem" in task:
        if "chat" not in lower_model_name and "it" not in lower_model_name and "inst" not in lower_model_name and "tulu" not in lower_model_name:
            task_prompt = ""
            task_postprompt = "\n"
        else:
            task_prompt = "Please write a poem starting with the following line: "
    elif "speech" in task:
        if "chat" not in lower_model_name and "it" not in lower_model_name and "inst" not in lower_model_name and "tulu" not in lower_model_name:
            task_prompt = ""
            task_postprompt = "\n"
        else:
            task_prompt = "Please write a speech starting with the following sentence: "
    elif "ml_papers" in task:
        if "chat" not in lower_model_name and "it" not in lower_model_name and "inst" not in lower_model_name and "tulu" not in lower_model_name:
            task_prompt = ""
            task_postprompt = "\n"
        else:
            task_prompt = "Given the following summary, please generate a full-length abstract for a scientific paper.\n\nSummary: "
            task_postprompt = "Abstract: "
    elif "theorem" in task:
        if "chat" not in lower_model_name and "it" not in lower_model_name and "inst" not in lower_model_name and "tulu" not in lower_model_name:
            task_prompt = ""
        else:
            task_prompt = "Please provide a proof for the following theorem:\n\n"
    corpus = json.load(open(input_file, 'r'))
    
    if "tulu" in lower_model_name: # Tulu Models, ie https://huggingface.co/allenai/tulu-2-dpo-70b
        task_postprompt = task_postprompt.lstrip()
        prompts = ["<|user|>\n" + task_prompt + ' '.join(p[prompt_col].strip().split()) + f"\n<|assistant|>\n{task_postprompt}" for p in corpus]
    elif "olmo" in lower_model_name: # Olmo models, ie https://huggingface.co/allenai/OLMo-7B and for internal, instruction tuned variant
        if "internal" in lower_model_name or "instruct" in lower_model_name:
            task_postprompt = task_postprompt.lstrip()
            prompts = ["<|user|>\n" + task_prompt + ' '.join(p[prompt_col].strip().split()) + f"\n<|assistant|>\n{task_postprompt}" for p in corpus]
        else:
            prompts = [f"{task_prompt}{' '.join(p[prompt_col].strip().split())}{task_postprompt}" for p in corpus]
    elif "llama2" in lower_model_name: # LLama2 models, ie, https://huggingface.co/docs/transformers/model_doc/llama2
        if "chat" in lower_model_name:
            cur_instructions = llama2_chat_prompt_guide[prompt_key] if prompt_key in llama2_chat_prompt_guide else "full"
            preprompt = LLAMA2_CHAT_PREPROMPT.substitute(instructions=cur_instructions)
            task_postprompt = task_postprompt.lstrip()
            prompts = [f"{preprompt}{task_prompt}{' '.join(p[prompt_col].strip().split())}{LLAMA2_CHAT_POSTPROMPT}{task_postprompt}" for p in corpus]
        else:
            prompts = [f"{task_prompt}{' '.join(p[prompt_col].strip().split())}{task_postprompt}" for p in corpus]
    elif "llama3" in lower_model_name: # LLama3 models, ie, https://huggingface.co/docs/transformers/main/en/model_doc/llama3
        if "inst" in lower_model_name:
            cur_instructions = llama3_chat_prompt_guide[prompt_key] if prompt_key in llama3_chat_prompt_guide else "lighest"
            preprompt = LLAMA3_INSTRUCT_PREPROMPT.substitute(instructions=cur_instructions)
            task_postprompt = task_postprompt.lstrip()
            prompts = [f"{preprompt}{task_prompt}{' '.join(p[prompt_col].strip().split())}{LLAMA3_INSTRUCT_POSTPROMPT}{task_postprompt}" for p in corpus]
        else:
            print("NOT IMPLEMENTED YET")
            import sys; sys.exit()
    elif "mistral" in lower_model_name or "mixtral" in lower_model_name:
        if "inst" in lower_model_name:
            task_postprompt = task_postprompt.lstrip()
            prompts = [f"[INST] {task_prompt}{' '.join(p[prompt_col].strip().split())} [/INST]{task_postprompt}" for p in corpus]
        else:
            print("NOT IMPLEMENTED YET")
            import sys; sys.exit()
    elif "gemma" in lower_model_name: # Gemma models, ie https://huggingface.co/google/gemma-2b-it
        if "it" in lower_model_name:
            task_postprompt = "\n"+task_postprompt.lstrip()
            prompts = [f"<start_of_turn>user\n{task_prompt}{' '.join(p[prompt_col].strip().split())}<end_of_turn>\n<start_of_turn>model{task_postprompt}" for p in corpus]
        else:
            prompts = [f"{task_prompt}{' '.join(p[prompt_col].strip().split())}{task_postprompt}" for p in corpus]
    else: # Default branch (no instructions) for all other models
            prompts = [f"{task_prompt}{' '.join(p[prompt_col].strip().split())}{task_postprompt}" for p in corpus]
    return prompts

class ModelGenerator():
    def __init__(self, model: str, seed: int = 0, hf_token: str = None, vllm: bool = True):
        print("Initializing vLLM model")
        
        set_seed(seed)
        
        if vllm:
            self.generate_function = self.generate_vllm
        else:
            print("Non-vllm inference not currently spuported")
        
        if hf_token:
            login(hf_token)

        print(f"device count {torch.cuda.device_count()}")

        mmodel_save_name = model
        if "/" in model:
            self.model_save_name = model.split("/")[-1]

        self.llm = LLM(model=model, tensor_parallel_size=torch.cuda.device_count(), download_dir = args.cache_dir) # Load the model
        
        if args.data_save_name is None:
            args.data_save_name = args.save_dir.split("/")[-1]
        prompt_key_name = f"_prompt-{args.prompt_key}" if args.prompt_key else ""

        # Make the tokenizer
        tokenizer_str = tokenizer_path if tokenizer_path else model  # Pass in a cached tokenizer (speed up tokenizer)
        add_bos_token = True # Adding the bos token explicitly (Mistral, mixtral for some reason have it as false, while other models default to True)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_str, padding_side="left", trust_remote_code=True, add_bos_token=add_bos_token)


    def make_prompt_format(self):
        prompts = load_prompts(args.input_file, args.model_save_name, prompt_key=args.prompt_key, task = args.data_save_name)
        print(f"\nExample prompt (index 0): {prompts[0]}\n")

    def generate_vllm(self, prompts: List[str], temperature: float = 1.0, top_p: float=0.95, sample: bool = True, max_new_tokens: int = 256, max_length: int = 2048, extra_stop_tokens: List[int] = None):
        extra_stop_tokens = [int(a) for a in extra_stop_tokens] if extra_stop_tokens else []

        outputs, final_prompts = [], []
        print(len(prompts))

        input_ids_list = self.tokenizer(prompts, truncation=True, max_length=max_length).input_ids # Do NOT set padding
        filtered_input_ids = []
        for prompt, input_ids in zip(prompts, input_ids_list): # Check that we didn't include any pad tokens and remove long sequences
            assert self.tokenizer.pad_token_id not in input_ids
            if len(input_ids) + max_new_tokens <= max_length:
                filtered_input_ids.append(input_ids)
                final_prompts.append(prompt)

        # Set up the sampling parameters
        stop_token_ids = list(set(extra_stop_tokens + [self.llm.get_tokenizer().eos_token_id]))
        if not sample:
            temperature = 0 # vllm will always use sampling unless temperature = 0
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids) # Set seed gives weird output for base model

        # Generation
        output = self.llm.generate(
            sampling_params=sampling_params, 
            prompt_token_ids=filtered_input_ids)
        
        for cp, o in zip(prompts, output): # Change this if we want to do multiple sampled sequences per input
            output_str = self.tokenizer.decode(o.outputs[0].token_ids, skip_special_tokens=True).strip()
            outputs.append(output_str)
                        
        return final_prompts, outputs
    


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Script to generate with vLLM')

    # Add arguments
    parser.add_argument('--vllm', action='store_true', help='Use VLM model if specified')
    parser.add_argument('--model', type=str, help='Model name for VLM')
    parser.add_argument('--model_save_name', type=str, help='Model saving name for final output')
    parser.add_argument('--data_save_name', type=str, default = None, help='Data saving name for final output')
    parser.add_argument('--tok_path', type=str, default=None, help='Optional separate tokenizer path')
    parser.add_argument('--max_length', type=int, default=2048, help='Maximum length for tokenization')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='Maximum new tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p for nucleus sampling')
    parser.add_argument('--sample', action='store_true', help='Enable sampling mode')
    parser.add_argument('--extra_stop_tokens', type=int, nargs='+', help='Extra stop tokens for generation')
    parser.add_argument('--hf_token', type=str, default=None, help="Optional huggingface token for gated models")
    parser.add_argument('--save_dir', type=str, default="data/", help="Save directory")
    parser.add_argument('--input_file', type=str, help="Path to inputs, a .json file")
    parser.add_argument('--seed', type=int, help="seed", default=0)
    parser.add_argument('--cache_dir', type=str, default=None, help="Directory to save models")
    parser.add_argument('--prompt_key', type=str, default=None, help="Key for specific prompt")
    parser.add_argument('--min_length', type=int, default=128, help="Key for specific prompt")
    parser.add_argument('--no_filter', action="store_true", help="Key for specific prompt")
    parser.add_argument('--save_in_loop', action="store_true", help="Whether or not to save in the loop")

    # Parse arguments
    main(parser.parse_args())

    """
    Command to run 

    # Generate for Llama2-7b-chat-hf
    CUDA_VISIBLE_DEVICES=0 python3 -m data.generate_hf \
        --vllm \
        --model meta-llama/Llama-2-7b-chat-hf \
        --model_save_name llama2-7b-chat \
        --hf_token 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 \
        --sample \
        --input_file data/new_book.json \
        --max_new_tokens 164 \
        --temperature 1.0 \
        --top_p 0.9 \
        --cache_dir ../cache/
    """