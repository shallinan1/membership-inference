from string import Template
from typing import List, Optional, Union

# LLama 2 Chat prompts
LLAMA2_CHAT_SYS_INSTRUCTIONS_FULL="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
LLAMA2_CHAT_SYS_INSTRUCTIONS_LIGHT="""You are a helpful, respectful and honest assistant."""
LLAMA2_CHAT_SYS_INSTRUCTIONS_LIGHTEST="""You are a helpful assistant."""
llama2_chat_prompt_guide = {
    'full': LLAMA2_CHAT_SYS_INSTRUCTIONS_FULL,
    'light': LLAMA2_CHAT_SYS_INSTRUCTIONS_LIGHT,
    'lightest': LLAMA2_CHAT_SYS_INSTRUCTIONS_LIGHTEST
    }
LLAMA2_CHAT_PREPROMPT = Template("""[INST] <<SYS>>\n$instructions\n<</SYS>>

""")
LLAMA2_CHAT_POSTPROMPT = " [/INST]"

# LLama 3 Instruct prompts
LLAMA3_CHAT_SYS_INSTRUCTIONS_LIGHTEST = LLAMA2_CHAT_SYS_INSTRUCTIONS_LIGHTEST
llama3_chat_prompt_guide = {
    'lightest': LLAMA3_CHAT_SYS_INSTRUCTIONS_LIGHTEST
}
# LLama3 prompts
LLAMA3_INSTRUCT_PREPROMPT = Template("""<|start_header_id|>system<|end_header_id|>

$instructions<|eot_id|><|start_header_id|>user<|end_header_id|>

""")
LLAMA3_INSTRUCT_POSTPROMPT = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

# LLama3.1 prompts - Same as LLama3
LLAMA31_INSTRUCT_PREPROMPT = Template("""<|start_header_id|>system<|end_header_id|>

$instructions<|eot_id|><|start_header_id|>user<|end_header_id|>

""")
LLAMA31_INSTRUCT_POSTPROMPT = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

def make_prompts(
    prompts: Union[List[str], str], 
    task_prompt: str, 
    task_preprompt: str = "",
    task_postprompt: str = "",
    model_name: str = "default", 
    prompt_key: Optional[str] = None, 
) -> List[str]:
    """
    Constructs and formats a list of prompts based on the specified model type and other parameters.

    Args:
        prompts (List[str]): A list of prompt strings to be formatted.
        task_prompt (str): The task-specific prompt to be prepended to each prompt.
        model_name (str, optional): The name of the model to determine formatting style. Default is "default".
        prompt_key (Optional[str], optional): A key to look up specific prompt instructions for models that support it. Default is None.

    Returns:
        List[str]: A list of formatted prompts ready for use with the specified model.
    """
    if isinstance(prompts, str):
        prompts = [prompts]
    lower_model_name = model_name.lower()

    preproc = False # TODO make this an arg
    if preproc:
        prompts = [' '.join(p.strip().split()) for p in prompts]

    if "tulu" in lower_model_name: # Tulu Models, ie tulu_v1, tulu_v2 https://huggingface.co/allenai/tulu-2-dpo-70b
        task_postprompt = task_postprompt.lstrip()
        prompts = ["<|user|>\n" + task_prompt + p + f"\n<|assistant|>\n{task_postprompt}" for p in prompts]
    
    elif "olmo" in lower_model_name: # Olmo models, ie https://huggingface.co/allenai/OLMo-7B and for internal, instruction tuned variant
        if "sft" in lower_model_name or "instruct" in lower_model_name:
            task_postprompt = task_postprompt.lstrip()
            prompts = ["<|user|>\n" + task_prompt + p + f"\n<|assistant|>\n{task_postprompt}" for p in prompts]
        else:
            prompts = [f"{task_prompt}{p}{task_postprompt}" for p in prompts]
    
    elif "llama-2" in lower_model_name: # LLama2 models, ie, https://huggingface.co/docs/transformers/model_doc/llama2
        if "chat" in lower_model_name:
            cur_instructions = llama2_chat_prompt_guide[prompt_key] if prompt_key in llama2_chat_prompt_guide else llama2_chat_prompt_guide["lightest"]
            preprompt = LLAMA2_CHAT_PREPROMPT.substitute(instructions=cur_instructions)
            task_postprompt = task_postprompt.lstrip()
            prompts = [f"{preprompt}{task_prompt}{' '.join(p.strip().split())}{LLAMA2_CHAT_POSTPROMPT}{task_postprompt}" for p in prompts]
        else:
            prompts = [f"{task_prompt}{' '.join(p.strip().split())}{task_postprompt}" for p in prompts]
    
    elif "llama-3.2" in lower_model_name: # LLama3 models, ie, https://huggingface.co/docs/transformers/main/en/model_doc/llama3
            print("NOT IMPLEMENTED YET")
            import sys; sys.exit()

    elif "llama-3.1" in lower_model_name: # LLama3 models, ie, https://huggingface.co/docs/transformers/main/en/model_doc/llama3
        if "inst" in lower_model_name:
            cur_instructions = llama3_chat_prompt_guide[prompt_key] if prompt_key in llama3_chat_prompt_guide else llama3_chat_prompt_guide["lightest"]
            preprompt = LLAMA31_INSTRUCT_PREPROMPT.substitute(instructions=cur_instructions)
            task_postprompt = task_postprompt.lstrip()
            prompts = [f"{preprompt}{task_prompt}{' '.join(p.strip().split())}{LLAMA31_INSTRUCT_POSTPROMPT}{task_postprompt}" for p in prompts]
        else:
            prompts = [f"{task_prompt}{task_preprompt}{p}{task_postprompt}" for p in prompts]
    
    elif "llama-3-" in lower_model_name:
        if "inst" in lower_model_name:
            cur_instructions = llama3_chat_prompt_guide[prompt_key] if prompt_key in llama3_chat_prompt_guide else llama3_chat_prompt_guide["lightest"]
            preprompt = LLAMA3_INSTRUCT_PREPROMPT.substitute(instructions=cur_instructions)
            task_postprompt = task_postprompt.lstrip()
            prompts = [f"{preprompt}{task_prompt}{' '.join(p.strip().split())}{LLAMA3_INSTRUCT_POSTPROMPT}{task_postprompt}" for p in prompts]
        else:
            prompts = [f"{task_prompt}{task_preprompt}{p}{task_postprompt}" for p in prompts]
    
    elif "mistral" in lower_model_name or "mixtral" in lower_model_name:
        if "inst" in lower_model_name:
            task_postprompt = task_postprompt.lstrip()
            prompts = [f"[INST] {task_prompt}{' '.join(p.strip().split())} [/INST]{task_postprompt}" for p in prompts]
        else:
            print("NOT IMPLEMENTED YET")
            import sys; sys.exit()
    
    elif "gemma" in lower_model_name: # Gemma models, ie https://huggingface.co/google/gemma-2b-it
        if "it" in lower_model_name:
            task_postprompt = "\n"+task_postprompt.lstrip()
            prompts = [f"<start_of_turn>user\n{task_prompt}{' '.join(p.strip().split())}<end_of_turn>\n<start_of_turn>model{task_postprompt}" for p in prompts]
        else:
            prompts = [f"{task_prompt}{' '.join(p.strip().split())}{task_postprompt}" for p in prompts]
    
    else: # Default branch (no instructions) for all other models such as openai model
            prompts = [f"{task_prompt}{task_preprompt}{p}{task_postprompt}" for p in prompts]
    return prompts

# Task prompts
task_prompts_dict_book = {
    "articles":
            {
            "instruct-autoregressive": # Prompts for instruction-tuned examples
                [{
                    "task_prompt": "",
                    "task_postprompt": "",
                    "task_preprompt": ""
                },
                {
                    "task_prompt": "Continue the article as accurately as possible, without any extra text.\n\n",
                    "task_postprompt": "",
                    "task_preprompt": ""
                },
                {
                    "task_prompt": "Continue the generation as closely to verbatim as possible.\n\n",
                    "task_postprompt": "",
                    "task_preprompt": ""
                },
                {
                    "task_prompt": """You will be shown a series of news articles. After these examples, \
you will receive a prefix from another news article and be asked to complete it based on the \
text of a well-known article. Provide only the continuation for the last given prefix without any \
extra commentary, formatting, or additional text.\n\nComplete the prefix: """,
                    "task_postprompt": "",
                    "task_preprompt": ""
                },
                {
                    "task_prompt": """You will be shown a series of news articles. After these examples, \
you will receive a prefix from another news article and be asked to complete it. \
Provide only the continuation for the last given prefix without any \
extra commentary, formatting, or additional text.\n\nComplete the prefix: """,
                    "task_postprompt": "",
                    "task_preprompt": ""
                }],
        }
    ,
    "dolma_v17": # Same as pile
        {"noninstruct-autoregressive": # GPT3 style - Can be abstracted to prompts for non-instruction tuned models
            [
                {
                    "task_prompt": "Continue the text: ",
                    "task_postprompt": "",
                    "task_preprompt": ""
                },
                {
                    "task_prompt": "", # Blank string ,
                    "task_postprompt": "",
                    "task_preprompt": ""
                },
            ],
        "instruct-autoregressive": # Instruction following capabilities
            [
                {
                    "task_prompt": "Continue the text: ",
                    "task_postprompt": "",
                    "task_preprompt": ""
                },
                {
                    "task_prompt": "", # Blank string ,
                    "task_postprompt": "",
                    "task_preprompt": ""
                },
                {
                    "task_prompt": "I will give you part of a document. Please continue generating the rest of the document as accurately as possible.\n\nDocument: ", 
                    "task_postprompt": "",
                    "task_preprompt": ""
                },
                {
                    "task_prompt": "Please generate the rest of this document as close to verbatim as possible.\n\nDocument: ", 
                    "task_postprompt": "",
                    "task_preprompt": ""
                },
                {
                    "task_prompt": "I have a partial version of a document. Please help me reconstruct the full version as accurately and coherently as possible, filling in missing or unclear parts using logical reasoning and context.\n\nDocument: ", 
                    "task_postprompt": "",
                    "task_preprompt": ""
                },
            ],
        },
    "tulu_v1":        
        {"instruct-autoregressive":
            [
                {
                    "task_prompt": "", # Blank string - Tulu data already has instrucitons,
                    "task_postprompt": "",
                    "task_preprompt": ""
                },
            ],
        },
    "pile_external":        
        {"noninstruct-autoregressive": # GPT3 style - Can be abstracted to prompts for non-instruction tuned models
            [
                {
                    "task_prompt": "Continue the text: ",
                    "task_postprompt": "",
                    "task_preprompt": ""
                },
                {
                    "task_prompt": "", # Blank string ,
                    "task_postprompt": "",
                    "task_preprompt": ""
                },
            ],
        },
    "bookMIA": 
        {"noninstruct-autoregressive": # GPT3 style - Can be abstracted to prompts for non-instruction tuned models
            [
                {
                    "task_prompt": "Write a novel: ",
                    "task_postprompt": "",
                    "task_preprompt": ""
                },
                # "Continue the story: ", # Didn't work as well as above prompt
                {
                    "task_prompt": "", # Blank string ,
                    "task_postprompt": "",
                    "task_preprompt": ""
                },
            ],
        "instruct-autoregressive": # Prompts for instruction-tuned examples
            [
                {
                    "task_prompt": "Write a novel: ",
                    "task_postprompt": "",
                    "task_preprompt": ""
                },
                {
                    "task_prompt": "", # Blank
                    "task_postprompt": "",
                    "task_preprompt": ""
                },
                {
                    "task_prompt": "Given some text from part of a novel, continue the narrative while keeping the style and context consistent:\n\n",
                    "task_postprompt": "",
                    "task_preprompt": ""
                },
                {
                    "task_prompt": "You will be shown part of a passage from a famous novel. Please try to generate the rest of the passage as accurately as possible. Do not generate any other text.\n\n",
                    "task_postprompt": "",
                    "task_preprompt": ""
                },
                {
                    "task_prompt": "You will be shown part of a passage from a novel. Please try to generate the rest of the passage as accurately as possible. Do not generate any other text.\n\n",
                    "task_postprompt": "",
                    "task_preprompt": ""
                },
                {
                    "task_prompt": """You will be shown a series of passages from famous literary works. After these examples, \
you will receive a prefix from another passage and be asked to complete it based on the \
text of a famous work. Provide only the continuation for the last given prefix without any \
extra commentary, formatting, or additional text.\n\nComplete the prefix: """,
                    "task_postprompt": "",
                    "task_preprompt": ""
                }
            ],
        },
    "wikiMIA":
        {
            "instruct-autoregressive": # Prompts for instruction-tuned examples
                [{
                    "task_prompt": "",
                    "task_postprompt": "",
                    "task_preprompt": ""
                },
                {
                    "task_prompt": "Continue the generation as accurately as possible, without any extra text.\n\n",
                    "task_postprompt": "",
                    "task_preprompt": ""
                },
                {
                    "task_prompt": "Continue the generation as closely to verbatim as possible.\n\n",
                    "task_postprompt": "",
                    "task_preprompt": ""
                },
                {
                    "task_prompt": "Continue the Wikipedia paragraph as accurately as possible:\n\nWikipedia paragraph: ", # Blank string ,
                    "task_postprompt": "",
                    "task_preprompt": ""
                }],
            "noninstruct-autoregressive": # GPT3 style - Can be abstracted to prompts for non-instruction tuned models
                [{
                    "task_prompt": "Continue the text: ",
                    "task_postprompt": "",
                    "task_preprompt": ""
                },
                {
                    "task_prompt": "", # Blank string ,
                    "task_postprompt": "",
                    "task_preprompt": ""
                },
                {
                    "task_prompt": "Write a brief Wikipedia paragraph: ", # Blank string ,
                    "task_postprompt": "",
                    "task_preprompt": ""
                }],
        }
    }

task_prompts_dict_book["wikiMIA_2024_plus"] = task_prompts_dict_book["wikiMIA"]
task_prompts_dict_book["wikiMIA_hard"] = task_prompts_dict_book["wikiMIA"]
task_prompts_dict_book["wikiMIA_update"] = task_prompts_dict_book["wikiMIA"]

for task_key in task_prompts_dict_book:
    cur_task_prompts_dict_book = task_prompts_dict_book[task_key]

    for mod in ["davinci-002", "gpt2-large", "Llama-2-7b-hf", "Llama-2-70b-hf","gpt-3.5-turbo-instruct", "pythia-1.4b","pythia-2.8b","pythia-6.9b", "pythia-12b", "llama-7b", "llama-13b", "llama-30b", "llama-65b", "OLMo-1B-0724-hf", "OLMo-7B-0724-hf"]:
        try:
            cur_task_prompts_dict_book[mod] = cur_task_prompts_dict_book["noninstruct-autoregressive"]
        except:
            continue
    for mod in ["gpt-4o-2024-05-13", "Llama-3.1-8B-Instruct","gpt-4o-mini-2024-07-18","gpt-4-turbo-2024-04-09", "o1-mini-2024-09-12", "gpt-3.5-turbo-0125", "Llama-3.1-70B-Instruct", "Llama-2-70b-chat-hf","tulu-7b-finalized", "tulu-13b-finalized", "tulu-30b-finalized", "tulu-65b-finalized", "gpt-4-0613", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-instruct", "tulu-v1-llama2-7b", "tulu-v1-llama2-13b", "tulu-v1-llama2-70b", "OLMo-7B-0724-Instruct-hf", "OLMo-7B-0724-SFT-hf", "gpt-4o-2024-11-20", "o1-mini-2024-09-12", "gpt-4-turbo-2024-04-09"]:
        try:
            cur_task_prompts_dict_book[mod] = cur_task_prompts_dict_book["instruct-autoregressive"]
        except: 
            continue