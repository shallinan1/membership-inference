import json
from tqdm import tqdm
# Load gen_path as jsonl
import re

def convert_to_tulu_v1_format(messages, turns = 1):
    messages = messages[:2*turns] # Check - does it always have to be user-assistnat or can it also be user-user-assistant?
    # TODO for multi-turn conversations, do we add eos after every asssitant? 
    # TODO look into training to see how the eos stuff adds up
    final_text = ""
    for c in messages:
        if c["role"] == "user":
            final_text += f"<|user|>\n"
        else:
            final_text += f"<|assistant|>\n"

        final_text += c["content"] + "\n"

    return final_text.strip()

def convert_to_tulu_v1_open(text):
    return f"<|user|>\n{text}<|assistant|>\n"

def remove_first_sentence_if_needed(text):
    # Match the first sentence using regex
    match = re.match(r"([^.!?]+[.!?])\s*(.*)", text, re.DOTALL)
    
    if match:
        first_sentence, rest = match.groups()
        # Check if the first character is not uppercase or a quotation mark
        if not first_sentence.lstrip()[0].isupper() and not first_sentence.lstrip().startswith(("'", '"')):
            return rest.strip()  # Return the rest without the first sentence
    return text  # Return the original text if no change is needed