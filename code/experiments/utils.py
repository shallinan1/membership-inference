import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any
from IPython import embed

# Function to plot ROC curve
def plot_roc_curve(fpr, tpr, roc_auc, strategy_title, save_path):
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # Diagonal line for random guess
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{strategy_title} ROC Curve')
    plt.grid(alpha=0.15)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

def zigzag_append(lists):
    return [item for group in zip(*lists) for item in group]

def chunk_list(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def remove_last_n_words(
        tokenizer,
        sentence: str, 
        n: int = 1,
        ) -> Tuple[List[str], List[str], int]:
    """
    Tokenizes a sentence, removes the last n words using offset mapping to identify
    word boundaries, and returns relevant information.
    
    Args:
        tokenizer: The tokenizer to use
        sentence (str): The input sentence
        n (int): Number of words to remove from the end (default: 1)
    Returns:
        Tuple containing:
            - List of tokens for the remaining text
            - List of tokens for the removed text
            - Number of tokens removed
    """
    
    # Tokenize with offset mapping
    encoding = tokenizer(sentence, return_offsets_mapping=True, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    offsets = encoding["offset_mapping"]
    
    # Identify word boundaries based on spaces in the original text
    word_boundaries = []
    in_whitespace = False
    
    for i in range(len(sentence)):
        is_whitespace = sentence[i] in [' ', '\n', '\r', '\t']
        # If we're moving from whitespace to a non-whitespace character,
        # we're at the start of a new word
        if in_whitespace and not is_whitespace:
            word_boundaries.append(i)
        in_whitespace = is_whitespace
    
    # Add the start of the first word (if the text doesn't begin with whitespace)
    if not (sentence and sentence[0] in [' ', '\n', '\r', '\t']):
        word_boundaries.insert(0, 0)
    
    if not word_boundaries or n > len(word_boundaries):
        return [], tokens, len(tokens) # Trying to remove more words than exist
    
    # Find the start position of the last n words
    if n == 0:
        return tokens, [], 0
    
    # Find the start position of the nth word from the end
    last_words_start = word_boundaries[-n] if n > 0 else len(sentence)
    
    # Find the index of the first token that belongs to the removed part
    split_index = 0
    for i, (start, end) in enumerate(offsets):
        if end <= last_words_start:
            split_index = i + 1
    
    # Use the split index to divide the tokens
    remaining_tokens = tokens[:split_index]
    removed_tokens = tokens[split_index:]
    
    embed()
    return remaining_tokens, removed_tokens, len(removed_tokens)

# Example usage
if __name__ == "__main__":
    import nltk
    from code.user_secrets import CACHE_PATH
    import os
    # Set up environment variables
    os.environ["HF_HOME"] = CACHE_PATH
    os.environ["HF_DATASETS_PATH"] = CACHE_PATH
    from transformers import AutoTokenizer


    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    tokenize_func = lambda x: nltk.tokenize.casual.casual_tokenize(x)    
    sentences = [
        "Natural language processing is an exciting field of artificial intelligence",
        "   Natural language processing is an exciting field of artificial intelligence",
        "We went to the store. Then we went fishing!",
        "ARTICLE\n\nMen go to the store often.\n\nBut do women?\tWe are not sure about that."
    ]
    
    for sentence in sentences:
        print(tokenize_func(sentence))

        # Remove last word
        remaining, removed, num_removed = remove_last_n_words(tokenizer, sentence, 1)
        print(f"Original: {sentence}")
        print(f"Remaining tokens: {remaining}")
        print(f"Removed tokens: {removed}")
        print(f"Number of tokens removed: {num_removed}")
        
        # Remove last 3 words
        remaining, removed, num_removed = remove_last_n_words(tokenizer, sentence, 3)
        print("\nRemoving last 3 words:")
        print(f"Remaining tokens: {remaining}")
        print(f"Removed tokens: {removed}")
        print(f"Number of tokens removed: {num_removed}")