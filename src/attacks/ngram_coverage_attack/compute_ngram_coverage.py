"""
N-gram Coverage Computation for Membership Inference Attack

This module implements the coverage computation component of the N-Gram Coverage Attack 
method for membership inference attacks.

The module analyzes generated text continuations by computing exact n-gram matches
against source documents and calculating coverage statistics for membership inference.

Pipeline:
    1. Load generated text continuations from step 1 (generate.py)
    2. Tokenize and process both generated texts and source documents
    3. Find exact n-gram matches using sliding window approach
    4. Calculate coverage percentage (portion of text covered by matching n-grams)
    5. Compute longest common substring and subsequence metrics
    6. Save results with metadata for creativity index calculation

Outputs:
    JSONL file containing coverage statistics, span information, and length metrics
    saved to outputs/ours/{task}/coverages/{data_split}/

Analysis Modes:
    - Single document: Compare generations against original snippet only
    - Multi-document: Compare against entire corpus (for bookMIA with source_docs)

Hardcoded Configuration:
    - Tokenization: Uses NLTK casual tokenizer for all text processing
    - Detokenization: Uses Moses detokenizer for English text reconstruction
    - Parallel processing: Limited to 4 CPU cores max to prevent memory issues

Usage:
    python -m src.attacks.ngram_coverage_attack.compute_ngram_coverage \
        --task TASK_NAME \
        --gen_data PATH_TO_GENERATIONS \
        --output_dir OUTPUT_DIRECTORY \
        --min_ngram N \
        [--source_docs DATASET_NAME] \
        [--parallel]
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
CACHE_PATH = os.getenv("CACHE_PATH")

# Set up environment variables
os.environ["HF_HOME"] = CACHE_PATH
os.environ["HF_DATASETS_PATH"] = CACHE_PATH

import nltk
import json
import argparse
import numpy as np
from tqdm import tqdm
from typing import Callable
from unidecode import unidecode
from sacremoses import MosesDetokenizer
from datasets import load_dataset, Dataset
import datasets
from dataclasses import dataclass
from datasets.utils.logging import disable_progress_bar
import re
from IPython import embed
from typing import List
from multiprocessing import Pool, cpu_count
import time

datasets.logging.set_verbosity_error()
disable_progress_bar() # Disable filter progress bar
md = MosesDetokenizer(lang='en')
from pylcs import lcs_string_length

tokenize_func = lambda x: nltk.tokenize.casual.casual_tokenize(x)
detokenize = lambda x: md.detokenize(x)

# BAD_START_STRINGS = [
#     "I cannot create",
#     "I cannot provide",
#     "I cannot generate",
#     "I'm sorry, but I can't",
#     "I'm sorry, this doesn't",
#     "I can't create content",
#     "I cannot continue this passage",
#     "I'm not able to provide a response.",
# ]

@dataclass
class Document:
    doc_id: str
    tokens: List[str]  # [num_tokens]

@dataclass
class Span:
    start_index: int
    end_index: int
    span_text: str
    occurrence: int

class Hypothesis:
    def __init__(self, target_doc: Document, min_ngram: int) -> None:
        self.target_doc = target_doc
        self.min_ngram = min_ngram
        self.spans = []
        self.finished = False

    def add_span(self, new_span: Span) -> None:
        self.spans.append(new_span)
        if new_span.end_index >= len(self.target_doc.tokens):
            self.finished = True

    def replace_span(self, new_span: Span) -> None:
        self.spans = self.spans[:-1] + [new_span]
        if new_span.end_index >= len(self.target_doc.tokens):
            self.finished = True

    def get_score(self) -> float:
        if not self.spans:
            return 0
        progress_len = self.spans[-1].end_index if not self.finished else len(self.target_doc.tokens)
        flags = [False for _ in range(progress_len)]
        for span in self.spans:
            span_length = span.end_index - span.start_index
            flags[span.start_index: span.end_index] = [True] * span_length
        coverage = len([fa for fa in flags if fa]) / len(flags)
        return coverage

    def format_span(self) -> str:
        return ' | '.join([s.span_text for s in self.spans])

    def __hash__(self) -> int:
        return hash(self.format_span())

    def __eq__(self, other) -> bool:
        if isinstance(other, Hypothesis):
            return self.format_span() == other.format_span()
        return NotImplemented

    def get_avg_span_len(self) -> float:
        if not self.spans:
            return 0
        span_len = [s.end_index - s.start_index for s in self.spans]
        return sum(span_len) / len(span_len)

    def export_json(self) -> dict:
        matched_spans = [{'start_index': s.start_index,
                          'end_index': s.end_index,
                          'span_text': s.span_text,
                          'occurrence': s.occurrence} for s in self.spans]
        return {
            'matched_spans': matched_spans,
            'coverage': self.get_score(),
            'avg_span_len': self.get_avg_span_len(),
        }

def find_max_common_sublist_length(A: List, B: List):
    n = len(A); m = len(B)
    
    # Auxiliary dp[] list
    dp = [0] * (m + 1)
    maxm = 0
 
    # Updating the dp[] list in Bottom Up approach
    for i in range(n - 1, -1, -1):
        prev = 0
        for j in range(m - 1, -1, -1):
            temp = dp[j]
            if A[i] == B[j]:
                dp[j] = prev + 1
                maxm = max(maxm, dp[j])
            else:
                dp[j] = 0
            prev = temp
 
    return maxm  # Return the maximum length

def process_single_sublist_pair(g_list_tokenized, source_docs_tokenized):
    return [longest_sublist(g, source_docs_tokenized) for g in g_list_tokenized]

def longest_sublist(text_tokenized: List[str], 
                    source_docs_tokenized: List[List[str]] ):
    longest_sublist_overall = 0
    for s in source_docs_tokenized:
        match_length = find_max_common_sublist_length(text_tokenized, s)
        longest_sublist_overall = max(longest_sublist_overall, match_length)
    
    return longest_sublist_overall

def process_single_substring_pair(g_list, source_docs):
    return [longest_substring(g, source_docs) for g in g_list]

def longest_substring(text: str, source_docs: List[str]):
    longest_substring_overall = 0
    for s in source_docs:
        match_length = lcs_string_length(text, s)
        longest_substring_overall = max(longest_substring_overall, match_length)
    
    return longest_substring_overall

def find_exact_match(detokenize: Callable, doc: Document, min_ngram: int, source_data: Dataset, num_cpus: int):
    hypothesis = Hypothesis(doc, min_ngram)

    first_pointer, second_pointer = 0, min_ngram
    while second_pointer <= len(doc.tokens):
        span_text = detokenize(doc.tokens[first_pointer: second_pointer])
        hit_data = source_data.filter(lambda x: span_text in x['text'], num_proc=num_cpus)
        occurrence = len(hit_data)

        if occurrence:
            matched_span = Span(start_index=first_pointer,
                                end_index=second_pointer,
                                span_text=span_text,
                                occurrence=occurrence)
            if not hypothesis.spans:
                hypothesis.add_span(matched_span)
            else:
                last_span = hypothesis.spans[-1]
                if matched_span.start_index <= last_span.start_index and last_span.end_index <= matched_span.end_index:
                    hypothesis.replace_span(matched_span)
                else:
                    hypothesis.add_span(matched_span)
            second_pointer += 1

            # print("***************************************************************************************************")
            # print(hypothesis.format_span())
            # print(f'score: {hypothesis.get_score():.4f}  avg_span_length: {hypothesis.get_avg_span_len()}')
            # print("***************************************************************************************************")

        else:
            if second_pointer - first_pointer > min_ngram:
                first_pointer += 1
            elif second_pointer - first_pointer == min_ngram:
                first_pointer += 1
                second_pointer += 1
            else:
                raise ValueError

    hypothesis.finished = True
    return hypothesis.export_json()

def process_single_doc(t_idx, all_gens, min_ngram, source_docs):
    outputs = []
    for t_doc in tqdm(all_gens, leave=False, position=1):
        tokenized_text = tokenize_func(unidecode(t_doc["text"]))
        tgt_doc = Document(f'tgt_{t_idx}', tokenized_text)
        if len(tgt_doc.tokens) < min_ngram: # Edge case if gen is too short
            output = {'matched_spans': [], 'coverage': 0, 'avg_span_len': 0, 'too_short': True}
        else:
            output = find_exact_match(detokenize, tgt_doc, min_ngram, source_docs[t_idx], 1) # Hardcode 1 CPU here
        # Add it to the existing dict which stores the text
        t_doc.update(output)
        outputs.append(t_doc)

    # avg_coverage = np.average([x['coverage'] for x in outputs]) if outputs else 0
    # std = np.std([x['coverage'] for x in outputs]) if outputs else 0
    # avg_len = np.average([x['avg_span_len'] for x in outputs]) if outputs else 0
    # print(f'average {min_ngram}-ngram coverage: {avg_coverage:.3f}, std: {std:.3f}, average length: {avg_len}')

    return outputs

def dj_search(generation_texts_list: List[List[str]],
              source_docs, 
              min_ngram: int, 
              subset: int = None, 
              num_cpus: int = 1):
    """
    Perform a document search by processing generated texts and finding matches in source documents.

    Args:
        generation_texts_list (List[List[str]]): A list of lists containing generated text strings to be processed.
        source_docs: A collection of source documents to match against the generated texts.
        min_ngram (int): The minimum n-gram length for considering a match.
        subset (int, optional): The number of items to process from the generation_texts_list. Defaults to None (process all).
        num_cpus (int, optional): The number of CPU cores to use for multiprocessing. Defaults to 1.

    Returns:
        None: The function writes the output to the specified output file.
    """
    data = [[{"text": g} for g in generation_texts] for generation_texts in generation_texts_list]
    data = data[:subset] if subset is not None else data

    all_outputs = []

    if num_cpus > 1:
        combinations = [(t_idx, all_gens, min_ngram, source_docs) for t_idx, all_gens in enumerate(data)]
        print(f"Launching search in parallel with {num_cpus} on {len(combinations)} inputs")

        with Pool(num_cpus) as pool:
            all_outputs = list(pool.starmap(process_single_doc, tqdm(combinations, total=len(combinations), position=0)))

    else:
        print("Launching search iteratively")
        for t_idx, all_gens in tqdm(enumerate(data), desc='target gens', total=len(data)):         
            outputs = process_single_doc(t_idx, all_gens, min_ngram, source_docs)
            all_outputs.append(outputs)

    return all_outputs

def main(args):
    start_time = time.time()

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Make the name of the output file the same as the input file but in the args.output_dir
    input_filename = os.path.basename(args.gen_data)  # Extract filename from source_docs path
    args.output_file = os.path.join(args.output_dir, input_filename).replace(".jsonl", f"_{args.min_ngram}.jsonl")

    # Load the input data from a JSONL file
    generations = []
    with open(args.gen_data, 'r') as f:
        for line in f:
            generations.append(json.loads(line.strip()))

    generation_texts = [g[args.generation_field] for g in generations]
    # generation_texts = [[text for text in text_list if not any(text.startswith(bad_start) for bad_start in BAD_START_STRINGS)] for text_list in generation_texts]

    # Merge the list of lists
    num_sequences = len(generation_texts[0][0])
    generation_texts = [sum(g, []) for g in generation_texts]

    # We only use the original snippet for the source docs, but we could use more documents as well (need to modify source_docs to do so)
    source_docs = [Dataset.from_dict({"text": [unidecode(g["snippet_no_prompt"])]}) for g in generations]
    args.output_file = args.output_file.replace(".jsonl", "_onedoc.jsonl")
    print(args.output_file)

    num_workers = min(cpu_count(), 4) if args.parallel else 1 # Set to 4 since it will get killed with too many cpus
    all_outputs = dj_search(generation_texts, source_docs, args.min_ngram, args.subset, num_workers)

    execution_time = time.time() - start_time
    minutes = int(execution_time // 60)
    seconds = int(execution_time % 60)
    print(f"Program executed in {minutes}:{seconds} ({execution_time:.4f} seconds)")

    assert len(all_outputs) == len(generations)
    for cur_generation, cur_all_output in zip(generations, all_outputs):
        cur_generation["coverage"] = cur_all_output

    generation_texts_word_tokenized = [[tokenize_func(g) for g in g_list] for g_list in generation_texts] 
    source_docs_word_tokenized = [[tokenize_func(s) for s in source["text"]] for source in source_docs]
    
    # Length (number of words) and characters
    generation_texts_length_chars = [[len(g) for g in g_list] for g_list in generation_texts]
    generation_texts_length_words = [[len(g) for g in g_list] for g_list in generation_texts_word_tokenized]

    if num_workers > 1:
        combinations_substring = [(g_list, source["text"]) for g_list, source in zip(generation_texts, source_docs)]
        with Pool(num_workers) as pool:
            longest_substring_chars = list(pool.starmap(process_single_substring_pair, tqdm(combinations_substring, total=len(combinations_substring), desc="maximum substring", position=0))) 
    
        combinations_sublist = list(zip(generation_texts_word_tokenized, source_docs_word_tokenized))

        with Pool(num_workers) as pool:
            longest_sublist_words = list(pool.starmap(process_single_sublist_pair, tqdm(combinations_sublist,total=len(combinations_sublist), desc="maximum sublist", position=0)))
    else:
        # TODO check this
        longest_substring_chars = [[longest_substring(g,source["text"]) for g in g_list] for g_list,source in tqdm(zip(generation_texts, source_docs))]
        longest_sublist_words = [[longest_sublist(g, source_doc_word_tokenized) for g in g_list_tokenized] 
                                 for g_list_tokenized, source_doc_word_tokenized in tqdm(zip(generation_texts_word_tokenized, source_docs_word_tokenized))]

    for cur_generation, gen_text_length_char, gen_text_length_word, longest_substring_char, longest_sublist_word in zip(generations, generation_texts_length_chars, generation_texts_length_words, longest_substring_chars, longest_sublist_words):
        cur_generation["gen_text_length_char"] = gen_text_length_char
        cur_generation["gen_text_length_word"] = gen_text_length_word
        cur_generation["longest_substring_char"] = longest_substring_char
        cur_generation["longest_sublist_word"] = longest_sublist_word

    with open(args.output_file, 'w') as f:
        json.dump(generations, f, indent=4)
        f.flush()

    # TODO use num_sequences to potentially unmerge these and get stats wrt each prompt
    # embed()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="N-gram coverage attack tool for analyzing text generation and detecting potential membership inference vulnerabilities by comparing generated texts against source documents using exact n-gram matching and longest common subsequence analysis.")
    parser.add_argument('--task', type=str, default='bookMIA',
                        help="Type of corpus to analyze (bookMIA, tulu_v1, pile_external, wikiMIA, dolma_v17, articles). Determines how source documents are loaded and processed.")
    parser.add_argument('--gen_data', type=str, required=True,
                        help="Path to JSONL file containing generated text data to analyze for potential membership inference vulnerabilities")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory where analysis results will be saved. Output filename will be derived from input filename with ngram size suffix")
    parser.add_argument("--min_ngram", type=int, default=3,
                        help="Minimum n-gram size for exact matching analysis (default: 3). Larger values require longer exact matches")
    parser.add_argument("--subset", type=int, default=int(1e6),
                        help="Maximum number of examples to process from the input data (default: 1,000,000). Use smaller values for testing")
    parser.add_argument("--generation_field", type=str, default="generation",
                        help="Name of the field in the input JSONL containing the generated text sequences to analyze")
    parser.add_argument('--parallel', action="store_true",
                        help="Enable parallel processing using multiple CPU cores for faster analysis of large datasets")

    args = parser.parse_args()
    main(args)