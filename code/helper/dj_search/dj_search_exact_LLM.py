from code.user_secrets import CACHE_PATH
import os
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
from code.helper.dj_search.dj_search_exact import Document, Span, Hypothesis
from datasets import load_dataset, Dataset
import datasets
from datasets.utils.logging import disable_progress_bar
import re
from IPython import embed
from typing import List
from multiprocessing import Pool, cpu_count
import time

datasets.logging.set_verbosity_error()
disable_progress_bar() # Disable filter progress bar
md = MosesDetokenizer(lang='en')

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
        if len(tgt_doc.tokens) <= min_ngram: # Edge case if gen is too short
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

# def process_single_doc_star(args):
#     return process_single_doc(*args)

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
    # embed()

    all_outputs = []

    if num_cpus > 1:
        combinations = [(t_idx, all_gens, min_ngram, source_docs) for t_idx, all_gens in enumerate(data)]
        print(f"Launching search in parallel with {num_cpus} on {len(combinations)} inputs")

        with Pool(num_cpus) as pool:
            all_outputs = list(pool.starmap(process_single_doc, tqdm(combinations, total=len(combinations), position=0)))
            # all_outputs = list(tqdm(pool.imap(process_single_doc_star, combinations), total=len(combinations), desc="Processing in parallel", position=0)) # This was slower

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

    if args.task == "bookMIA":
        if args.source_docs is None: # Case when we want to reference only against the original text
            source_docs = [Dataset.from_dict({"text": [unidecode(g["snippet_no_prompt"])]}) for g in generations]
            args.output_file = args.output_file.replace(".jsonl", "_onedoc.jsonl")
        else:
            # For each book index in the generations, all snippets from the dataset should be the source data
            ds = load_dataset(args.source_docs)
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

            # Make the source docs by iterating through through the generations
            source_docs = []
            for g in generations:
                cur_book_id = g["book_id"]
                cur_snippet_id = g["snippet_id"]
                # All snippets excluding original one
                all_book_snippets = df[(df["book_id"] == cur_book_id) & (df["snippet_id"] != cur_snippet_id)].snippet.to_list()
                all_book_snippets.append(g["snippet_no_prompt"]) # Add original snippet
                all_book_snippets_cleaned = [unidecode(x) for x in all_book_snippets]
                source_docs.append(Dataset.from_dict({"text": all_book_snippets_cleaned}))

            args.output_file = args.output_file.replace(".jsonl", "_alldoc.jsonl")

    num_workers = min(cpu_count(), 4) if args.parallel else 1 # Set to 16 since it will get killed with too many cpus
    all_outputs = dj_search(generation_texts, source_docs, args.min_ngram, args.subset, num_workers)

    execution_time = time.time() - start_time
    minutes = int(execution_time // 60)
    seconds = int(execution_time % 60)
    print(f"Program executed in {minutes}:{seconds} ({execution_time:.4f} seconds)")

    assert len(all_outputs) == len(generations)

    for cur_generation, cur_all_output in zip(generations, all_outputs):
        cur_generation["coverage"] = cur_all_output

    # Can't save in progress if we're using multiprocessing - too complex
    with open(args.output_file, 'w') as f:
        json.dump(generations, f, indent=4)
        f.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='bookMIA',
                        help="which type of corpus to analyze")
    parser.add_argument('--gen_data', type=str)
    parser.add_argument('--source_docs', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument("--min_ngram", type=int, default=3,
                        help="minimum n-gram size")
    parser.add_argument("--subset", type=int, default=int(1e6),
                        help="size of example subset to run search on")
    parser.add_argument("--generation_field", type=str, default="generation",
                        help="Name of generations")
    parser.add_argument('--parallel', action="store_true")

    args = parser.parse_args()
    main(args)


"""
Example commands to run:

python3 -m code.helper.dj_search.dj_search_exact_LLM \
    --task bookMIA \
    --output_dir /gscratch/xlab/hallisky/membership-inference/outputs/ours/bookMIA/coverages/train \
    --gen_data /gscratch/xlab/hallisky/membership-inference/outputs/ours/bookMIA/generations/train/gpt-3.5-turbo-0125_maxTok512_minTok0_numSeq20_topP0.95_temp1.0_numSent5_startSent1_numWord-1_startWord-1_useSentF_promptIdx5_len494_2025-01-11-23:06:39.jsonl \
    --min_ngram 4 \
    --parallel \
    --source_docs swj0419/BookMIA;

python3 -m code.helper.dj_search.dj_search_exact_LLM \
    --task bookMIA \
    --output_dir /gscratch/xlab/hallisky/membership-inference/outputs/ours/bookMIA/coverages/train \
    --gen_data /gscratch/xlab/hallisky/membership-inference/outputs/ours/bookMIA/generations/train/gpt-3.5-turbo-0125_maxTok512_minTok0_numSeq20_topP0.95_temp1.0_numSent5_startSent1_numWord-1_startWord-1_useSentF_promptIdx5_len494_2025-01-11-23:06:39.jsonl \
    --min_ngram 4 \
    --parallel;

"""