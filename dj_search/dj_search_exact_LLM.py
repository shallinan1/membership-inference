from user_secrets import CACHE_PATH
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
from dj_search.dj_search_exact import Document, Span, Hypothesis
from datasets import load_dataset, Dataset
import datasets
from datasets.utils.logging import disable_progress_bar
import re
from IPython import embed
from typing import List

datasets.logging.set_verbosity_error()
disable_progress_bar() # Disable filter progress bar
md = MosesDetokenizer(lang='en')


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


def dj_search(generation_texts_list: List[List[str]],
              source_docs, output_file: str, min_ngram: int, 
              subset: int = None, 
              num_cpus: int = 1):
    data = [[{"text": g} for g in generation_texts] for generation_texts in generation_texts_list]
    data = data[:subset] if subset is not None else data

    tokenize_func = lambda x: nltk.tokenize.casual.casual_tokenize(x)
    detokenize = lambda x: md.detokenize(x)

    all_outputs = []

    # if os.path.isfile(output_file):
    #     outputs = json.load(open(output_file, 'r'))
    #     data = data[len(outputs):]
    #     print(f'resume from previous output file with {len(outputs)} items')

    for t_idx, all_gens in tqdm(enumerate(data), desc='target gens', total=len(data)):
        outputs = []
        for t_doc in all_gens:
            tokenized_text = tokenize_func(unidecode(t_doc["text"]))
            tgt_doc = Document(f'tgt_{t_idx}', tokenized_text)
            if len(tgt_doc.tokens) <= min_ngram:
                continue

            output = find_exact_match(detokenize, tgt_doc, min_ngram, source_docs[t_idx], num_cpus)
            t_doc.update(output)
            outputs.append(t_doc)

            avg_coverage = np.average([x['coverage'] for x in outputs])
            std = np.std([x['coverage'] for x in outputs])
            avg_len = np.average([x['avg_span_len'] for x in outputs])
            print(f'average {min_ngram}-ngram coverage: {avg_coverage:.3f}, std: {std:.3f}, average length: {avg_len}')

        all_outputs.append(outputs)
        with open(output_file, 'w') as f:
            json.dump(all_outputs, f, indent=4)
            f.flush()
def main():
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

    args = parser.parse_args()

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

    if args.task == "bookMIA":
        if args.source_docs is None: # Case when we want to reference only against the original text
            source_docs = [Dataset.from_dict({"text": [g["snippet"]]}) for g in generations]
            args.output_file = args.output_file.replace(".jsonl", "_onedoc.jsonl")
        else:
            # For each book index in the generations, all snippets from the dataset should be the source data
            # Load dataset
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
                all_book_snippets = df[df["book_id"] == cur_book_id].snippet.to_list()
                source_docs.append(Dataset.from_dict({"text": all_book_snippets}))

            args.output_file = args.output_file.replace(".jsonl", "_alldoc.jsonl")

    dj_search(generation_texts, source_docs, args.output_file, args.min_ngram, args.subset)

if __name__ == '__main__':
    main()


"""
python3 -m dj_search.dj_search_exact_LLM \
    --task bookMIA \
    --output_dir /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/coverages/ \
    --gen_data /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/generations/davinci-002_maxTokens512_numSeq1_topP0.95_numSent1_promptIdx2_len100.jsonl \
    --min_ngram 5;

python3 -m dj_search.dj_search_exact_LLM \
    --task bookMIA \
    --output_dir /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/coverages/ \
    --gen_data /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/generations/davinci-002_maxTokens512_numSeq1_topP0.95_numSent1_promptIdx1_len100.jsonl \
    --min_ngram 5;

python3 -m dj_search.dj_search_exact_LLM \
    --task bookMIA \
    --output_dir /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/coverages/ \
    --gen_data /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/generations/davinci-002_maxTokens512_numSeq1_topP0.95_numSent1_promptIdx0_len100.jsonl \
    --min_ngram 5;

python3 -m dj_search.dj_search_exact_LLM \
    --task bookMIA \
    --output_dir /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/coverages/ \
    --gen_data /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/generations/davinci-002_maxTokens512_numSeq1_topP0.95_numSent1_promptIdx3_len100.jsonl \
    --min_ngram 5;

    




python3 -m dj_search.dj_search_exact_LLM \
    --task bookMIA \
    --output_dir /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/coverages/ \
    --gen_data /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/generations/davinci-002_maxTokens512_numSeq1_topP0.95_numSent1_promptIdx2_len100.jsonl \
    --min_ngram 4 \
    --source_docs swj0419/BookMIA;

python3 -m dj_search.dj_search_exact_LLM \
    --task bookMIA \
    --output_dir /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/coverages/ \
    --gen_data /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/generations/davinci-002_maxTokens512_numSeq1_topP0.95_numSent1_promptIdx1_len100.jsonl \
    --min_ngram 4 \
    --source_docs swj0419/BookMIA;

python3 -m dj_search.dj_search_exact_LLM \
    --task bookMIA \
    --output_dir /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/coverages/ \
    --gen_data /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/generations/davinci-002_maxTokens512_numSeq1_topP0.95_numSent1_promptIdx0_len100.jsonl \
    --min_ngram 4 \
    --source_docs swj0419/BookMIA;

python3 -m dj_search.dj_search_exact_LLM \
    --task bookMIA \
    --output_dir /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/coverages/ \
    --gen_data /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/generations/davinci-002_maxTokens512_numSeq1_topP0.95_numSent1_promptIdx3_len100.jsonl \
    --min_ngram 4 \
    --source_docs swj0419/BookMIA;

    
python3 -m dj_search.dj_search_exact_LLM \
    --task bookMIA \
    --output_dir /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/coverages/ \
    --gen_data /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/generations/gpt-3.5-turbo-0125_maxTokens512_numSeq10_topP0.95_numSent5_startSent1_promptIdx0_len100.jsonl \
    --min_ngram 4;

python3 -m dj_search.dj_search_exact_LLM \
    --task bookMIA \
    --output_dir /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/coverages/ \
    --gen_data /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/generations/gpt-3.5-turbo-0125_maxTokens512_numSeq10_topP0.95_numSent5_startSent1_promptIdx3_len100.jsonl \
    --min_ngram 4;

python3 -m dj_search.dj_search_exact_LLM \
    --task bookMIA \
    --output_dir /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/coverages/ \
    --gen_data /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/generations/gpt-3.5-turbo-0125_maxTokens512_numSeq10_topP0.95_numSent5_startSent1_promptIdx4_len100.jsonl \
    --min_ngram 4;

python3 -m dj_search.dj_search_exact_LLM \
    --task bookMIA \
    --output_dir /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/coverages/ \
    --gen_data /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/generations/gpt-3.5-turbo-0125_maxTokens512_numSeq10_topP0.95_numSent5_startSent1_promptIdx5_len100.jsonl \
    --min_ngram 4;



python3 -m dj_search.dj_search_exact_LLM \
    --task bookMIA \
    --output_dir /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/coverages/ \
    --gen_data /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/generations/gpt-4o-mini-2024-07-18_maxTokens512_numSeq10_topP0.95_numSent3_startSent1_promptIdx0_len100.jsonl \
    --min_ngram 4;

python3 -m dj_search.dj_search_exact_LLM \
    --task bookMIA \
    --output_dir /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/coverages/ \
    --gen_data /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/generations/gpt-4o-mini-2024-07-18_maxTokens512_numSeq10_topP0.95_numSent3_startSent1_promptIdx3_len100.jsonl \
    --min_ngram 4;

python3 -m dj_search.dj_search_exact_LLM \
    --task bookMIA \
    --output_dir /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/coverages/ \
    --gen_data /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/generations/gpt-4o-mini-2024-07-18_maxTokens512_numSeq10_topP0.95_numSent3_startSent1_promptIdx4_len100.jsonl \
    --min_ngram 4;

python3 -m dj_search.dj_search_exact_LLM \
    --task bookMIA \
    --output_dir /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/coverages/ \
    --gen_data /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/generations/gpt-4o-mini-2024-07-18_maxTokens512_numSeq10_topP0.95_numSent3_startSent1_promptIdx5_len100.jsonl \
    --min_ngram 4;

python3 -m dj_search.dj_search_exact_LLM \
    --task bookMIA \
    --output_dir /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/coverages/ \
    --gen_data /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/generations/gpt-4o-mini-2024-07-18_maxTokens512_numSeq10_topP0.95_numSent3_startSent1_promptIdx0_len100.jsonl \
    --min_ngram 4 \
    --source_docs swj0419/BookMIA;

python3 -m dj_search.dj_search_exact_LLM \
    --task bookMIA \
    --output_dir /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/coverages/ \
    --gen_data /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/generations/gpt-4o-mini-2024-07-18_maxTokens512_numSeq10_topP0.95_numSent3_startSent1_promptIdx3_len100.jsonl \
    --min_ngram 4 \
    --source_docs swj0419/BookMIA;

python3 -m dj_search.dj_search_exact_LLM \
    --task bookMIA \
    --output_dir /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/coverages/ \
    --gen_data /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/generations/gpt-4o-mini-2024-07-18_maxTokens512_numSeq10_topP0.95_numSent3_startSent1_promptIdx4_len100.jsonl \
    --min_ngram 4 \
    --source_docs swj0419/BookMIA;

python3 -m dj_search.dj_search_exact_LLM \
    --task bookMIA \
    --output_dir /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/coverages/ \
    --gen_data /gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/generations/gpt-4o-mini-2024-07-18_maxTokens512_numSeq10_topP0.95_numSent3_startSent1_promptIdx5_len100.jsonl \
    --min_ngram 4 \
    --source_docs swj0419/BookMIA;
"""
