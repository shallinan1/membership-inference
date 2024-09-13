import os
import nltk
import json
import argparse
import numpy as np
from tqdm import tqdm
from typing import Callable
from unidecode import unidecode
from sacremoses import MosesDetokenizer
from DJ_search_exact import Document, Span, Hypothesis
from datasets import load_dataset, Dataset

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

            print("***************************************************************************************************")
            print(hypothesis.format_span())
            print(f'score: {hypothesis.get_score():.4f}  avg_span_length: {hypothesis.get_avg_span_len()}')
            print("***************************************************************************************************")

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


def dj_search(data_path, source_data_path, output_file, min_ngram, subset=100, num_cpus=96):
    data = json.load(open(data_path))[:subset]
    source_data = load_dataset("json", data_files=source_data_path)['train']

    tokenize_func = lambda x: nltk.tokenize.casual.casual_tokenize(x)
    detokenize = lambda x: md.detokenize(x)

    outputs = []
    if os.path.isfile(output_file):
        outputs = json.load(open(output_file, 'r'))
        data = data[len(outputs):]
        print(f'resume from previous output file with {len(outputs)} items')

    for t_idx, t_doc in tqdm(enumerate(data), desc='target docs', total=len(data)):
        tokenized_text = tokenize_func(unidecode(t_doc['text']))
        tgt_doc = Document(f'tgt_{t_idx}', tokenized_text)
        if len(tgt_doc.tokens) <= min_ngram:
            continue

        output = find_exact_match(detokenize, tgt_doc, min_ngram, source_data, num_cpus)
        t_doc.update(output)
        outputs.append(t_doc)

        avg_coverage = np.average([x['coverage'] for x in outputs])
        std = np.std([x['coverage'] for x in outputs])
        avg_len = np.average([x['avg_span_len'] for x in outputs])
        print(f'average {min_ngram}-ngram coverage: {avg_coverage:.3f}, std: {std:.3f}, average length: {avg_len}')

        with open(output_file, 'w') as f:
            json.dump(outputs, f, indent=4)
            f.flush()


def main():
    parser = argparse.ArgumentParser()
    data_path = '/net/nfs.cirrascale/mosaic/ximinglu/complexity'
    parser.add_argument('--task', type=str, default='gpt3.5_new_book',
                        help="which type of corpus to analyze")
    parser.add_argument('--data', type=str,
                        default='data/new_book/gpt3.5_new_book.json')
    parser.add_argument('--source_data', type=str,
                        default=f'{data_path}/data/LM_data/WildChat-Arena.json')
    parser.add_argument('--output_dir', type=str,
                        default=f'{data_path}/outputs_LM')
    parser.add_argument("--min_ngram", type=int, default=3,
                        help="minimum n-gram size")
    parser.add_argument("--subset", type=int, default=100,
                        help="size of example subset to run search on")

    args = parser.parse_args()
    args.output_file = os.path.join(args.output_dir, args.task + '.json')
    dj_search(args.data, args.source_data, args.output_file, args.min_ngram, args.subset)


if __name__ == '__main__':
    main()

