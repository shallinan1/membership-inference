    
import os
from code.utils import load_jsonl, load_json, combine_lists, combine_dicts, combine_list_of_dicts
from enum import Enum
from nltk import sent_tokenize

def get_all_gen_paths(gen_path):
    # Assemble all the gen_paths
    gen_path_start = gen_path.rsplit("2024-", 1)[0]

    # Get all paths that might have the same starting filename, excluding the date
    directory = os.path.dirname(gen_path)
    gen_paths = []

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        # Check if the filename starts with the desired prefix
        if filename.startswith(os.path.basename(gen_path_start)):
            full_path = os.path.join(directory, filename)
            gen_paths.append(full_path)

    gen_paths = sorted(gen_paths) # Important - sort!
    return gen_paths

def load_all_files(gen_path, all_doc=False, min_ngram=5, low_ci_bound=3, high_ci_bound=12):    

    gen_paths = get_all_gen_paths(gen_path)

    # Exit if we are not the alphabetically first one to reduce redandcy
    if gen_path not in gen_paths[0]:
        return

    doc_string = "onedoc" if not all_doc else "alldoc"

    coverage_paths = [cur_path.replace(".jsonl", f"_{min_ngram}_{doc_string}.jsonl").replace("generations", "coverages") for cur_path in gen_paths]
    ci_paths = [cur_path.replace(".jsonl", f"_CI_{low_ci_bound}_{high_ci_bound}.jsonl").replace("coverages", "cis") for cur_path in coverage_paths]
    for c in coverage_paths:
        assert os.path.exists(c)
    for i, c in enumerate(ci_paths):
        if i != 0:
            assert not os.path.exists(c) # Shouldn't exist!
    ci_path = ci_paths[0]

    # Initialize empty lists for combined generation data and coverage data
    combined_gen_data = []
    combined_coverage_data = []
    # Iterate through the found generation JSONL files and load their data
    for gen_path, coverage_path in zip(gen_paths, coverage_paths):
        # Load the jsonl for generation data
        gen_data = load_jsonl(gen_path)
        coverage_data = load_json(coverage_path)

        # Combine the generation data
        if not combined_gen_data:
            combined_gen_data = gen_data
        else:
            combined_gen_data = combine_list_of_dicts(combined_gen_data, gen_data)

        if not combined_coverage_data:
            combined_coverage_data = coverage_data
        else:
            combined_coverage_data = combine_lists(combined_coverage_data, coverage_data)

    combined_ci_data = load_jsonl(ci_path)

    assert len(combined_gen_data) == len(combined_coverage_data) == len(combined_ci_data)
    
    # Update all stats dict
    no_empty_gen_data = []
    no_empty_coverage_data = []
    no_empty_cis = []
    omitted = 0
    for i, c in enumerate(combined_coverage_data):
        if len(c) > 0:
            no_empty_gen_data.append(combined_gen_data[i])
            no_empty_coverage_data.append(combined_coverage_data[i])
            no_empty_cis.append(combined_ci_data[i])
        else:
            omitted += 1
    if omitted > 0:
        print(f"omitted something")

    return no_empty_gen_data, no_empty_coverage_data, no_empty_cis, ci_path

def split_text_and_newlines(text):
    texts = []
    newline_counts = []
    parts = text.split("\n")
    current_text = parts[0]
    newline_count = 0

    texts.append(current_text.strip())
    
    for part in parts[1:]:
        newline_count += 1
        
        if part.strip() != "":
            texts.append(part.strip())
            newline_counts.append(newline_count)
            newline_count = 0

    return texts, newline_counts

def extract_chunk_words(text, start_word, num_words):
    # TODO fill
    pass

def extract_chunk_sentence(text, start_sentence, num_sentences):
    # Split the text by new lines and remember the spacing
    split_texts, newline_counts = split_text_and_newlines(text)

    text_sentence_count = 0
    for cur_split_text in split_texts:
        text_sentences = sent_tokenize(cur_split_text)
        text_sentence_count += len(text_sentences)

    # Make it so we at least have the last sentence to generate
    num_sentences = min(text_sentence_count - start_sentence - 1, num_sentences)
    num_sentence_copy = num_sentences

    if text_sentence_count == 1:
        try:
            prompt_text = text_sentences[start_sentence]
        except:
            # embed()
            print("Something wrong")
            return None
          
    cur_sentence_count, total_sents = 0, 0
    prompt_text, rest_of_text = "", ""

    for i, cur_split_text in enumerate(split_texts):
        text_sentences = sent_tokenize(cur_split_text)
        cur_split_text_length = len(text_sentences)
        assert num_sentences >= 0
        assert cur_sentence_count <= num_sentence_copy

        if cur_sentence_count < num_sentence_copy: # Need to build the prompt
          if cur_sentence_count == 0: # Edge case - starting
            if total_sents + cur_split_text_length < start_sentence + 1:
              total_sents += cur_split_text_length
              continue # Continue if we haven't yet reached the start idx
            else:
              start_idx = start_sentence - total_sents
              assert start_idx >= 0 and start_idx < cur_split_text_length
          else:
            start_idx = 0

          sents_to_add = min(num_sentences, cur_split_text_length - start_idx)
          final_idx = start_idx + sents_to_add
          assert final_idx <= cur_split_text_length

          prompt_text += " ".join(text_sentences[start_idx:final_idx])
          num_sentences -= sents_to_add

          cur_sentence_count += sents_to_add
          total_sents += cur_split_text_length

          if final_idx == sents_to_add and num_sentences > 0:
              prompt_text += newline_counts[i] * "\n"
          else:
            assert num_sentences == 0
            rest_of_text += " ".join(text_sentences[final_idx:])
            if i < len(newline_counts):
              rest_of_text += newline_counts[i] * "\n"
            else: 
              assert i == len(newline_counts) 

        else: # We're done building the prompt - build rest_of_text
          rest_of_text += " ".join(text_sentences)
          if i < len(newline_counts):
            rest_of_text += newline_counts[i] * "\n"
          else: 
            assert i == len(newline_counts) 

        assert num_sentences + cur_sentence_count == num_sentence_copy

    return prompt_text, rest_of_text

# Example usage - testing the above
# text1 = "Hello\n\n\nWorld. The world is bright \nhahaha...wohoo\nPython\n\nis\nGreat"
# text2 = "hey\n\hah\n\nhoh\n\nhehe\nhar\n\ntest"
# text3 = "Math\n\nThe eigenvector associating zero eigenvalue of is . Remember what mean? If and is collinear,"
# text4 = "The eigenvector associating zero eigenvalue of is . Remember what mean? If and is collinear."
# for text in [text1, text2, text3, text4]:
#   a,b = extract_chunk_sentence(text, 0, 4)
#   print(repr(a), repr(b))
  
#   a,b = extract_chunk_sentence(text, 2, 1)
#   print(repr(a), repr(b))

#   a,b = extract_chunk_sentence(text, 1, 1)
#   print(repr(a), repr(b))

#   a,b = extract_chunk_sentence(text, 3, 10)
#   print(repr(a), repr(b))
  
#   a,b = extract_chunk_sentence(text, 0, 10)
#   print(repr(a), repr(b))

#   a,b = extract_chunk_sentence(text, 0, 2)
#   print(repr(a), repr(b))


#   a,b = extract_chunk_sentence(text, 0, 1)
#   print(repr(a), repr(b))

#   print("\n")