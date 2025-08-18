    
import os
from code.utils import load_jsonl, load_json, combine_lists, combine_dicts, combine_list_of_dicts
from enum import Enum
from nltk import sent_tokenize

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