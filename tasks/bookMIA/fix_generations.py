import os
import json
import re
from nltk.tokenize import sent_tokenize

# Regex pattern to extract numSent and startSent from the filename
filename_pattern = r"numSent(\d+)_startSent(\d+)"

# Path to the directory containing .jsonl files
directory = "tasks/bookMIA/generations"

# Iterate over each file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".jsonl"):
        filepath = os.path.join(directory, filename)
        
        # Extract numSent and startSent from the filename using regex
        match = re.search(filename_pattern, filename)
        if match:
            num_sentences = int(match.group(1))  # Extract and convert numSent to int
            start_sentence = int(match.group(2))  # Extract and convert startSent to int
        else:
            print(f"Filename {filename} does not match the expected pattern.")
            continue

        # Load the list of dictionaries from the JSONL file
        with open(filepath, 'r') as file:
            lines = file.readlines()
            data = [json.loads(line) for line in lines]

        # Process each dictionary in the data
        for entry in data:
            snippet = entry.get('snippet')
            
            # Tokenize the snippet into sentences
            snippet_sentences = sent_tokenize(snippet)
            
            # Remove the prompt sentences from the snippet
            snippet_no_prompt = " ".join(snippet_sentences[start_sentence + num_sentences:])
            
            # Add the new field to the entry
            entry['snippet_no_prompt'] = snippet_no_prompt

        # Write the updated data back to the JSONL file (or overwrite the original one)
        with open(filepath, 'w') as file:
            for entry in data:
                file.write(json.dumps(entry) + '\n')

"""
python3 -m tasks.bookMIA.fix_generations
"""