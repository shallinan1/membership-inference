import os
import shutil
from tqdm import tqdm

# Define the directory containing the JSONL files
directory = "/gscratch/xlab/hallisky/membership-inference/tasks/bookMIA/coverages"

# Iterate over all files in the specified directory
for filename in tqdm(os.listdir(directory)):
    print(filename)
    # Check if the file is a .jsonl file
    if filename.endswith(".jsonl"):
        # Construct the full path of the file
        filepath = os.path.join(directory, filename)

        # Modify the filename if it contains 'maxTokens'
        new_filename = filename.replace("maxTokens", "maxTok")

        # Check if 'minTok' is not in the filename
        if "minTok" not in filename:
            # Insert 'minTok0_' before 'numSeq'
            new_filename = new_filename.replace("_numSent", "_minTok0_numSeq")

        if "temp" not in filename:
            # Insert 'minTok0_' before 'numSeq'
            new_filename = new_filename.replace("_numSent", "_temp1.0_numSent")


        # Construct the full path for the new filename
        new_filepath = os.path.join(directory, new_filename)

        # Only copy and delete if the filename has changed
        if new_filepath != filepath:
            # Copy the original file to the new file
            shutil.copy(filepath, new_filepath)

            # Delete the original file after copying
            os.remove(filepath)

            # Print the change for confirmation
            print(f"Processed {filename} -> {new_filename} (Old file deleted)")
        else:
            print(f"No change for {filename}")
