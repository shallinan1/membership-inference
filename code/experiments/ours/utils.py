    
import os
from utils import load_jsonl, load_json, combine_lists, combine_dicts, combine_list_of_dicts
from enum import Enum

class BookMIALength(Enum):
    TRAIN = 494  # Example length for training dataset
    VAL = 493   # Example length for validation dataset
    TEST = 8883    # Example length for test dataset

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
