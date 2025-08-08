# Data Directory

This directory contains different datasets for membership inference experiments across various domains.

## Dataset Variants

### Wikipedia-based Datasets

#### wikiMIA/
**Original WikiMIA benchmark dataset**
- Source: Hugging Face dataset `swj0419/wikiMIA` (WikiMIA_length64 subset)
- Processing: Standard train/val/test splits with no additional filtering
- Use case: Original WikiMIA benchmark for comparison

#### wikiMIA_2024_plus/
**Custom scraped Wikipedia data with quality filtering**
- Source: Custom scraped Wikipedia articles (`scraped/scraped.jsonl`)
- Size: 112 examples per class (224 total)
- Filtering: Removes surname pages, lists, and disambiguation pages using keyword filtering
- Label handling: Flips original labels due to scraping error
- Requirements: Minimum 25 words per entry
- Use case: Clean, filtered Wikipedia content for baseline experiments

#### wikiMIA_hard/
**Temporal Wikipedia changes dataset**
- Source: Custom scraped before/after versions of Wikipedia articles
- Size: 125 article pairs (250 total examples)
- Data structure: Each article contributes 2 examples (old version = member, new version = nonmember)
- Filtering: 
  - Removes technical content (ISBN, patents, mathematical notation)
  - Requires similar length versions (80%+ length ratio)
  - Requires significant content difference (50%+ change)
- Use case: Testing membership inference on temporally related content

#### wikiMIA_update/
**Updated WikiMIA-24 benchmark dataset**
- Source: Hugging Face dataset `wjfu99/WikiMIA-24` (WikiMIA_length64 subset)
- Size: 271 examples per class (542 total) - matches original WikiMIA paper
- Filtering: None (uses pre-processed benchmark data)
- Use case: Standardized benchmark for comparison with published results

### Book-based Datasets

#### bookMIA/
**Book membership inference dataset**
- Source: Hugging Face dataset `swj0419/BookMIA`
- Processing: Handles problematic null characters in text snippets
- Data structure: Book snippets with membership labels
- Use case: Testing membership inference on literary content

### Instruction Tuning Datasets

#### tulu_v1/
**Tulu v1 instruction-following dataset**
- Source: Processed Tulu v1 data (`tulu_v1_data.jsonl` and `inverse_tulu_v1_data.jsonl`)
- Size: Complex sampling with length-based binning (4x4 bins by user/response length)
- Processing: 
  - Filters extreme length outliers (5th-95th percentiles)
  - Uniform sampling across length bins
  - Dataset-balanced sampling (30 samples per bin per dataset)
- Data structure: Conversational format with user messages and assistant responses
- Use case: Testing membership inference on instruction-tuned dialogue data

### Pre-training Datasets

#### dolma_v17/
**Dolma v1.7 pre-training corpus**
- Source: 
  - Member: `emozilla/dolma-v1_7-3B` (0.01% subsample)
  - Nonmember: `allenai/paloma` dolma-v1_5 test set
- Size: 500 examples per class
- Processing: Length-based bucket sampling to ensure similar distributions
- Use case: Testing membership inference on large-scale pre-training data

#### pile_external/
**External Pile dataset from MIMIR**
- Source: `iamgroot42/mimir` pile_cc dataset (ngram_7_0.2 split)
- Processing: Uses pre-existing member/nonmember splits from prior research
- Use case: Reproducing results from "Do MIA work" paper

#### pile_ours/
**Custom Pile dataset processing**
- Source: Local Pile data (`/data/pile/test.jsonl`)
- Focus: Books3 subset filtering
- Processing: Extracts Books3 content for focused analysis
- Use case: Custom analysis of specific Pile subsets

## Data Processing Notes

All datasets use the same train/val/test split ratios (90%/5%/5%) but with swapped train/test labels in the output files - the "train.jsonl" file actually contains the test set (majority) and "test.jsonl" contains the training set (minority).

## Usage

Each dataset directory contains a `preprocess.py` script that can be run as:
```bash
python3 -m data.[dataset_name].preprocess
```