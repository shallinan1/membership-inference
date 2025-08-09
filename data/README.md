# Data Directory

This directory contains preprocessing scripts to create datasets for membership inference experiments. Each dataset is processed and saved to `split-random-overall/` subdirectories with train/val/test splits.

## Dataset Preprocessing Scripts

### Wikipedia-based

- **wikiMIA/**: Original WikiMIA benchmark dataset ([paper](https://arxiv.org/abs/2310.16789))
  
  To generate the processed dataset, run:
  ```bash
  python3 -m data.wikiMIA.preprocess
  ```
  
- **wikiMIA_hard/**: Temporal Wikipedia changes (before/after article versions)
  
  To create the dataset from scratch (scrape articles and filter data), run:
  ```bash
  python3 -m data.wikiMIA_hard.scrape_articles
  python3 -m data.wikiMIA_hard.filter_articles
  ```
  
- **wikiMIA_update/**: Updated WikiMIA-24 benchmark dataset ([paper](https://arxiv.org/abs/2408.08661))
  
  To generate the processed dataset, run:
  ```bash
  python3 -m data.wikiMIA_update.preprocess
  ```

### Other Domains

- **bookMIA/**: Book membership inference dataset ([paper](https://arxiv.org/abs/2310.16789))
  
  To generate the processed dataset, run:
  ```bash
  python3 -m data.bookMIA.preprocess
  ```
- **tulu_v1/**: Tulu v1 instruction-following dataset
  - Downloads 15+ diverse instruction datasets (Super-NaturalInstructions, Flan v2, ShareGPT, etc.)
  - Reformats each dataset into standardized conversation format with user/assistant roles
  - Creates two splits: core Tulu v1 datasets vs. complementary datasets (inverse_tulu_v1)
  - Applies length-based filtering and stratified sampling to balance distributions
  - Final preprocessing creates train/val/test splits with length histogram matching
  
  To download, reformat, and generate the processed dataset, run:
  ```bash
  # Set HuggingFace token for LIMA dataset access
  export HF_TOKEN=your_token_here
  
  # Download raw datasets
  bash data/tulu_v1/download_data.sh
  
  # Reformat datasets into standardized format
  python3 data/tulu_v1/reformat_datasets.py --raw_data_dir data/tulu_v1/raw_train/ --output_dir data/tulu_v1/processed/ --dataset tulu_v1
  
  # Create final train/val/test splits
  python3 -m data.tulu_v1.preprocess
  ```
- **dolma_v17/**: Dolma v1.7 pre-training corpus
  
  To generate the processed dataset, run:
  ```bash
  python3 -m data.dolma_v17.preprocess
  ```

- **pile_external/**: External Pile dataset from MIMIR ([paper](https://arxiv.org/abs/2402.07841))
  
  To generate the processed dataset, run:
  ```bash
  python3 -m data.pile_external.preprocess
  ```

## Usage

Each dataset has its specific preprocessing commands listed above. For most datasets, the general pattern is:
```bash
python3 -m data.[dataset_name].preprocess
```

Processed data is saved to `data/[dataset_name]/split-random-overall/` with files:
- `train.jsonl` (actually test set - majority class)
- `test.jsonl` (actually train set - minority class) 
- `val.jsonl` (validation set)