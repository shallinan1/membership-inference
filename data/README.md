# Dataset Preprocessing

This directory contains preprocessing scripts to reproduce the datasets for membership inference experiments in the paper. Each dataset is processed and saved to `split-random-overall/` subdirectories with train/val/test splits.

## Prerequisites

The **tulu_v1** dataset requires a HuggingFace token for accessing the LIMA dataset:
- Request access at https://huggingface.co/datasets/GAIR/lima and set the `HF_TOKEN` environment variable:
  ```bash
  export HF_TOKEN=your_token_here
  ```

## Dataset Preprocessing Scripts

### Wikipedia-based

- **wikiMIA/**: Original WikiMIA benchmark dataset ([paper](https://arxiv.org/abs/2310.16789))
  
  To generate the processed dataset, run:
  ```bash
  python3 -m data.wikiMIA.preprocess
  ```
  
- **wikiMIA_hard/**: WikiMIA-2024 Hard dataset (temporal Wikipedia changes before/after article versions, introduced in our paper)
  
  To download the processed dataset from HuggingFace, run:
  ```bash
  python3 -m data.wikiMIA_hard.preprocess
  ```
  
  To instead create the dataset from scratch (scrape articles and filter data), run:
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
  
  To download, reformat, and generate the processed dataset, run:
  ```bash
  # Download raw datasets (requires HF_TOKEN - see Prerequisites above)
  ./data/tulu_v1/download_data.sh
  
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

## Output Format

All datasets are processed into a standardized format:

**Directory Structure:**
```
data/[dataset_name]/split-random-overall/
├── train.jsonl
├── test.jsonl  
└── val.jsonl
```

**JSONL Schema:**
Each line contains a JSON object with fields like:
```json
{
  "label": 0,           // 0 = non-member, 1 = member
  "snippet": "text...", // Text content (varies by dataset)
  "dataset": "name"     // Source dataset identifier
}
```

