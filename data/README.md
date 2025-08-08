# Data Directory

This directory contains preprocessing scripts to create datasets for membership inference experiments. Each dataset is processed and saved to `split-random-overall/` subdirectories with train/val/test splits.

## Dataset Preprocessing Scripts

### Wikipedia-based

- **wikiMIA/**: Original WikiMIA benchmark dataset ([paper](https://arxiv.org/abs/2310.16789))
  
- **wikiMIA_hard/**: Temporal Wikipedia changes (before/after article versions)
- **wikiMIA_update/**: Updated WikiMIA-24 benchmark dataset ([paper](https://arxiv.org/abs/2408.08661))

### Other Domains

- **bookMIA/**: Book membership inference dataset
- **tulu_v1/**: Tulu v1 instruction-following dataset
- **dolma_v17/**: Dolma v1.7 pre-training corpus
- **pile_external/**: External Pile dataset from MIMIR

## Usage

Run preprocessing scripts with:
```bash
python3 -m data.[dataset_name].preprocess
```

Processed data is saved to `data/[dataset_name]/split-random-overall/` with files:
- `train.jsonl` (actually test set - majority class)
- `test.jsonl` (actually train set - minority class) 
- `val.jsonl` (validation set)