# N-Gram Coverage Attack

Implementation of the **N-Gram Coverage Attack** method for membership inference attacks introduced in our paper.

## Method Overview

The N-Gram Coverage Attack detects training data membership by analyzing how well language models can reproduce exact n-gram sequences from source documents:

1. **Text Generation**: Generate continuations from partial text prompts
2. **N-gram Matching**: Find exact n-gram overlaps between generations and source documents  
3. **Coverage Analysis**: Compute coverage statistics across different n-gram lengths
4. **Creativity Scoring**: Calculate creativity indices based on coverage patterns
5. **Membership Classification**: Use coverage/creativity scores for membership inference

## Pipeline Usage

The attack consists of four sequential steps:

### 1. Generate Text Continuations

```bash
# OpenAI model example
python -m src.attacks.ngram_coverage_attack.generate \
    --openai \
    --model gpt-3.5-turbo-0125 \
    --task bookMIA \
    --data_split train \
    --start_sentence 1 \
    --num_sentences 5 \
    --num_sequences 20 \
    --max_tokens 512 \
    --task_prompt_idx 5

# vLLM model example  
python -m src.attacks.ngram_coverage_attack.generate \
    --model meta-llama/Llama-2-7b-hf \
    --task bookMIA \
    --data_split train \
    --num_sentences 3 \
    --num_sequences 10 \
    --max_tokens 512
```

Generations will be stored in `outputs/ours/{task}/generations/{split}/`

### 2. Compute N-Gram Coverage

```bash
# With source documents (for bookMIA)
python -m src.attacks.ngram_coverage_attack.compute_ngram_coverage \
    --task bookMIA \
    --gen_data outputs/ours/bookMIA/generations/train/MODEL_FILE.jsonl \
    --output_dir outputs/ours/bookMIA/coverages/train/ \
    --min_ngram 4 \
    --source_docs swj0419/BookMIA \
    --parallel

# Without source documents (single document tasks)
python -m src.attacks.ngram_coverage_attack.compute_ngram_coverage \
    --task wikiMIA \
    --gen_data outputs/ours/wikiMIA/generations/train/MODEL_FILE.jsonl \
    --output_dir outputs/ours/wikiMIA/coverages/train/ \
    --min_ngram 4 \
    --parallel
```

Coverages will be stored in `outputs/ours/{task}/coverages/{split}/`

### 3. Calculate Creativity Indices

```bash
python -m src.attacks.ngram_coverage_attack.get_creativity_index \
    --coverage_path outputs/ours/bookMIA/coverages/train/MODEL_FILE_4.jsonl \
    --output_dir outputs/ours/bookMIA/creativities/train/ \
    --min_ngram 2 \
    --max_ngram 12
```

### 4. Evaluate Scores

```bash
python -m src.attacks.ngram_coverage_attack.run_scores \
    --outputs_file outputs/ours/bookMIA/creativities/train/MODEL_FILE_CI2-12.jsonl
```

### 5. Aggregate Results (Optional)

```bash
python -m src.analysis.aggregate_results
```

## Output Structure

The pipeline produces files with detailed metadata in filenames:
```
{model}_{params}_{timestamp}.jsonl
```

Final outputs contain:
- Original text snippets and extracted prompts
- Generated continuations 
- N-gram coverage statistics
- Creativity indices
- ROC-AUC scores and TPR@FPR metrics

## File Descriptions

- `generate.py`: Text generation using OpenAI API or vLLM models
- `compute_ngram_coverage.py`: N-gram matching and coverage analysis
- `get_creativity_index.py`: Creativity score computation from coverage stats
- `run_scores.py`: Evaluation metrics and ROC analysis
- `utils.py`: Text processing utilities and sentence extraction

## Configuration Notes

- **OpenAI models**: Uses "gpt-3.5-turbo" tokenizer for all models
- **vLLM models**: Uses "lightest" prompt formatting for minimal overhead
- **Parallel processing**: Automatically scales based on CPU count (max 4 cores for n-gram computation)
- **Rate limiting**: Automatically applied for OpenAI API requests

