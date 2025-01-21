# Baselines

## Loss/Logit Based

First get the generations/output probabilities. Example on train split of `pile_external` with `EleutherAI/pythia-1.4b`:

```
CUDA_VISIBLE_DEVICES=0 python3 -m code.experiments.baselines.compute_text_probs \
    --target_model EleutherAI/pythia-1.4b \
    --key_name snippet \
    --task pile_external \
    --split train;
```

## Output-only (todo)
