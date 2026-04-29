# Phase-2 Ablation Plan

This plan turns the current strong proposed run into a clean academic experiment.
The goal is to prove which part of the method contributes to strict retrieval,
clinical-valid retrieval, and broad cluster retrieval.

## Fixed Experimental Rules

All ablations must keep these factors identical:

- Dataset: `data/v8_clean.csv`
- Images: `data/images_384`
- Split: patient-level split with seed `42`
- Architecture: SwinV2 image encoder, ClinicalBERT text encoder, study-level multi-view fusion, same projection heads
- Training budget: same epochs, batch size, gradient accumulation, optimizer, scheduler, and image resolution
- Evaluation: same benchmark manifest and same evaluator

Only the training objective is allowed to change. This keeps the comparison scientifically defensible.

## Ablation Runs

| Run | Command preset | What it tests |
|---|---|---|
| Proposed | `proposed` | Full current method: strict contrastive warmup, IDF-Jaccard/healthy-cluster soft targets, and clinical supervised contrastive curriculum. |
| Strict baseline | `strict_only` | Whether ordinary patient-level contrastive learning is enough without cluster guidance. |
| Cluster target only | `cluster_only` | Contribution of IDF-weighted Jaccard and healthy-cluster soft targets without the clinical supervised contrastive term. |
| No clinical curriculum | `no_clinical_schedule` | Whether gradually increasing the clinical term is better than using a fixed clinical weight. |

Example commands:

```bash
python train_ablation.py strict_only --epochs 100 --batch_size 32 --grad_accum 4
python train_ablation.py cluster_only --epochs 100 --batch_size 32 --grad_accum 4
python train_ablation.py no_clinical_schedule --epochs 100 --batch_size 32 --grad_accum 4
python train_ablation.py proposed --epochs 150 --batch_size 32 --grad_accum 4
```

## Expected Interpretation

- If `strict_only` has lower clinical-valid R@1 than `proposed`, the paper can argue that patient-level contrastive learning creates false negatives for same-pathology cases.
- If `cluster_only` improves clinical-valid R@1 over `strict_only`, the IDF-Jaccard/healthy-cluster soft target is contributing.
- If `proposed` improves over `cluster_only`, the clinical supervised contrastive curriculum is contributing beyond soft labels.
- If `no_clinical_schedule` is worse or less stable than `proposed`, the curriculum is justified as a training-stability component.

## Metrics To Report

Primary:

- Clinical-valid mean R@1, R@5, R@10
- Strict mean R@1, R@5, R@10

Secondary:

- Cluster mean R@1, R@5, R@10
- Clinical-all mean R@1
- MRR
- Best epoch and stability over epochs

## Minimum Paper-Quality Table

| Method | Strict R@1 | Clinical-valid R@1 | Cluster R@1 | Best epoch | Claim |
|---|---:|---:|---:|---:|---|
| Strict baseline | TBD | TBD | TBD | TBD | Baseline exact-pair contrastive. |
| Cluster target only | TBD | TBD | TBD | TBD | Effect of cluster-guided false-negative mitigation. |
| No clinical curriculum | TBD | TBD | TBD | TBD | Effect of fixed clinical loss weight. |
| Proposed | 3.8462 | 59.8639 | 69.3634 | 120 | Current full method reference. |

## Decision Rule

Keep the proposed method if it preserves strict R@1 near the current best range while clearly outperforming the strict baseline on clinical-valid R@1. If a simpler ablation reaches similar clinical-valid R@1 with better strict R@1, prefer the simpler method for the final paper because it is easier to defend.
