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

Only the explicitly listed factor is allowed to change. This keeps the comparison scientifically defensible.

## Ablation Runs

| Run | Command preset | What it tests |
|---|---|---|
| R1 | `single_view_strict` | Weakest baseline: one image/view per study, strict InfoNCE only. |
| R2 | `study_multiview_strict` | Effect of study-level frontal/lateral fusion under strict InfoNCE only. |
| R3 | `cluster_guided_basic` | Basic clustering-guided objective: hard disease-overlap positives, no IDF-Jaccard, no Healthy Cluster. |
| R4 | `cluster_idf_jaccard` | Adds IDF-weighted Jaccard soft disease similarity, still without Healthy Cluster. |
| R5 | `cluster_idf_healthy` | Adds Healthy Cluster to IDF-Jaccard cluster-guided learning. |
| R6 | `clinical_constant` | Adds clinical supervised contrastive loss with fixed weight, no curriculum. |
| R7 | `proposed` with seed 42 | Full current method: strict warmup, IDF-Jaccard, Healthy Cluster, and clinical curriculum. |
| R8 | `proposed` with seed 123 | Same full method with a different training seed and fixed split seed 42. |

Example commands:

```bash
python train_ablation.py single_view_strict --epochs 150 --batch_size 32 --grad_accum 4 --seed 42 --split_seed 42
python train_ablation.py study_multiview_strict --epochs 150 --batch_size 32 --grad_accum 4 --seed 42 --split_seed 42
python train_ablation.py cluster_guided_basic --epochs 150 --batch_size 32 --grad_accum 4 --seed 42 --split_seed 42
python train_ablation.py cluster_idf_jaccard --epochs 150 --batch_size 32 --grad_accum 4 --seed 42 --split_seed 42
python train_ablation.py cluster_idf_healthy --epochs 150 --batch_size 32 --grad_accum 4 --seed 42 --split_seed 42
python train_ablation.py clinical_constant --epochs 150 --batch_size 32 --grad_accum 4 --seed 42 --split_seed 42
python train_ablation.py proposed --epochs 150 --batch_size 32 --grad_accum 4 --seed 42 --split_seed 42
python train_ablation.py proposed --epochs 150 --batch_size 32 --grad_accum 4 --seed 123 --split_seed 42
```

## Expected Interpretation

- If R2 improves over R1, study-level multi-view fusion is justified.
- If R3 improves clinical-valid R@1 over R2, basic clustering-guided false-negative mitigation is contributing.
- If R4 improves over R3, IDF-Jaccard disease similarity is contributing.
- If R5 improves over R4, Healthy Cluster handling is contributing.
- If R7 improves over R6 or is more stable, the clinical curriculum is justified over a fixed clinical weight.
- If R8 is close to R7, the full proposed method is not just a lucky seed.

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
| R1 single-view strict | TBD | TBD | TBD | TBD | Weakest strict baseline. |
| R2 study multi-view strict | TBD | TBD | TBD | TBD | Effect of frontal/lateral fusion. |
| R3 basic cluster-guided | TBD | TBD | TBD | TBD | Effect of hard cluster positives. |
| R4 + IDF-Jaccard | TBD | TBD | TBD | TBD | Effect of soft disease similarity. |
| R5 + Healthy Cluster | TBD | TBD | TBD | TBD | Effect of normal/healthy handling. |
| R6 fixed clinical weight | TBD | TBD | TBD | TBD | Effect of clinical loss without curriculum. |
| R7 proposed seed42 | 3.8462 | 59.8639 | 69.3634 | 120 | Current full method reference. |
| R8 proposed seed123 | TBD | TBD | TBD | TBD | Stability check. |

## Decision Rule

Keep the proposed method if it preserves strict R@1 near the current best range while clearly outperforming the strict baseline on clinical-valid R@1. If a simpler ablation reaches similar clinical-valid R@1 with better strict R@1, prefer the simpler method for the final paper because it is easier to defend.
