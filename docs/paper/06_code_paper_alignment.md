# Code-Paper Alignment Checklist

This document maps the paper claims to the current implementation so the method
description, ablation table, and code stay consistent.

## Main Implementation Files

- `train_proposed.py`: paper-facing study-level model and training loop.
- `train_single_gpu.py`: shared baseline utilities, transforms, disease-vector conversion, recall metrics, and the two-stage cluster loss.
- `train_ablation.py`: official phase-2 ablation launcher.
- `tools/generate_study_retrieval_report.py`: qualitative report generator for study-level checkpoints.
- `benchmark_suite/evaluate_shared.py`: fixed shared benchmark evaluator.

## Method Components

| Paper component | Code location | Notes |
|---|---|---|
| SwinV2 image encoder | `StudyMedicalSwinBERT.image_encoder` | Uses `microsoft/swinv2-base-patch4-window12to24-192to384-22kto1k-ft`. |
| Bio_ClinicalBERT text encoder | `StudyMedicalSwinBERT.text_encoder` | Uses `emilyalsentzer/Bio_ClinicalBERT`. |
| Projection heads | `ProjectionHead`, `img_proj`, `txt_proj` | Maps image/text features into the shared 512-d space. |
| Learnable temperature | `logit_scale` | Clamped inside `forward()` to keep the scale bounded. |
| Study-level multi-view fusion | `StudyIUXrayDataset`, `encode_study_images()` | Up to one frontal and one lateral image are encoded separately, then attention-pooled into one study image embedding. |
| Strict warmup | `MultiPositiveInfoNCE` | Active before `cluster_start`. Positive mask is same `patient_id`. |
| Cluster-guided target | `HierarchicalClusterLoss` | Active from `cluster_start`. |
| IDF-Jaccard soft target | `idf_weighted_jaccard()` | Controlled by `--use_idf_jaccard`. |
| Healthy Cluster | `HierarchicalClusterLoss` | Controlled by `--use_healthy_cluster`; both-normal pairs receive extra similarity. |
| Clinical supervised contrastive loss | `ClinicalSupervisedContrastiveLoss` | Pulls non-normal disease-overlap image-text pairs together; diagonal exact pairs are excluded in the loss. |
| Clinical curriculum | `clinical_weight_for_epoch()` | Controlled by `--use_clinical_schedule` and the ramp/hold/decay arguments. |
| Auxiliary pathology heads | `img_pathology_head`, `txt_pathology_head`, `img_normal_head`, `txt_normal_head` | Weighted by `--aux_weight`. |

## Official Phase-2 Runs

All runs use the same CSV, image folder, patient-level split seed, image size,
batch size, gradient accumulation, optimizer schedule, and evaluator. The only
changes are the listed ablation factors.

| Run | Preset | What changes |
|---|---|---|
| R1 | `single_view_strict` | `max_views=1`, no cluster loss, no clinical loss. |
| R2 | `study_multiview_strict` | `max_views=2`, no cluster loss, no clinical loss. |
| R3 | `cluster_guided_basic` | Hard disease-overlap positives, no IDF-Jaccard, no Healthy Cluster, no clinical loss. |
| R4 | `cluster_idf_jaccard` | Adds IDF-Jaccard soft target, no Healthy Cluster, no clinical loss. |
| R5 | `cluster_idf_healthy` | Adds Healthy Cluster, no clinical loss. |
| R6 | `clinical_constant` | Adds fixed clinical supervised contrastive weight, no curriculum. |
| R7 | `proposed` seed 42 | Full proposed method with clinical curriculum. |
| R8 | `proposed` seed 123 | Same as R7, different training seed, fixed `split_seed=42`. |

## Evaluation Definitions

- `Strict R@K`: positive if retrieved item has the exact same `patient_id`.
- `Cluster R@K`: positive if samples share at least one disease label or both are normal.
- `Clinical-valid R@K`: non-normal disease-overlap positives, evaluated only on queries with at least one positive in the gallery.
- In evaluation, clinical-valid includes the exact same-patient pair when that study is non-normal because it is also a disease-overlap positive.
- In the clinical training loss, the diagonal exact-patient pair is excluded because strict matching is already handled by the main contrastive objective.

## Reproducibility Rules

- Keep `split_seed=42` fixed across ablations. Changing `seed` should only affect initialization, augmentation, sampling, and dropout.
- Use checkpoint config when generating reports. The report/evaluator now read `MAX_VIEWS` and `SPLIT_SEED` from the checkpoint config by default.
- Select checkpoints by validation metrics first, then evaluate on test. Do not choose the best checkpoint by test score.
- Do not claim SOTA unless external methods are reproduced on the same split and metrics.

## Known Caveats To State Honestly

- Pathology labels are weak labels from rule-based extraction, not radiologist-confirmed labels.
- Cluster R@K can be inflated by broad normal/healthy matches; clinical-valid is the cleaner disease-overlap metric.
- Strict R@1 remains low compared with the clinical-aware metrics, so the paper should not claim the model solves exact patient-level retrieval perfectly.
- IU-Xray is small; multi-seed proposed runs are needed to show stability.
