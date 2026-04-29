# Paper Documentation

This folder records the technical facts needed to write the thesis/paper for
clustering-guided medical image-text retrieval on IU-Xray.

## Research Objective

The project studies image-report retrieval at two levels:

- Strict retrieval: the retrieved image/report must belong to the same patient or study.
- Clinical retrieval: an image/report from another patient should not be treated as a false hard negative if it shares the same clinically meaningful pathology cluster.

Main thesis statement:

> Standard contrastive learning treats every non-matching patient as a negative pair. In medical data, this creates false negatives because different patients can share the same disease pattern. The proposed method mitigates this issue with cluster-guided soft targets and clinical supervised contrastive training.

## Files

- `01_data_pipeline.md`: data source, CSV cleaning, image processing, weak pathology labels, patient-level split.
- `02_model_and_training.md`: SwinV2 image encoder, ClinicalBERT text encoder, study-level multi-view fusion, loss functions, curriculum.
- `03_evaluation_and_results.md`: strict, cluster, clinical-valid metrics and current results.
- `04_paper_writing_plan.md`: recommended paper structure, claims, limitations, tables, and figures.
- `05_phase2_ablation_plan.md`: clean ablation plan for proving the contribution of each algorithmic component.
- `drafts/`: LaTeX/PDF drafts kept for reference.

## Current Main Checkpoint

Use this result as the current proposed-method reference:

- Historical run folder: `v8_curriculum_a100_150ep_seed42_bs32_v1`
- Checkpoint: `best.pt`
- Best epoch: `120`
- Test Strict mean R@1: `3.8462%`
- Test Cluster mean R@1: `69.3634%`
- Test Clinical-valid mean R@1: `59.8639%`
- Test Clinical-all mean R@1: `23.3422%`

## Reporting Rules

- Report `clinical-valid R@1` as the main metric for the clustering-guided contribution.
- Report `strict R@1` together with clinical metrics to show exact-pair retrieval quality is not ignored.
- Treat broad `cluster R@1` as a supporting metric because normal/healthy clusters can inflate the score.
- Do not claim strict-retrieval SOTA unless external baselines are reproduced under the same split and evaluator.
- Do not claim prototype-bank or hard-negative-mining contributions for the current main result; those hooks are disabled in the final proposed run.
