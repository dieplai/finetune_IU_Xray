# Clustering-Guided Medical Image-Text Retrieval

This repository contains the thesis code for IU-Xray image-report retrieval with **clustering-guided false-negative mitigation**.

Core research question:

> If an image and a report belong to the same clinical/pathology cluster, they should not be treated as ordinary hard negatives even when they come from different patient/study IDs.

The paper-facing implementation is:

```text
train_proposed.py
```

Use `train_ablation.py` for phase-2 experiments that remove parts of the proposed objective to quantify their academic contribution.

## Repository Layout

```text
data_processing/      Data cleaning, image preprocessing, weak pathology labeling
benchmark_suite/      Fixed patient-level split and shared evaluator
tools/                Qualitative retrieval report generator
docs/paper/           Paper documentation, method notes, metrics, writing plan
colab/                Colab cell-based training workflow
tests/                Logic tests
src/                  Original baseline modules kept for reference
train_single_gpu.py   Shared baseline utilities, transforms, labels, cluster loss
train_proposed.py     Current proposed study-level method
train_ablation.py     Phase-2 ablation launcher
```

Large assets are intentionally ignored by git:

```text
data/
archive/
runs/
outputs/
*.pt, *.pth, *.ckpt
*.zip
```

## Current Main Method

- Image encoder: `microsoft/swinv2-base-patch4-window12to24-192to384-22kto1k-ft`
- Text encoder: `emilyalsentzer/Bio_ClinicalBERT`
- Input unit: one patient/study sample with up to frontal + lateral views
- Projection: image/text MLP heads into a shared 512-d embedding space
- Fusion: view-type embedding + attention pooling over available views
- Loss: strict contrastive warmup, IDF-Jaccard/healthy-cluster soft targets, clinical supervised contrastive curriculum
- Final proposed run: A100, batch 32, grad accumulation 4, 150 epochs, seed 42

Important implementation note: prototype-bank and hard-negative-mining hooks exist in legacy code, but the final proposed run disables them (`proto_start=999`, `rank_start=999`, `mine_start=999`). Do not claim final results come from those components unless a separate ablation is run.

## Metrics

Report strict and clinical metrics separately:

- `Strict R@K`: exact same patient/study retrieval.
- `Cluster R@K`: broad pathology overlap or both-normal match.
- `Clinical-valid R@K`: non-normal disease-overlap retrieval over queries that have at least one valid clinical positive.

For paper claims, use `clinical-valid` as the main false-negative mitigation metric and keep `strict` as the exact-pair quality control.

## Main Result Snapshot

Current best proposed checkpoint:

```text
run: v8_curriculum_a100_150ep_seed42_bs32_v1
checkpoint: best.pt
epoch: 120
```

Test metrics:

| Metric | Mean R@1 |
|---|---:|
| Strict | 3.8462 |
| Cluster | 69.3634 |
| Clinical-valid | 59.8639 |
| Clinical-all | 23.3422 |

See `docs/paper/03_evaluation_and_results.md` for the detailed table and v7 comparison.

## Training Proposed Method

```bash
python train_proposed.py \
  --csv_path data/v8_clean.csv \
  --img_dir data/images_384 \
  --out_dir runs/proposed_seed42 \
  --epochs 150 \
  --batch_size 32 \
  --grad_accum 4 \
  --eval_every 5 \
  --freeze_ep 5 \
  --cluster_start 12 \
  --clinical_start 15 \
  --use_clinical_schedule \
  --seed 42
```

## Build Colab Bundle

If training on Colab without cloning from GitHub, build the exact single zip
used by the proposed/v8 A100 workflow:

```bash
python tools/make_colab_bundle.py --output colab/iu_xray_proposed_a100_150ep_bundle.zip
```

Upload that zip to:

```text
MyDrive/finetune_IU_Xray_colab/
```

Then copy the cells from `colab/proposed_a100_150ep_cells.md`.

## Phase-2 Ablations

Run these after the proposed method to prove the contribution of each algorithmic component:

```bash
python train_ablation.py strict_only --epochs 100 --batch_size 32 --grad_accum 4
python train_ablation.py cluster_only --epochs 100 --batch_size 32 --grad_accum 4
python train_ablation.py no_clinical_schedule --epochs 100 --batch_size 32 --grad_accum 4
```

Presets:

- `strict_only`: disables IDF-Jaccard, healthy cluster, clinical loss, HNM, prototype.
- `cluster_only`: keeps IDF-Jaccard + healthy cluster, disables clinical supervised contrastive.
- `no_clinical_schedule`: keeps cluster and clinical loss but uses fixed clinical weight instead of curriculum.
- `proposed`: full current proposed configuration.

## Evaluation

Build fixed benchmark assets once:

```bash
python benchmark_suite/build_benchmark_assets.py
```

Evaluate a study checkpoint:

```bash
python benchmark_suite/evaluate_shared.py \
  --adapter study_swinbert \
  --checkpoint runs/proposed_seed42/best.pt \
  --output-dir benchmark_outputs/proposed_seed42
```

Generate qualitative report:

```bash
python tools/generate_study_retrieval_report.py \
  --checkpoint runs/proposed_seed42/best.pt \
  --csv_path data/v8_clean.csv \
  --img_dir data/images_384 \
  --output_dir reports/proposed_seed42
```

## Paper Docs

Detailed paper notes are in:

```text
docs/paper/
```

Start with:

- `docs/paper/01_data_pipeline.md`
- `docs/paper/02_model_and_training.md`
- `docs/paper/03_evaluation_and_results.md`
- `docs/paper/04_paper_writing_plan.md`
- `docs/paper/05_phase2_ablation_plan.md`

## Reproducibility Rules

- Always split by `patient_id`, not image row.
- Always use seed 42 for the current fixed benchmark.
- Compare models with the same evaluator and same test manifest.
- Do not report cluster-only results as strict model quality.
- Do not claim SOTA unless external baselines are reproduced under the same split and metric.
