# Colab A100 Quickstart

Use the single-zip workflow for either the current proposed A100 curriculum run
or the full phase-2 ablation suite.

## Upload To Google Drive

Upload this file:

```text
iu_xray_proposed_a100_150ep_bundle.zip
```

to:

```text
MyDrive/finetune_IU_Xray_colab/
```

Build the zip from this local repo with:

```bash
python tools/make_colab_bundle.py --output colab/iu_xray_proposed_a100_150ep_bundle.zip
```

## Run In Colab

Open a new Colab notebook, select an A100 runtime, then copy cells from:

```text
proposed_a100_150ep_cells.md
```

The cells will:

- mount Google Drive
- copy the single zip to Colab SSD
- extract code, data, images, and benchmark assets to `/content/finetune_IU_Xray`
- train `train_proposed.py` for 150 epochs
- auto-sync logs/checkpoints to Drive every 10 minutes
- generate qualitative Image-to-Text and Text-to-Image reports after training

## Phase-2 Paper Ablation Packages

For the official paper ablation suite, use:

```text
colab/phase2_packages/
```

Each run folder is self-contained and contains:

```text
<RUN_ID>_bundle.zip
<RUN_ID>_colab_cells.md
README.md
```

Upload one run folder to:

```text
MyDrive/finetune_IU_Xray_phase2/<RUN_ID>/
```

Then copy cells from `<RUN_ID>_colab_cells.md` into Colab.

The prepared 150-epoch runs are:

- `R1_single_view_strict`
- `R2_study_multiview_strict`
- `R3_cluster_guided_basic`
- `R4_cluster_idf_jaccard`
- `R5_cluster_idf_healthy`
- `R6_clinical_constant`
- `R7_full_v8_seed42`
- `R8_full_v8_seed123`

Rebuild all phase-2 packages with:

```bash
python tools/make_phase2_colab_packages.py
```

## Expected Outputs

Default A100 config:

```text
batch_size = 32
grad_accum = 4
effective_batch = 128
epochs = 150
freeze_ep = 5
cluster_start = 12
clinical_start = 15
clinical schedule = ramp/hold/decay
```

Results are copied to:

```text
MyDrive/finetune_IU_Xray_colab/runs/proposed_a100_150ep_seed42_bs32/
```

Main files:

- `screen.log`
- `progress.log`
- `history.csv`
- `best.pt`
- `best_balanced.pt`
- `best_summary.json`
- `best_balanced_summary.json`
- `test_results.json`
- `test_results_balanced.json`
- `test_results_clinical_valid.json`
- `study_retrieval_report/study_retrieval_report.html`
- `study_retrieval_report/study_retrieval_report.csv`
- `study_retrieval_report/study_retrieval_report.json`
