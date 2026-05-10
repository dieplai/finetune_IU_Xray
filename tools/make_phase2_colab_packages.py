#!/usr/bin/env python3
"""Build Colab-ready packages for the phase-2 IU-Xray ablation suite.

Each run folder contains:
  - a full zip bundle with code, cleaned CSV, resized images, and benchmark data
  - a Markdown file with copy/paste Colab cells for that exact run
  - a small README explaining the run objective

The full bundle is identical across run folders; the run-specific command lives
in the Colab cells so every folder can still be uploaded and executed alone.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "colab" / "phase2_packages"

RUNS = [
    {
        "id": "R1_single_view_strict",
        "preset": "single_view_strict",
        "seed": 42,
        "split_seed": 42,
        "purpose": "Single-view strict InfoNCE baseline. Tests the weakest no-cluster setting.",
    },
    {
        "id": "R2_study_multiview_strict",
        "preset": "study_multiview_strict",
        "seed": 42,
        "split_seed": 42,
        "purpose": "Study-level frontal/lateral fusion with strict InfoNCE only.",
    },
    {
        "id": "R3_cluster_guided_basic",
        "preset": "cluster_guided_basic",
        "seed": 42,
        "split_seed": 42,
        "purpose": "Hard disease-overlap cluster positives, no IDF-Jaccard, no Healthy Cluster.",
    },
    {
        "id": "R4_cluster_idf_jaccard",
        "preset": "cluster_idf_jaccard",
        "seed": 42,
        "split_seed": 42,
        "purpose": "Adds IDF-weighted Jaccard soft disease similarity, without Healthy Cluster.",
    },
    {
        "id": "R5_cluster_idf_healthy",
        "preset": "cluster_idf_healthy",
        "seed": 42,
        "split_seed": 42,
        "purpose": "Adds Healthy Cluster to IDF-Jaccard cluster-guided learning.",
    },
    {
        "id": "R6_clinical_constant",
        "preset": "clinical_constant",
        "seed": 42,
        "split_seed": 42,
        "purpose": "Full cluster objective plus fixed clinical loss weight, no curriculum.",
    },
    {
        "id": "R7_full_v8_seed42",
        "preset": "proposed",
        "seed": 42,
        "split_seed": 42,
        "purpose": "Full proposed v8 method with clinical curriculum, seed 42.",
    },
    {
        "id": "R8_full_v8_seed123",
        "preset": "proposed",
        "seed": 123,
        "split_seed": 42,
        "purpose": "Full proposed v8 method with clinical curriculum, seed 123, fixed split seed 42.",
    },
]

INCLUDE_FILES = [
    ".gitignore",
    "README.md",
    "requirements.txt",
    "train.py",
    "train_single_gpu.py",
    "train_proposed.py",
    "train_ablation.py",
    "data/v8_clean.csv",
    "data/iu_xray_dataset_metadata.csv",
]

INCLUDE_DIRS = [
    "src",
    "tests",
    "tools",
    "data_processing",
    "benchmark_suite",
    "docs/paper",
    "data/images_384",
]

EXCLUDE_PARTS = {
    ".git",
    ".venv",
    ".vscode",
    ".claude",
    "__pycache__",
    ".pytest_cache",
    "archive",
    "server_results",
    "runs",
    "outputs",
    "LVTN_2026",
    "data_kaggle",
    "phase2_packages",
}

EXCLUDE_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".pyc",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".exe",
    ".log",
    ".tmp",
    ".aux",
    ".out",
    ".toc",
    ".fls",
    ".fdb_latexmk",
    ".synctex.gz",
}


def should_skip(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in EXCLUDE_PARTS for part in rel.parts):
        return True
    return any(path.name.endswith(suffix) for suffix in EXCLUDE_SUFFIXES)


def iter_bundle_paths() -> list[Path]:
    paths: list[Path] = []
    for rel in INCLUDE_FILES:
        path = ROOT / rel
        if path.exists() and not should_skip(path):
            paths.append(path)
    for rel in INCLUDE_DIRS:
        root = ROOT / rel
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and not should_skip(path):
                paths.append(path)
    return sorted(set(paths), key=lambda p: p.relative_to(ROOT).as_posix())


def verify_required(paths: set[Path]) -> None:
    required = [
        ROOT / "train_proposed.py",
        ROOT / "train_ablation.py",
        ROOT / "train_single_gpu.py",
        ROOT / "requirements.txt",
        ROOT / "tools" / "generate_study_retrieval_report.py",
        ROOT / "data" / "v8_clean.csv",
        ROOT / "benchmark_suite" / "data" / "iu_xray_test_manifest_seed42.csv",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if path not in paths]
    image_dir = ROOT / "data" / "images_384"
    if not image_dir.exists() or not any(image_dir.iterdir()):
        missing.append("data/images_384/*")
    if missing:
        raise SystemExit("Missing required bundle inputs:\n- " + "\n- ".join(missing))


def build_shared_bundle(output_path: Path) -> None:
    paths = iter_bundle_paths()
    verify_required(set(paths))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED, compresslevel=1) as zf:
        for path in paths:
            zf.write(path, path.relative_to(ROOT).as_posix())
    print(f"shared_bundle={output_path}")
    print(f"files={len(paths)}")
    print(f"size_mb={output_path.stat().st_size / 1024**2:.2f}")


def colab_cells(run: dict) -> str:
    run_id = run["id"]
    preset = run["preset"]
    seed = run["seed"]
    split_seed = run["split_seed"]
    bundle_name = f"{run_id}_bundle.zip"
    return f"""# Colab cells for {run_id}

Purpose: {run["purpose"]}

Upload this folder to Google Drive:

```text
MyDrive/finetune_IU_Xray_phase2/{run_id}/
  {bundle_name}
  {run_id}_colab_cells.md
```

Runtime: A100 80GB recommended. Keep the training cell running; it prints logs directly.

## Cell 1 - Mount Drive and configure run

```python
from google.colab import drive
drive.mount("/content/drive")

import os, pathlib, shutil, subprocess, time, json, zipfile, shlex

DRIVE_ROOT = "/content/drive/MyDrive/finetune_IU_Xray_phase2"
PACKAGE_NAME = "{run_id}"
BUNDLE_ZIP = f"{{DRIVE_ROOT}}/{{PACKAGE_NAME}}/{bundle_name}"

REPO_DIR = "/content/finetune_IU_Xray"
RUN_NAME = "{run_id}_150ep"
OUT_DIR = f"/content/runs/{{RUN_NAME}}"
DRIVE_OUT = f"{{DRIVE_ROOT}}/{{PACKAGE_NAME}}/runs/{{RUN_NAME}}"

PRESET = "{preset}"
EPOCHS = 150
BATCH_SIZE = 32
GRAD_ACCUM = 4
EVAL_EVERY = 5
SEED = {seed}
SPLIT_SEED = {split_seed}

pathlib.Path(DRIVE_OUT).mkdir(parents=True, exist_ok=True)
print("package:", PACKAGE_NAME)
print("preset:", PRESET)
print("bundle:", BUNDLE_ZIP)
print("out_dir:", OUT_DIR)
print("drive_out:", DRIVE_OUT)
print("seed:", SEED, "split_seed:", SPLIT_SEED)
print("effective_batch:", BATCH_SIZE * GRAD_ACCUM)
```

## Cell 2 - Copy zip to Colab SSD and extract

```python
assert os.path.exists(BUNDLE_ZIP), f"Missing bundle on Drive: {{BUNDLE_ZIP}}"

SSD_BUNDLE = f"/content/{{PACKAGE_NAME}}_bundle.zip"
print(f"Copying bundle to SSD: {{BUNDLE_ZIP}} -> {{SSD_BUNDLE}}")
t0 = time.time()
shutil.copy2(BUNDLE_ZIP, SSD_BUNDLE)
print("bundle_size_mb:", round(os.path.getsize(SSD_BUNDLE) / 1024**2, 2))
print("copy_sec:", round(time.time() - t0, 1))

subprocess.call(["df", "-h", "/content"])

print("Checking zip integrity...")
assert zipfile.is_zipfile(SSD_BUNDLE), f"Not a valid zip: {{SSD_BUNDLE}}"
with zipfile.ZipFile(SSD_BUNDLE) as zf:
    bad = zf.testzip()
    if bad is not None:
        raise RuntimeError(f"Corrupt zip member: {{bad}}")
    print("entries:", len(zf.namelist()))

print("Cleaning old SSD repo and extracting...")
shutil.rmtree(REPO_DIR, ignore_errors=True)
pathlib.Path(REPO_DIR).mkdir(parents=True, exist_ok=True)
t0 = time.time()
with zipfile.ZipFile(SSD_BUNDLE) as zf:
    zf.extractall(REPO_DIR)
print("extract_sec:", round(time.time() - t0, 1))

required = [
    "train_proposed.py",
    "train_ablation.py",
    "train_single_gpu.py",
    "requirements.txt",
    "tools/generate_study_retrieval_report.py",
    "data/v8_clean.csv",
    "data/images_384",
]
for rel in required:
    path = f"{{REPO_DIR}}/{{rel}}"
    assert os.path.exists(path), f"Missing after unzip: {{rel}}"

print("SSD repo ready:", REPO_DIR)
print("image_count:", len(list(pathlib.Path(f"{{REPO_DIR}}/data/images_384").glob("*"))))
```

## Cell 3 - Install dependencies and preflight

```python
import pandas as pd, torch

os.chdir(REPO_DIR)
subprocess.call(["bash", "-lc", "command -v rsync >/dev/null || (apt-get update && apt-get install -y rsync)"])
subprocess.check_call(["python", "-m", "pip", "install", "-q", "-r", "requirements.txt"])
subprocess.call(["nvidia-smi"])

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("gpu:", props.name)
    print("vram_gb:", round(props.total_memory / 1024**3, 2))
    assert props.total_memory / 1024**3 >= 35, "Runtime VRAM is too small for this config"

subprocess.check_call([
    "python", "-m", "py_compile",
    "train_single_gpu.py",
    "train_proposed.py",
    "train_ablation.py",
    "tools/generate_study_retrieval_report.py",
])

df = pd.read_csv("data/v8_clean.csv")
print("rows:", len(df))
print("patients:", df["patient_id"].astype(str).nunique())
print("projection_counts:")
print(df["projection"].fillna("").value_counts().head(10))
print("Preflight OK")
```

## Cell 4 - Start autosync to Drive

```python
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(DRIVE_OUT).mkdir(parents=True, exist_ok=True)

sync_cmd = f'''
while true; do
  mkdir -p "{{DRIVE_OUT}}"
  rsync -a --update "{{OUT_DIR}}/" "{{DRIVE_OUT}}/"
  date > "{{DRIVE_OUT}}/last_sync.txt"
  sleep 600
done
'''

sync_proc = subprocess.Popen(
    ["bash", "-lc", sync_cmd],
    stdout=open(f"{{OUT_DIR}}/autosync.out", "a"),
    stderr=open(f"{{OUT_DIR}}/autosync.err", "a"),
)
print("autosync_pid:", sync_proc.pid)
print("syncing_to:", DRIVE_OUT)
```

## Cell 5 - Train 150 epochs in foreground

```python
os.chdir(REPO_DIR)
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

args = [
    "python", "-u", "train_ablation.py", PRESET,
    "--csv_path", "data/v8_clean.csv",
    "--img_dir", "data/images_384",
    "--out_dir", OUT_DIR,
    "--epochs", str(EPOCHS),
    "--batch_size", str(BATCH_SIZE),
    "--grad_accum", str(GRAD_ACCUM),
    "--eval_every", str(EVAL_EVERY),
    "--freeze_ep", "5",
    "--num_workers", "4",
    "--seed", str(SEED),
    "--split_seed", str(SPLIT_SEED),
    "--balanced_clinical_weight", "0.10",
    "--min_strict_for_balanced", "4.0",
    "--clinical_best_min_strict", "3.5",
    "--clinical_ramp_end", "25",
    "--clinical_ramp_start_weight", "0.03",
    "--clinical_peak_weight", "0.12",
    "--clinical_hold_end", "60",
    "--clinical_mid_weight", "0.08",
    "--clinical_decay_end", "105",
    "--clinical_final_weight", "0.04",
]
cmd = " ".join(shlex.quote(str(x)) for x in args)

with open(f"{{OUT_DIR}}/colab_launch_command.txt", "w", encoding="utf-8") as f:
    f.write(cmd + "\\n")

print("Training starts now. Keep this cell running.")
print("RUN_NAME:", RUN_NAME)
print("PRESET:", PRESET)
print("OUT_DIR:", OUT_DIR)
print("command:", cmd)

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
log_path = pathlib.Path(OUT_DIR) / "screen.log"

with open(log_path, "a", encoding="utf-8", buffering=1) as log:
    log.write(f"\\n\\n==== START {{time.ctime()}} ====\\n")
    log.write(cmd + "\\n")
    proc = subprocess.Popen(
        args,
        cwd=REPO_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
        log.write(line)
    train_exit_code = proc.wait()
    log.write(f"\\n==== END {{time.ctime()}} exit_code={{train_exit_code}} ====\\n")

print("train_exit_code:", train_exit_code)
print("Syncing results to Drive...")
subprocess.run(["rsync", "-a", "--update", f"{{OUT_DIR}}/", f"{{DRIVE_OUT}}/"], check=False)
print("Drive sync done:", DRIVE_OUT)
if train_exit_code != 0:
    raise RuntimeError(f"Training failed with exit code {{train_exit_code}}")
```

## Cell 6 - Print metrics snapshot

```python
run_dir = pathlib.Path(OUT_DIR)
files = [
    "progress.log",
    "history.csv",
    "latest_metrics.json",
    "best_summary.json",
    "best_balanced_summary.json",
    "best_clinical_valid_summary.json",
    "test_results.json",
    "test_results_balanced.json",
    "test_results_clinical_valid.json",
]

for name in files:
    path = run_dir / name
    print("\\n" + "=" * 100)
    print(name)
    print("=" * 100)
    if not path.exists():
        print("NOT_READY")
        continue
    if path.suffix == ".json":
        print(json.dumps(json.load(open(path, encoding="utf-8")), indent=2, ensure_ascii=False))
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
        print(text[-8000:])
```

## Cell 7 - Generate qualitative retrieval reports

```python
os.chdir(REPO_DIR)

for ckpt_name in ["best.pt", "best_balanced.pt", "best_clinical_valid.pt"]:
    ckpt = f"{{OUT_DIR}}/{{ckpt_name}}"
    if not os.path.exists(ckpt):
        print("SKIP missing", ckpt)
        continue
    report_dir = f"{{OUT_DIR}}/report_{{ckpt_name.replace('.pt','')}}"
    subprocess.check_call([
        "python", "tools/generate_study_retrieval_report.py",
        "--checkpoint", ckpt,
        "--csv-path", "data/v8_clean.csv",
        "--img-dir", "data/images_384",
        "--output-dir", report_dir,
        "--top-k", "10",
        "--cases-per-bucket", "5",
        "--num-workers", "4",
    ])
    report_json = pathlib.Path(report_dir) / "study_retrieval_report.json"
    if report_json.exists():
        r = json.load(open(report_json, encoding="utf-8"))
        print("\\n==", ckpt_name, "==")
        print("strict:", r["metrics"].get("strict_mean_r1"))
        print("cluster:", r["metrics"].get("cluster_mean_r1"))
        print("clinical_valid:", r["metrics"].get("clinical_valid_mean_r1"))

subprocess.check_call(["rsync", "-a", "--update", f"{{OUT_DIR}}/", f"{{DRIVE_OUT}}/"])
print("Reports synced to:", DRIVE_OUT)
```

## Cell 8 - Final backup and file list

```python
pathlib.Path(DRIVE_OUT).mkdir(parents=True, exist_ok=True)
subprocess.check_call(["rsync", "-a", "--update", f"{{OUT_DIR}}/", f"{{DRIVE_OUT}}/"])
print("Final backup done:", DRIVE_OUT)
subprocess.call(["find", DRIVE_OUT, "-maxdepth", "2", "-type", "f", "-printf", "%p %kKB\\n"])
```
"""


def write_run_folder(output_root: Path, shared_bundle: Path, run: dict) -> None:
    run_dir = output_root / run["id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    bundle_name = f"{run['id']}_bundle.zip"
    shutil.copy2(shared_bundle, run_dir / bundle_name)
    (run_dir / f"{run['id']}_colab_cells.md").write_text(colab_cells(run), encoding="utf-8")
    (run_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {run['id']}",
                "",
                run["purpose"],
                "",
                "Upload this whole folder to:",
                "",
                f"`MyDrive/finetune_IU_Xray_phase2/{run['id']}/`",
                "",
                "Then copy cells from the Markdown file into Colab and run from Cell 1 to Cell 8.",
                "",
                "Use 150 epochs for the official paper ablation table.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build phase-2 Colab packages.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--no-zips", action="store_true", help="Only regenerate README/cell files.")
    parser.add_argument("--keep-shared", action="store_true", help="Keep the temporary shared bundle copy.")
    args = parser.parse_args()

    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "epochs": 150,
        "batch_size": 32,
        "grad_accum": 4,
        "split_seed": 42,
        "runs": RUNS,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    shared_bundle = output_root / "_shared" / "iu_xray_phase2_shared_bundle.zip"
    if not args.no_zips:
        build_shared_bundle(shared_bundle)

    if not shared_bundle.exists():
        raise SystemExit(f"Missing shared bundle: {shared_bundle}")

    for run in RUNS:
        write_run_folder(output_root, shared_bundle, run)
        print(f"created_run_folder={output_root / run['id']}")

    if not args.keep_shared and shared_bundle.exists():
        shared_bundle.unlink()
        try:
            shared_bundle.parent.rmdir()
        except OSError:
            pass

    print(f"done output_root={output_root}")


if __name__ == "__main__":
    main()
