# Colab cells for R2_study_multiview_strict

Purpose: Study-level frontal/lateral fusion with strict InfoNCE only.

Upload this folder to Google Drive:

```text
MyDrive/finetune_IU_Xray_phase2/R2_study_multiview_strict/
  R2_study_multiview_strict_bundle.zip
  R2_study_multiview_strict_colab_cells.md
```

Runtime: A100 80GB recommended. Keep the training cell running; it prints logs directly.

## Cell 1 - Mount Drive and configure run

```python
from google.colab import drive
drive.mount("/content/drive")

import os, pathlib, shutil, subprocess, time, json, zipfile, shlex

DRIVE_ROOT = "/content/drive/MyDrive/finetune_IU_Xray_phase2"
PACKAGE_NAME = "R2_study_multiview_strict"
BUNDLE_ZIP = f"{DRIVE_ROOT}/{PACKAGE_NAME}/R2_study_multiview_strict_bundle.zip"

REPO_DIR = "/content/finetune_IU_Xray"
RUN_NAME = "R2_study_multiview_strict_150ep"
OUT_DIR = f"/content/runs/{RUN_NAME}"
DRIVE_OUT = f"{DRIVE_ROOT}/{PACKAGE_NAME}/runs/{RUN_NAME}"

PRESET = "study_multiview_strict"
EPOCHS = 150
BATCH_SIZE = 32
GRAD_ACCUM = 4
EVAL_EVERY = 5
SEED = 42
SPLIT_SEED = 42

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
assert os.path.exists(BUNDLE_ZIP), f"Missing bundle on Drive: {BUNDLE_ZIP}"

SSD_BUNDLE = f"/content/{PACKAGE_NAME}_bundle.zip"
print(f"Copying bundle to SSD: {BUNDLE_ZIP} -> {SSD_BUNDLE}")
t0 = time.time()
shutil.copy2(BUNDLE_ZIP, SSD_BUNDLE)
print("bundle_size_mb:", round(os.path.getsize(SSD_BUNDLE) / 1024**2, 2))
print("copy_sec:", round(time.time() - t0, 1))

subprocess.call(["df", "-h", "/content"])

print("Checking zip integrity...")
assert zipfile.is_zipfile(SSD_BUNDLE), f"Not a valid zip: {SSD_BUNDLE}"
with zipfile.ZipFile(SSD_BUNDLE) as zf:
    bad = zf.testzip()
    if bad is not None:
        raise RuntimeError(f"Corrupt zip member: {bad}")
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
    path = f"{REPO_DIR}/{rel}"
    assert os.path.exists(path), f"Missing after unzip: {rel}"

print("SSD repo ready:", REPO_DIR)
print("image_count:", len(list(pathlib.Path(f"{REPO_DIR}/data/images_384").glob("*"))))
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
  mkdir -p "{DRIVE_OUT}"
  rsync -a --update "{OUT_DIR}/" "{DRIVE_OUT}/"
  date > "{DRIVE_OUT}/last_sync.txt"
  sleep 600
done
'''

sync_proc = subprocess.Popen(
    ["bash", "-lc", sync_cmd],
    stdout=open(f"{OUT_DIR}/autosync.out", "a"),
    stderr=open(f"{OUT_DIR}/autosync.err", "a"),
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

with open(f"{OUT_DIR}/colab_launch_command.txt", "w", encoding="utf-8") as f:
    f.write(cmd + "\n")

print("Training starts now. Keep this cell running.")
print("RUN_NAME:", RUN_NAME)
print("PRESET:", PRESET)
print("OUT_DIR:", OUT_DIR)
print("command:", cmd)

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
log_path = pathlib.Path(OUT_DIR) / "screen.log"

with open(log_path, "a", encoding="utf-8", buffering=1) as log:
    log.write(f"\n\n==== START {time.ctime()} ====\n")
    log.write(cmd + "\n")
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
    log.write(f"\n==== END {time.ctime()} exit_code={train_exit_code} ====\n")

print("train_exit_code:", train_exit_code)
print("Syncing results to Drive...")
subprocess.run(["rsync", "-a", "--update", f"{OUT_DIR}/", f"{DRIVE_OUT}/"], check=False)
print("Drive sync done:", DRIVE_OUT)
if train_exit_code != 0:
    raise RuntimeError(f"Training failed with exit code {train_exit_code}")
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
    print("\n" + "=" * 100)
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
    ckpt = f"{OUT_DIR}/{ckpt_name}"
    if not os.path.exists(ckpt):
        print("SKIP missing", ckpt)
        continue
    report_dir = f"{OUT_DIR}/report_{ckpt_name.replace('.pt','')}"
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
        print("\n==", ckpt_name, "==")
        print("strict:", r["metrics"].get("strict_mean_r1"))
        print("cluster:", r["metrics"].get("cluster_mean_r1"))
        print("clinical_valid:", r["metrics"].get("clinical_valid_mean_r1"))

subprocess.check_call(["rsync", "-a", "--update", f"{OUT_DIR}/", f"{DRIVE_OUT}/"])
print("Reports synced to:", DRIVE_OUT)
```

## Cell 8 - Final backup and file list

```python
pathlib.Path(DRIVE_OUT).mkdir(parents=True, exist_ok=True)
subprocess.check_call(["rsync", "-a", "--update", f"{OUT_DIR}/", f"{DRIVE_OUT}/"])
print("Final backup done:", DRIVE_OUT)
subprocess.call(["find", DRIVE_OUT, "-maxdepth", "2", "-type", "f", "-printf", "%p %kKB\n"])
```
