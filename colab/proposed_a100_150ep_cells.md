# Colab A100 Cells: Proposed Curriculum 150ep, Single Zip, No GitHub

Copy từng cell dưới đây vào Google Colab theo đúng thứ tự. Flow này chỉ cần upload **1 file zip** lên Google Drive, sau đó Colab copy zip về SSD `/content`, giải nén trên SSD và train/eval trên SSD.

Chuẩn bị trước trên Google Drive:

```text
MyDrive/finetune_IU_Xray_colab/
  iu_xray_proposed_a100_150ep_bundle.zip
```

Runtime khuyến nghị: A100 80GB.

## Cell 1: Mount Drive Và Cấu Hình

```python
from google.colab import drive
drive.mount("/content/drive")

import os, pathlib, shutil, subprocess, time, json, zipfile

DRIVE_ROOT = "/content/drive/MyDrive/finetune_IU_Xray_colab"
BUNDLE_ZIP = f"{DRIVE_ROOT}/iu_xray_proposed_a100_150ep_bundle.zip"

REPO_DIR = "/content/finetune_IU_Xray"
RUN_NAME = "proposed_a100_150ep_seed42_bs32"
OUT_DIR = f"/content/runs/{RUN_NAME}"
DRIVE_OUT = f"{DRIVE_ROOT}/runs/{RUN_NAME}"

EPOCHS = 150
BATCH_SIZE = 32
GRAD_ACCUM = 4
EVAL_EVERY = 5
SEED = 42

pathlib.Path(DRIVE_ROOT).mkdir(parents=True, exist_ok=True)
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(DRIVE_OUT).mkdir(parents=True, exist_ok=True)

print("BUNDLE_ZIP:", BUNDLE_ZIP)
print("REPO_DIR:", REPO_DIR)
print("OUT_DIR:", OUT_DIR)
print("DRIVE_OUT:", DRIVE_OUT)
print("effective_batch:", BATCH_SIZE * GRAD_ACCUM)
```

## Cell 2: Copy Bundle Sang SSD Và Giải Nén

```python
import os, pathlib, shutil, zipfile, time, subprocess

assert os.path.exists(BUNDLE_ZIP), f"Missing bundle on Drive: {BUNDLE_ZIP}"

SSD_BUNDLE = "/content/iu_xray_proposed_a100_150ep_bundle.zip"
print(f"Copying bundle to SSD: {BUNDLE_ZIP} -> {SSD_BUNDLE}")
t0 = time.time()
shutil.copy2(BUNDLE_ZIP, SSD_BUNDLE)
print("bundle_size_mb:", round(os.path.getsize(SSD_BUNDLE) / 1024**2, 2))
print("copy_sec:", round(time.time() - t0, 1))

subprocess.call(["df", "-h", "/content"])

print("\nChecking zip integrity...")
assert zipfile.is_zipfile(SSD_BUNDLE), f"Not a valid zip file: {SSD_BUNDLE}"
with zipfile.ZipFile(SSD_BUNDLE) as zf:
    bad = zf.testzip()
    if bad is not None:
        raise RuntimeError(f"Corrupt zip member: {bad}")
    names = zf.namelist()
    print("entries:", len(names))
    print("first_entries:", names[:15])

print("\nCleaning old SSD repo...")
shutil.rmtree(REPO_DIR, ignore_errors=True)
pathlib.Path(REPO_DIR).mkdir(parents=True, exist_ok=True)

print("Extracting bundle to SSD repo...")
t0 = time.time()
with zipfile.ZipFile(SSD_BUNDLE) as zf:
    zf.extractall(REPO_DIR)
print("extract_sec:", round(time.time() - t0, 1))

if not os.path.exists(f"{REPO_DIR}/train_proposed.py"):
    candidates = list(pathlib.Path(REPO_DIR).glob("*/train_proposed.py"))
    if candidates:
        nested = candidates[0].parent
        tmp = pathlib.Path("/content/_repo_unpacked")
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.move(str(nested), str(tmp))
        shutil.rmtree(REPO_DIR, ignore_errors=True)
        shutil.move(str(tmp), REPO_DIR)

required = [
    "train_proposed.py",
    "train_single_gpu.py",
    "requirements.txt",
    "tools/generate_study_retrieval_report.py",
    "data/v8_clean.csv",
    "data/images_384",
    "benchmark_suite/data/iu_xray_test_manifest_seed42.csv",
]
for rel in required:
    path = f"{REPO_DIR}/{rel}"
    assert os.path.exists(path), f"Missing after unzip: {rel}"

print("\nSSD repo ready:", REPO_DIR)
print("image_count:", len(list(pathlib.Path(f"{REPO_DIR}/data/images_384").glob("*"))))
```

## Cell 3: Install Dependencies Và Check GPU

```python
import os, subprocess, torch

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
```

## Cell 4: Preflight Code/Data

```python
import os, subprocess, pandas as pd, pathlib

os.chdir(REPO_DIR)

subprocess.check_call([
    "python", "-m", "py_compile",
    "train_single_gpu.py",
    "train_proposed.py",
    "tools/generate_study_retrieval_report.py",
])

df = pd.read_csv("data/v8_clean.csv")
print("rows:", len(df))
print("patients:", df["patient_id"].astype(str).nunique())
print("images:", len(list(pathlib.Path("data/images_384").glob("*"))))
print("projection_counts:")
print(df["projection"].fillna("").value_counts().head(10))

print("Preflight OK")
```

## Cell 5: Start Auto Backup Sang Drive

```python
import os, subprocess, pathlib, time

pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(DRIVE_OUT).mkdir(parents=True, exist_ok=True)

sync_cmd = f"""
while true; do
  mkdir -p "{DRIVE_OUT}"
  rsync -a --update "{OUT_DIR}/" "{DRIVE_OUT}/"
  date > "{DRIVE_OUT}/last_sync.txt"
  sleep 600
done
"""

sync_proc = subprocess.Popen(
    ["bash", "-lc", sync_cmd],
    stdout=open(f"{OUT_DIR}/autosync.out", "a"),
    stderr=open(f"{OUT_DIR}/autosync.err", "a"),
)

print("autosync_pid:", sync_proc.pid)
print("syncing_to:", DRIVE_OUT)
```

## Cell 6: Start Training Proposed A100 150ep

Cell này chạy foreground. Giữ cell chạy để log/lỗi hiện trực tiếp.

```python
import os, subprocess, pathlib, time, shlex

os.chdir(REPO_DIR)
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(DRIVE_OUT).mkdir(parents=True, exist_ok=True)

args = [
    "python", "-u", "train_proposed.py",
    "--csv_path", "data/v8_clean.csv",
    "--img_dir", "data/images_384",
    "--out_dir", OUT_DIR,
    "--epochs", str(EPOCHS),
    "--batch_size", str(BATCH_SIZE),
    "--grad_accum", str(GRAD_ACCUM),
    "--eval_every", str(EVAL_EVERY),
    "--freeze_ep", "5",
    "--cluster_start", "12",
    "--clinical_start", "15",
    "--use_clinical_schedule",
    "--clinical_ramp_end", "25",
    "--clinical_ramp_start_weight", "0.03",
    "--clinical_peak_weight", "0.12",
    "--clinical_hold_end", "60",
    "--clinical_mid_weight", "0.08",
    "--clinical_decay_end", "105",
    "--clinical_final_weight", "0.04",
    "--balanced_clinical_weight", "0.10",
    "--min_strict_for_balanced", "4.0",
    "--clinical_best_min_strict", "3.5",
    "--num_workers", "4",
    "--seed", str(SEED),
]
cmd = " ".join(shlex.quote(str(x)) for x in args)

with open(f"{OUT_DIR}/colab_launch_command.txt", "w", encoding="utf-8") as f:
    f.write(cmd + "\n")

print("Training starts now. Keep this cell running.")
print("RUN_NAME:", RUN_NAME)
print("OUT_DIR:", OUT_DIR)
print("log_file:", f"{OUT_DIR}/screen.log")
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

## Cell 7: Snapshot Metrics Để Gửi Cho Tôi

Chạy cell này bất cứ lúc nào. Nó không stop training nếu Cell 6 vẫn chạy riêng.

```python
import json, pathlib, os, time

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
        print(text[-6000:])
```

## Cell 8: Sau Khi Train Xong, Generate Qualitative Reports

```python
import os, subprocess, pathlib, json

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
    r = json.load(open(f"{report_dir}/study_retrieval_report.json", encoding="utf-8"))
    print("\n==", ckpt_name, "==")
    print("strict:", r["metrics"]["strict_mean_r1"])
    print("clinical_valid:", r["metrics"]["clinical_valid_mean_r1"])
    print("cluster:", r["metrics"]["cluster_mean_r1"])

subprocess.check_call(["rsync", "-a", "--update", f"{OUT_DIR}/", f"{DRIVE_OUT}/"])
print("Reports synced to:", DRIVE_OUT)
```

## Cell 9: Final Backup

```python
import subprocess, pathlib, os

pathlib.Path(DRIVE_OUT).mkdir(parents=True, exist_ok=True)
subprocess.check_call(["rsync", "-a", "--update", f"{OUT_DIR}/", f"{DRIVE_OUT}/"])
print("Final backup done:", DRIVE_OUT)
subprocess.call(["find", DRIVE_OUT, "-maxdepth", "2", "-type", "f", "-printf", "%p %kKB\n"])
```

## Nếu Muốn Thử Batch 36

Batch 36 có thể tận dụng thêm VRAM nhưng không chắc tăng chất lượng với data nhỏ. Chỉ nên thử smoke test 5 epoch trước:

```python
RUN_NAME = "proposed_a100_smoke5_seed42_bs36"
OUT_DIR = f"/content/runs/{RUN_NAME}"
DRIVE_OUT = f"{DRIVE_ROOT}/runs/{RUN_NAME}"
EPOCHS = 5
BATCH_SIZE = 36
GRAD_ACCUM = 4
print("Smoke test:", RUN_NAME, "effective_batch:", BATCH_SIZE * GRAD_ACCUM)
```

Nếu smoke batch 36 qua được epoch 5 với peak dưới khoảng 76GB, có thể chạy chính `EPOCHS=150`. Nếu không, giữ batch 32.
