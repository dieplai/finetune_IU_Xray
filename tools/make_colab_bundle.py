#!/usr/bin/env python3
"""Create the single Colab zip used by the proposed/v8 A100 workflow.

The zip intentionally contains code, the cleaned CSV, resized images, and the
fixed benchmark assets so Colab can train entirely from SSD after extraction.
Model checkpoints and previous run outputs are excluded.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT = ROOT / "colab" / "iu_xray_proposed_a100_150ep_bundle.zip"

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
    "colab",
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
    name = path.name
    if any(name.endswith(suffix) for suffix in EXCLUDE_SUFFIXES):
        return True
    return False


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Colab single-zip bundle.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    paths = iter_bundle_paths()
    verify_required(set(paths))

    if output.exists():
        output.unlink()

    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=6) as zf:
        for path in paths:
            arcname = path.relative_to(ROOT).as_posix()
            zf.write(path, arcname)

    size_mb = output.stat().st_size / 1024**2
    print(f"created: {output}")
    print(f"files: {len(paths)}")
    print(f"size_mb: {size_mb:.2f}")


if __name__ == "__main__":
    main()
