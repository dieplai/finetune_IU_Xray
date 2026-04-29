#!/usr/bin/env bash
set -euo pipefail

# Example usage in Colab after:
# 1. Mount Google Drive
# 2. cd to your repo root
#
# Example:
#   from google.colab import drive
#   drive.mount('/content/drive')
#   %cd /content/drive/MyDrive/finetune_IU_Xray
#   !bash benchmark_suite/run_on_colab.sh /content/drive/MyDrive/checkpoints/best.pt single_swinbert

CKPT_PATH="${1:-}"
ADAPTER="${2:-single_swinbert}"

if [[ -z "$CKPT_PATH" ]]; then
  echo "Usage: bash benchmark_suite/run_on_colab.sh <checkpoint_path> [single_swinbert|study_swinbert]"
  exit 1
fi

pip install -r requirements.txt

python benchmark_suite/build_benchmark_assets.py

python benchmark_suite/evaluate_shared.py \
  --adapter "$ADAPTER" \
  --checkpoint "$CKPT_PATH" \
  --output-dir benchmark_outputs/colab_eval

if [[ "$ADAPTER" == "single_swinbert" ]]; then
  python benchmark_suite/generate_fixed_report.py \
    --checkpoint "$CKPT_PATH" \
    --output-dir benchmark_outputs/colab_report
fi

echo "Done. See benchmark_outputs/colab_eval and benchmark_outputs/colab_report"
