#!/usr/bin/env python3
"""Phase-2 ablation launcher for the proposed IU-Xray retrieval model.

This wrapper keeps all architecture/data code identical to `train_proposed.py`
and only changes training objectives. Use it to quantify how much each proposed
algorithmic component contributes under the same split and evaluator.
"""

from __future__ import annotations

import argparse
import sys

import train_proposed


PRESETS = {
    "proposed": [],
    "strict_only": [
        "--cluster_start", "999",
        "--clinical_start", "999",
        "--clinical_weight", "0.0",
        "--no-use_clinical_schedule",
    ],
    "cluster_only": [
        "--cluster_start", "12",
        "--clinical_start", "999",
        "--clinical_weight", "0.0",
        "--no-use_clinical_schedule",
    ],
    "no_clinical_schedule": [
        "--cluster_start", "12",
        "--clinical_start", "15",
        "--clinical_weight", "0.12",
        "--no-use_clinical_schedule",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run proposed-method ablations with shared architecture and split."
    )
    parser.add_argument(
        "preset",
        choices=sorted(PRESETS),
        help=(
            "proposed=full current method; strict_only=no cluster/clinical; "
            "cluster_only=IDF-Jaccard+healthy cluster only; "
            "no_clinical_schedule=fixed clinical weight without curriculum"
        ),
    )
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    known, extra = parser.parse_known_args()

    forwarded = ["train_proposed.py", *PRESETS[known.preset]]
    if known.out_dir:
        forwarded += ["--out_dir", known.out_dir]
    else:
        forwarded += ["--out_dir", f"runs/phase2_{known.preset}_seed42"]
    if known.epochs is not None:
        forwarded += ["--epochs", str(known.epochs)]
    if known.batch_size is not None:
        forwarded += ["--batch_size", str(known.batch_size)]
    if known.grad_accum is not None:
        forwarded += ["--grad_accum", str(known.grad_accum)]
    if known.seed is not None:
        forwarded += ["--seed", str(known.seed)]
    forwarded += extra

    sys.argv = forwarded
    train_proposed.main()


if __name__ == "__main__":
    main()
