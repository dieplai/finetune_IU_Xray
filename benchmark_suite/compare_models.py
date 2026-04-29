import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_suite.evaluate_shared import evaluate_checkpoint, DEFAULT_MANIFEST
import torch


def parse_model_arg(raw: str):
    parts = raw.split("=", 2)
    if len(parts) != 3:
        raise ValueError("Each --model must be in the form name=adapter=checkpoint_path")
    return {"name": parts[0], "adapter": parts[1], "checkpoint": parts[2]}


def markdown_table(rows):
    headers = [
        "name", "adapter", "strict_mean_r1",
        "strict_i2t_r1", "strict_t2i_r1",
        "strict_i2t_r5", "strict_t2i_r5",
        "strict_i2t_r10", "strict_t2i_r10",
        "cluster_mean_r1",
    ]
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        vals = []
        for h in headers:
            v = row.get(h, "")
            if isinstance(v, float):
                vals.append(f"{v:.2f}")
            else:
                vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="Compare multiple IU-Xray checkpoints on the same benchmark manifest.")
    parser.add_argument("--model", action="append", required=True, help="Format: name=adapter=checkpoint_path")
    parser.add_argument("--img-dir", default=str(ROOT / "data" / "images_384"))
    parser.add_argument("--manifest-csv", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", default="benchmark_compare")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for raw in args.model:
        spec = parse_model_arg(raw)
        row = evaluate_checkpoint(
            adapter=spec["adapter"],
            ckpt_path=spec["checkpoint"],
            manifest_csv=args.manifest_csv,
            img_dir=args.img_dir,
            num_workers=args.num_workers,
            device=device,
        )
        row["name"] = spec["name"]
        rows.append(row)

    rows.sort(key=lambda r: r["strict_mean_r1"], reverse=True)
    with open(output_dir / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    if rows:
        with open(output_dir / "comparison.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    with open(output_dir / "comparison.md", "w", encoding="utf-8") as f:
        f.write(markdown_table(rows) + "\n")
    print(json.dumps(rows, indent=2))
    print(f"saved_dir={output_dir}")


if __name__ == "__main__":
    main()
