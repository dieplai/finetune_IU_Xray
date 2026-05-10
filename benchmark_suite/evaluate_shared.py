import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_single_gpu as single
import train_proposed as study


BENCHMARK_DATA_DIR = ROOT / "benchmark_suite" / "data"
DEFAULT_MANIFEST = BENCHMARK_DATA_DIR / "iu_xray_test_manifest_seed42.csv"


def unwrap_state_dict(obj):
    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"], obj
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"], obj
    if isinstance(obj, dict):
        return obj, {}
    return obj, {}


def strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(str(k).startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}


def load_single_model(ckpt_path: str, device: torch.device):
    raw = torch.load(ckpt_path, map_location="cpu")
    state_dict, meta = unwrap_state_dict(raw)
    state_dict = strip_module_prefix(state_dict)
    model = single.MedicalSwinBERT().to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return model, meta, missing, unexpected


def load_study_model(ckpt_path: str, device: torch.device):
    raw = torch.load(ckpt_path, map_location="cpu")
    state_dict, meta = unwrap_state_dict(raw)
    state_dict = strip_module_prefix(state_dict)
    model = study.StudyMedicalSwinBERT().to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return model, meta, missing, unexpected


def build_single_loader(manifest_csv: str, img_dir: str, num_workers: int):
    df = pd.read_csv(manifest_csv)
    df["patient_id"] = df["patient_id"].astype(str)
    ds = single.IUXrayDataset(df, img_dir, single.get_val_transform())
    return df, DataLoader(
        ds, batch_size=16, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=single.collate_fn,
    )


def build_study_loader(manifest_csv: str, img_dir: str, num_workers: int, max_views: int):
    df = pd.read_csv(manifest_csv)
    df["patient_id"] = df["patient_id"].astype(str)
    ds = study.StudyIUXrayDataset(
        df,
        img_dir,
        single.get_val_transform(),
        train_mode=False,
        max_views=max_views,
    )
    return df, DataLoader(
        ds, batch_size=16, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=study.collate_study,
    )


def flatten_metrics(
    si,
    st,
    ci,
    ct,
    sr1,
    cr1,
    clinical_i2t=None,
    clinical_t2i=None,
    clinical_valid_i2t=None,
    clinical_valid_t2i=None,
    clinical_all_r1=None,
    clinical_valid_r1=None,
):
    row = {
        "strict_mean_r1": sr1,
        "cluster_mean_r1": cr1,
        "strict_i2t_r1": si["R@1"],
        "strict_i2t_r5": si["R@5"],
        "strict_i2t_r10": si["R@10"],
        "strict_t2i_r1": st["R@1"],
        "strict_t2i_r5": st["R@5"],
        "strict_t2i_r10": st["R@10"],
        "cluster_i2t_r1": ci["R@1"],
        "cluster_i2t_r5": ci["R@5"],
        "cluster_i2t_r10": ci["R@10"],
        "cluster_t2i_r1": ct["R@1"],
        "cluster_t2i_r5": ct["R@5"],
        "cluster_t2i_r10": ct["R@10"],
    }
    if clinical_i2t is not None and clinical_t2i is not None:
        row.update({
            "clinical_all_mean_r1": clinical_all_r1,
            "clinical_i2t_r1": clinical_i2t["R@1"],
            "clinical_i2t_r5": clinical_i2t["R@5"],
            "clinical_i2t_r10": clinical_i2t["R@10"],
            "clinical_t2i_r1": clinical_t2i["R@1"],
            "clinical_t2i_r5": clinical_t2i["R@5"],
            "clinical_t2i_r10": clinical_t2i["R@10"],
        })
    if clinical_valid_i2t is not None and clinical_valid_t2i is not None:
        row.update({
            "clinical_valid_mean_r1": clinical_valid_r1,
            "clinical_valid_i2t_r1": clinical_valid_i2t["R@1"],
            "clinical_valid_i2t_r5": clinical_valid_i2t["R@5"],
            "clinical_valid_i2t_r10": clinical_valid_i2t["R@10"],
            "clinical_valid_t2i_r1": clinical_valid_t2i["R@1"],
            "clinical_valid_t2i_r5": clinical_valid_t2i["R@5"],
            "clinical_valid_t2i_r10": clinical_valid_t2i["R@10"],
        })
    return row


def evaluate_checkpoint(adapter: str, ckpt_path: str, manifest_csv: str, img_dir: str, num_workers: int, device: torch.device):
    clinical_i2t = clinical_t2i = None
    clinical_valid_i2t = clinical_valid_t2i = None
    clinical_all_r1 = clinical_valid_r1 = None

    if adapter == "single_swinbert":
        tokenizer = AutoTokenizer.from_pretrained(single.TEXT_MODEL)
        _, loader = build_single_loader(manifest_csv, img_dir, num_workers)
        model, meta, missing, unexpected = load_single_model(ckpt_path, device)
        si, st, ci, ct, sr1, cr1 = single.evaluate(model, loader, tokenizer, device)
    elif adapter == "study_swinbert":
        tokenizer = AutoTokenizer.from_pretrained(study.TEXT_MODEL)
        model, meta, missing, unexpected = load_study_model(ckpt_path, device)
        ckpt_config = meta.get("config", {}) if isinstance(meta, dict) else {}
        max_views = int(ckpt_config.get("MAX_VIEWS", ckpt_config.get("max_views", study.MAX_VIEWS)))
        _, loader = build_study_loader(manifest_csv, img_dir, num_workers, max_views=max_views)
        metrics = study.evaluate_study(model, loader, tokenizer, device)
        if len(metrics) == 6:
            si, st, ci, ct, sr1, cr1 = metrics
            clinical_i2t = clinical_t2i = None
            clinical_valid_i2t = clinical_valid_t2i = None
            clinical_all_r1 = clinical_valid_r1 = None
        else:
            (
                si,
                st,
                ci,
                ct,
                clinical_i2t,
                clinical_t2i,
                clinical_valid_i2t,
                clinical_valid_t2i,
                _clinical_valid_i2t_n,
                _clinical_valid_t2i_n,
                sr1,
                cr1,
                clinical_all_r1,
                clinical_valid_r1,
                _mrr,
            ) = metrics
    else:
        raise ValueError(f"Unknown adapter: {adapter}")

    row = {
        "adapter": adapter,
        "checkpoint": ckpt_path,
        "manifest_csv": manifest_csv,
        "load_status": "ok" if len(missing) == 0 else "partial",
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "meta_epoch": meta.get("epoch"),
        "meta_best_r1": meta.get("best_r1"),
    }
    row.update(flatten_metrics(
        si,
        st,
        ci,
        ct,
        sr1,
        cr1,
        clinical_i2t=clinical_i2t,
        clinical_t2i=clinical_t2i,
        clinical_valid_i2t=clinical_valid_i2t,
        clinical_valid_t2i=clinical_valid_t2i,
        clinical_all_r1=clinical_all_r1,
        clinical_valid_r1=clinical_valid_r1,
    ))
    return row


def main():
    parser = argparse.ArgumentParser(description="Shared benchmark evaluator for IU-Xray retrieval checkpoints.")
    parser.add_argument("--adapter", choices=["single_swinbert", "study_swinbert"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--img-dir", default=str(ROOT / "data" / "images_384"))
    parser.add_argument("--manifest-csv", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", default="benchmark_outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    row = evaluate_checkpoint(
        adapter=args.adapter,
        ckpt_path=args.checkpoint,
        manifest_csv=args.manifest_csv,
        img_dir=args.img_dir,
        num_workers=args.num_workers,
        device=device,
    )

    json_path = output_dir / "metrics.json"
    csv_path = output_dir / "metrics.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row, indent=2))
    print(f"saved_json={json_path}")
    print(f"saved_csv={csv_path}")


if __name__ == "__main__":
    main()
