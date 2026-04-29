#!/usr/bin/env python3
"""Generate qualitative retrieval reports for study-level IU-Xray checkpoints.

This is the report adapter for train_proposed.py checkpoints. It evaluates one
patient/study as one sample, so strict retrieval is patient/study identity while
clinical-valid retrieval is non-normal disease-overlap retrieval.
"""

import argparse
import base64
import io
import json
import os
import sys
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_single_gpu as base
import train_proposed as study


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


def build_test_frame(csv_path: str, seed: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["patient_id"] = df["patient_id"].astype(str)
    return base.patient_split(df, "test", seed=seed).reset_index(drop=True)


def build_loader(df_test: pd.DataFrame, img_dir: str, num_workers: int):
    ds = study.StudyIUXrayDataset(
        df_test,
        img_dir,
        base.get_val_transform(),
        train_mode=False,
    )
    loader = DataLoader(
        ds,
        batch_size=16,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=study.collate_study,
    )
    return ds, loader


def load_model(ckpt_path: str, device: torch.device):
    raw = torch.load(ckpt_path, map_location="cpu")
    state_dict, meta = unwrap_state_dict(raw)
    state_dict = strip_module_prefix(state_dict)
    model = study.StudyMedicalSwinBERT().to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return model, meta, missing, unexpected


@torch.no_grad()
def encode_dataset(model, loader, tokenizer, device: torch.device):
    model.eval()
    all_ie, all_te, all_pids, all_labels, all_caps = [], [], [], [], []
    for batch in loader:
        images = batch["images"].to(device)
        view_mask = batch["view_mask"].to(device)
        view_type_ids = batch["view_type_ids"].to(device)
        tok = tokenizer(
            batch["caption"],
            padding="max_length",
            truncation=True,
            max_length=study.TEXT_MAX_LEN,
            return_tensors="pt",
        ).to(device)
        ie, te, *_ = model(
            images,
            view_mask,
            view_type_ids,
            tok["input_ids"],
            tok["attention_mask"],
        )
        all_ie.append(ie.cpu())
        all_te.append(te.cpu())
        all_pids.extend(batch["pid"])
        all_caps.extend(batch["caption"])
        all_labels.append(batch["labels"].cpu())
    return {
        "img_emb": torch.cat(all_ie),
        "txt_emb": torch.cat(all_te),
        "pids": np.array(all_pids),
        "labels": torch.cat(all_labels).float(),
        "captions": all_caps,
    }


def representative_image_ids(sample: dict, max_images: int = 2) -> list[str]:
    ids = []
    for view in ("f", "l", "o"):
        for image_id in sample["views"].get(view, []):
            if image_id not in ids:
                ids.append(image_id)
            if len(ids) >= max_images:
                return ids
    return ids


def sample_meta(ds: study.StudyIUXrayDataset, img_dir: str) -> list[dict]:
    rows = []
    for sample in ds.samples:
        image_ids = representative_image_ids(sample)
        rows.append({
            "patient_id": str(sample["pid"]),
            "caption": str(sample["caption"]),
            "image_ids": image_ids,
            "image_paths": [str(Path(img_dir) / image_id) for image_id in image_ids],
        })
    return rows


def to_thumb_base64(path: str, size=(240, 240)) -> str:
    with Image.open(path).convert("RGB") as img:
        img.thumbnail(size)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def short_text(text: str, limit: int = 360) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def recall_valid(sim: torch.Tensor, gt_mask: torch.Tensor):
    valid = gt_mask.any(dim=1)
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return {"R@1": 0.0, "R@5": 0.0, "R@10": 0.0}, 0
    return base.recall_at_k(sim[valid], gt_mask[valid]), n_valid


def metric_block(sim_i2t, sim_t2i, gt_strict, gt_cluster, gt_clinical):
    strict_i2t = base.recall_at_k(sim_i2t, gt_strict)
    strict_t2i = base.recall_at_k(sim_t2i, gt_strict.T)
    cluster_i2t = base.recall_at_k(sim_i2t, gt_cluster)
    cluster_t2i = base.recall_at_k(sim_t2i, gt_cluster.T)
    clinical_all_i2t = base.recall_at_k(sim_i2t, gt_clinical)
    clinical_all_t2i = base.recall_at_k(sim_t2i, gt_clinical.T)
    clinical_valid_i2t, clinical_valid_i2t_n = recall_valid(sim_i2t, gt_clinical)
    clinical_valid_t2i, clinical_valid_t2i_n = recall_valid(sim_t2i, gt_clinical.T)
    return {
        "strict_i2t": strict_i2t,
        "strict_t2i": strict_t2i,
        "strict_mean_r1": (strict_i2t["R@1"] + strict_t2i["R@1"]) / 2,
        "cluster_i2t": cluster_i2t,
        "cluster_t2i": cluster_t2i,
        "cluster_mean_r1": (cluster_i2t["R@1"] + cluster_t2i["R@1"]) / 2,
        "clinical_all_i2t": clinical_all_i2t,
        "clinical_all_t2i": clinical_all_t2i,
        "clinical_all_mean_r1": (clinical_all_i2t["R@1"] + clinical_all_t2i["R@1"]) / 2,
        "clinical_valid_i2t": clinical_valid_i2t,
        "clinical_valid_t2i": clinical_valid_t2i,
        "clinical_valid_mean_r1": (clinical_valid_i2t["R@1"] + clinical_valid_t2i["R@1"]) / 2,
        "clinical_valid_counts": {
            "i2t": clinical_valid_i2t_n,
            "t2i": clinical_valid_t2i_n,
        },
    }


def classify_top1(sim, gt_strict, gt_clinical, gt_cluster, i: int) -> str:
    top1 = int(sim[i].argmax().item())
    if bool(gt_strict[i, top1].item()):
        return "strict_hit"
    if bool(gt_clinical[i, top1].item()):
        return "clinical_rescue"
    if bool(gt_cluster[i, top1].item()):
        return "cluster_rescue"
    return "miss"


def select_queries(sim, gt_strict, gt_clinical, gt_cluster, cases_per_bucket: int):
    buckets = {"strict_hit": [], "clinical_rescue": [], "cluster_rescue": [], "miss": []}
    for i in range(sim.shape[0]):
        buckets[classify_top1(sim, gt_strict, gt_clinical, gt_cluster, i)].append(i)
    selected = []
    for bucket in ("strict_hit", "clinical_rescue", "cluster_rescue", "miss"):
        selected.extend(buckets[bucket][:cases_per_bucket])
    return selected, {k: len(v) for k, v in buckets.items()}


def build_cases(meta, sim_i2t, sim_t2i, gt_strict, gt_cluster, gt_clinical, top_k, cases_per_bucket):
    i2t_selected, i2t_counts = select_queries(sim_i2t, gt_strict, gt_clinical, gt_cluster, cases_per_bucket)
    t2i_selected, t2i_counts = select_queries(sim_t2i, gt_strict.T, gt_clinical.T, gt_cluster.T, cases_per_bucket)

    def make_i2t(q_idx: int):
        top_scores, top_idx = sim_i2t[q_idx].topk(top_k)
        retrievals = []
        for rank, (cand_idx, score) in enumerate(zip(top_idx.tolist(), top_scores.tolist()), start=1):
            retrievals.append({
                "rank": rank,
                "candidate_index": cand_idx,
                "candidate_patient_id": meta[cand_idx]["patient_id"],
                "candidate_image_ids": meta[cand_idx]["image_ids"],
                "candidate_report": meta[cand_idx]["caption"],
                "score": float(score),
                "strict_hit": bool(gt_strict[q_idx, cand_idx].item()),
                "clinical_hit": bool(gt_clinical[q_idx, cand_idx].item()),
                "cluster_hit": bool(gt_cluster[q_idx, cand_idx].item()),
            })
        return {
            "direction": "image_to_text",
            "query_index": q_idx,
            "query_patient_id": meta[q_idx]["patient_id"],
            "query_image_ids": meta[q_idx]["image_ids"],
            "query_image_paths": meta[q_idx]["image_paths"],
            "ground_truth_report": meta[q_idx]["caption"],
            "bucket": classify_top1(sim_i2t, gt_strict, gt_clinical, gt_cluster, q_idx),
            "retrievals": retrievals,
        }

    def make_t2i(q_idx: int):
        top_scores, top_idx = sim_t2i[q_idx].topk(top_k)
        retrievals = []
        for rank, (cand_idx, score) in enumerate(zip(top_idx.tolist(), top_scores.tolist()), start=1):
            retrievals.append({
                "rank": rank,
                "candidate_index": cand_idx,
                "candidate_patient_id": meta[cand_idx]["patient_id"],
                "candidate_image_ids": meta[cand_idx]["image_ids"],
                "candidate_image_paths": meta[cand_idx]["image_paths"],
                "score": float(score),
                "strict_hit": bool(gt_strict[q_idx, cand_idx].item()),
                "clinical_hit": bool(gt_clinical[q_idx, cand_idx].item()),
                "cluster_hit": bool(gt_cluster[q_idx, cand_idx].item()),
            })
        return {
            "direction": "text_to_image",
            "query_index": q_idx,
            "query_patient_id": meta[q_idx]["patient_id"],
            "query_image_ids": meta[q_idx]["image_ids"],
            "query_image_paths": meta[q_idx]["image_paths"],
            "query_report": meta[q_idx]["caption"],
            "bucket": classify_top1(sim_t2i, gt_strict.T, gt_clinical.T, gt_cluster.T, q_idx),
            "retrievals": retrievals,
        }

    return {
        "bucket_counts_i2t": i2t_counts,
        "bucket_counts_t2i": t2i_counts,
        "image_to_text_cases": [make_i2t(i) for i in i2t_selected],
        "text_to_image_cases": [make_t2i(i) for i in t2i_selected],
    }


def render_images(paths: list[str]) -> str:
    tags = []
    for path in paths:
        try:
            b64 = to_thumb_base64(path)
            tags.append(f"<img src='data:image/jpeg;base64,{b64}' alt='xray'>")
        except Exception as exc:
            tags.append(f"<p>image_load_error: {escape(str(exc))}</p>")
    return "".join(tags)


def html_for_report(report: dict) -> str:
    m = report["metrics"]

    def metric_rows():
        rows = []
        for name in (
            "strict_i2t",
            "strict_t2i",
            "cluster_i2t",
            "cluster_t2i",
            "clinical_valid_i2t",
            "clinical_valid_t2i",
        ):
            item = m[name]
            rows.append(
                f"<tr><td>{escape(name)}</td><td>{item['R@1']:.2f}</td>"
                f"<td>{item['R@5']:.2f}</td><td>{item['R@10']:.2f}</td></tr>"
            )
        return "".join(rows)

    def i2t_case(case):
        rows = []
        for item in case["retrievals"]:
            rows.append(
                "<tr>"
                f"<td>{item['rank']}</td>"
                f"<td>{item['score']:.4f}</td>"
                f"<td>{escape(item['candidate_patient_id'])}</td>"
                f"<td>{'Y' if item['strict_hit'] else 'N'}</td>"
                f"<td>{'Y' if item['clinical_hit'] else 'N'}</td>"
                f"<td>{'Y' if item['cluster_hit'] else 'N'}</td>"
                f"<td>{escape(short_text(item['candidate_report']))}</td>"
                "</tr>"
            )
        return (
            "<section class='case'>"
            f"<h3>Image to Text | {escape(case['bucket'])} | pid={escape(case['query_patient_id'])}</h3>"
            "<div class='case-grid'>"
            f"<div class='image-row'>{render_images(case['query_image_paths'])}"
            f"<p><b>Query images:</b> {escape(', '.join(case['query_image_ids']))}</p></div>"
            f"<div><p><b>Ground-truth report</b><br>{escape(short_text(case['ground_truth_report'], 700))}</p></div>"
            "</div>"
            "<table><thead><tr><th>Rank</th><th>Score</th><th>PID</th><th>Strict</th>"
            "<th>Clinical</th><th>Cluster</th><th>Retrieved report</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
            "</section>"
        )

    def t2i_case(case):
        cards = []
        for item in case["retrievals"]:
            cards.append(
                "<div class='thumb-card'>"
                f"<div class='image-row'>{render_images(item['candidate_image_paths'])}</div>"
                f"<p><b>#{item['rank']}</b> score={item['score']:.4f}<br>"
                f"pid={escape(item['candidate_patient_id'])}<br>"
                f"strict={'Y' if item['strict_hit'] else 'N'} | "
                f"clinical={'Y' if item['clinical_hit'] else 'N'} | "
                f"cluster={'Y' if item['cluster_hit'] else 'N'}</p>"
                "</div>"
            )
        return (
            "<section class='case'>"
            f"<h3>Text to Image | {escape(case['bucket'])} | pid={escape(case['query_patient_id'])}</h3>"
            "<div class='case-grid'>"
            f"<div class='image-row'>{render_images(case['query_image_paths'])}"
            f"<p><b>Ground-truth images:</b> {escape(', '.join(case['query_image_ids']))}</p></div>"
            f"<div><p><b>Query report</b><br>{escape(short_text(case['query_report'], 700))}</p></div>"
            "</div>"
            f"<div class='thumb-grid'>{''.join(cards)}</div>"
            "</section>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Study-level IU-Xray Retrieval Report</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; background: #f5f7fa; color: #202733; }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    section {{ margin: 22px 0; padding: 18px; background: white; border-radius: 12px; box-shadow: 0 1px 8px rgba(0,0,0,0.08); }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    th, td {{ border: 1px solid #d9e1ea; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #edf3f9; }}
    img {{ max-width: 220px; border-radius: 8px; border: 1px solid #c9d3df; margin-right: 8px; }}
    .case-grid {{ display: grid; grid-template-columns: 480px 1fr; gap: 18px; align-items: start; }}
    .thumb-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; margin-top: 14px; }}
    .thumb-card {{ background: #f8fbff; padding: 10px; border-radius: 10px; border: 1px solid #dde6f1; }}
    .meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 10px; }}
    .pill {{ padding: 10px 12px; background: #edf3f9; border-radius: 10px; }}
    .image-row {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  </style>
</head>
<body>
  <h1>Study-level IU-Xray Retrieval Report</h1>
  <section>
    <h2>Summary</h2>
    <div class="meta">
      <div class="pill">Strict mean R@1: {m['strict_mean_r1']:.2f}</div>
      <div class="pill">Clinical-valid mean R@1: {m['clinical_valid_mean_r1']:.2f}</div>
      <div class="pill">Cluster mean R@1: {m['cluster_mean_r1']:.2f}</div>
      <div class="pill">Clinical-valid query counts: i2t={m['clinical_valid_counts']['i2t']}, t2i={m['clinical_valid_counts']['t2i']}</div>
      <div class="pill">Image-to-Text buckets: {escape(str(report['bucket_counts_i2t']))}</div>
      <div class="pill">Text-to-Image buckets: {escape(str(report['bucket_counts_t2i']))}</div>
    </div>
    <table>
      <thead><tr><th>Metric</th><th>R@1</th><th>R@5</th><th>R@10</th></tr></thead>
      <tbody>{metric_rows()}</tbody>
    </table>
  </section>
  <section>
    <h2>Image-to-Text Cases</h2>
    {''.join(i2t_case(c) for c in report['image_to_text_cases'])}
  </section>
  <section>
    <h2>Text-to-Image Cases</h2>
    {''.join(t2i_case(c) for c in report['text_to_image_cases'])}
  </section>
</body>
</html>"""


def write_csv(report: dict, output_csv: Path):
    rows = []
    for direction, cases_key in (("i2t", "image_to_text_cases"), ("t2i", "text_to_image_cases")):
        for case in report[cases_key]:
            for item in case["retrievals"]:
                rows.append({
                    "direction": direction,
                    "bucket": case["bucket"],
                    "query_index": case["query_index"],
                    "query_patient_id": case["query_patient_id"],
                    "query_image_ids": "|".join(case["query_image_ids"]),
                    "rank": item["rank"],
                    "candidate_index": item["candidate_index"],
                    "candidate_patient_id": item["candidate_patient_id"],
                    "candidate_image_ids": "|".join(item.get("candidate_image_ids", [])),
                    "score": item["score"],
                    "strict_hit": item["strict_hit"],
                    "clinical_hit": item["clinical_hit"],
                    "cluster_hit": item["cluster_hit"],
                })
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def main():
    parser = argparse.ArgumentParser(description="Generate study-level qualitative retrieval report.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv-path", default=study.CSV_PATH)
    parser.add_argument("--img-dir", default=study.IMG_DIR)
    parser.add_argument("--seed", type=int, default=study.SEED)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--cases-per-bucket", type=int, default=5)
    parser.add_argument("--output-dir", default="study_retrieval_report")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_test = build_test_frame(args.csv_path, args.seed)
    ds, loader = build_loader(df_test, args.img_dir, args.num_workers)
    tokenizer = AutoTokenizer.from_pretrained(study.TEXT_MODEL)
    model, meta, missing, unexpected = load_model(args.checkpoint, device)
    embeddings = encode_dataset(model, loader, tokenizer, device)

    sim_i2t = embeddings["img_emb"] @ embeddings["txt_emb"].T
    sim_t2i = embeddings["txt_emb"] @ embeddings["img_emb"].T
    pids = embeddings["pids"]
    gt_strict = torch.from_numpy(pids[:, None] == pids[None, :])
    disease_vecs = base.labels_to_disease_vecs(embeddings["labels"])
    gt_cluster = study.build_cluster_mask(disease_vecs)
    gt_clinical = study.build_clinical_cluster_mask(disease_vecs)

    metrics = metric_block(sim_i2t, sim_t2i, gt_strict, gt_cluster, gt_clinical)
    cases = build_cases(
        sample_meta(ds, args.img_dir),
        sim_i2t,
        sim_t2i,
        gt_strict,
        gt_cluster,
        gt_clinical,
        args.top_k,
        args.cases_per_bucket,
    )

    report = {
        "checkpoint": args.checkpoint,
        "csv_path": args.csv_path,
        "img_dir": args.img_dir,
        "seed": args.seed,
        "test_studies": int(len(ds)),
        "meta": {
            "epoch": meta.get("epoch"),
            "best_r1": meta.get("best_r1"),
            "balanced_score": meta.get("balanced_score"),
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
        },
        "metrics": metrics,
        **cases,
    }

    json_path = output_dir / "study_retrieval_report.json"
    html_path = output_dir / "study_retrieval_report.html"
    csv_path = output_dir / "study_retrieval_report.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_for_report(report))
    write_csv(report, csv_path)

    print(json.dumps({
        "json": str(json_path),
        "html": str(html_path),
        "csv": str(csv_path),
        "test_studies": report["test_studies"],
        "strict_mean_r1": metrics["strict_mean_r1"],
        "clinical_valid_mean_r1": metrics["clinical_valid_mean_r1"],
        "cluster_mean_r1": metrics["cluster_mean_r1"],
        "bucket_counts_i2t": report["bucket_counts_i2t"],
        "bucket_counts_t2i": report["bucket_counts_t2i"],
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }, indent=2))


if __name__ == "__main__":
    main()
