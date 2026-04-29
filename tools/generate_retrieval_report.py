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

import train_single_gpu as t


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
    return t.patient_split(df, "test", seed=seed).reset_index(drop=True)


def build_test_loader(df_test: pd.DataFrame, img_dir: str, num_workers: int) -> DataLoader:
    ds = t.IUXrayDataset(df_test, img_dir, t.get_val_transform())
    return DataLoader(
        ds,
        batch_size=16,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=t.collate_fn,
    )


def load_model(ckpt_path: str, device: torch.device):
    raw = torch.load(ckpt_path, map_location="cpu")
    state_dict, meta = unwrap_state_dict(raw)
    state_dict = strip_module_prefix(state_dict)
    model = t.MedicalSwinBERT().to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return model, meta, missing, unexpected


@torch.no_grad()
def encode_test_set(model, loader, tokenizer, device: torch.device):
    model.eval()
    all_ie, all_te, all_pids, all_labels, all_caps, all_proj = [], [], [], [], [], []
    for batch in loader:
        imgs = batch["image"].to(device)
        tok = tokenizer(
            batch["caption"],
            padding="max_length",
            truncation=True,
            max_length=t.TEXT_MAX_LEN,
            return_tensors="pt",
        ).to(device)
        ie, te, _ = model(imgs, tok["input_ids"], tok["attention_mask"])
        all_ie.append(ie.cpu())
        all_te.append(te.cpu())
        all_pids.extend(batch["pid"])
        all_caps.extend(batch["caption"])
        all_proj.extend(batch["proj"])
        all_labels.append(batch["labels"].cpu())

    ie = torch.cat(all_ie)
    te = torch.cat(all_te)
    pids = np.array(all_pids)
    labels = torch.cat(all_labels)
    return {
        "img_emb": ie,
        "txt_emb": te,
        "pids": pids,
        "labels": labels,
        "captions": all_caps,
        "projections": all_proj,
    }


def to_thumb_base64(path: str, size=(256, 256)) -> str:
    with Image.open(path).convert("RGB") as img:
        img.thumbnail(size)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def short_text(text: str, limit: int = 260) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def disease_overlap(labels: torch.Tensor) -> torch.Tensor:
    path = t.labels_to_disease_vecs(labels.float())
    overlap = (path @ path.T) > 0
    is_normal = path.sum(-1) == 0
    both_normal = is_normal.unsqueeze(1) & is_normal.unsqueeze(0)
    return overlap | both_normal


def row_image_path(df: pd.DataFrame, img_dir: str, idx: int) -> str:
    return os.path.join(img_dir, str(df.iloc[idx]["image_id"]))


def retrieval_rows(sim: torch.Tensor, q_idx: int, top_k: int):
    vals, idxs = sim[q_idx].topk(top_k)
    return idxs.tolist(), vals.tolist()


def build_case_records(
    df_test: pd.DataFrame,
    img_dir: str,
    embeddings: dict,
    top_k: int,
    cases_per_bucket: int,
):
    sim_i2t = embeddings["img_emb"] @ embeddings["txt_emb"].T
    sim_t2i = embeddings["txt_emb"] @ embeddings["img_emb"].T
    pids = embeddings["pids"]
    gt_strict = torch.from_numpy(pids[:, None] == pids[None, :])
    gt_cluster = disease_overlap(embeddings["labels"])

    def classify(sim, i):
        top1 = int(sim[i].argmax().item())
        strict_hit = bool(gt_strict[i, top1].item())
        cluster_hit = bool(gt_cluster[i, top1].item())
        if strict_hit:
            return "strict_hit"
        if cluster_hit:
            return "cluster_rescue"
        return "miss"

    buckets_i2t = {"strict_hit": [], "cluster_rescue": [], "miss": []}
    buckets_t2i = {"strict_hit": [], "cluster_rescue": [], "miss": []}
    for i in range(len(df_test)):
        buckets_i2t[classify(sim_i2t, i)].append(i)
        buckets_t2i[classify(sim_t2i, i)].append(i)

    def make_i2t_case(q_idx: int):
        top_idx, top_scores = retrieval_rows(sim_i2t, q_idx, top_k)
        items = []
        for rank, (cand_idx, score) in enumerate(zip(top_idx, top_scores), start=1):
            items.append({
                "rank": rank,
                "candidate_index": cand_idx,
                "candidate_patient_id": str(df_test.iloc[cand_idx]["patient_id"]),
                "candidate_projection": str(df_test.iloc[cand_idx].get("projection", "")),
                "candidate_image_id": str(df_test.iloc[cand_idx]["image_id"]),
                "candidate_report": str(df_test.iloc[cand_idx]["org_caption"]),
                "score": float(score),
                "strict_hit": bool(gt_strict[q_idx, cand_idx].item()),
                "cluster_hit": bool(gt_cluster[q_idx, cand_idx].item()),
            })
        return {
            "query_index": q_idx,
            "query_patient_id": str(df_test.iloc[q_idx]["patient_id"]),
            "query_projection": str(df_test.iloc[q_idx].get("projection", "")),
            "query_image_id": str(df_test.iloc[q_idx]["image_id"]),
            "query_image_path": row_image_path(df_test, img_dir, q_idx),
            "ground_truth_report": str(df_test.iloc[q_idx]["org_caption"]),
            "bucket": classify(sim_i2t, q_idx),
            "retrievals": items,
        }

    def make_t2i_case(q_idx: int):
        top_idx, top_scores = retrieval_rows(sim_t2i, q_idx, top_k)
        items = []
        for rank, (cand_idx, score) in enumerate(zip(top_idx, top_scores), start=1):
            items.append({
                "rank": rank,
                "candidate_index": cand_idx,
                "candidate_patient_id": str(df_test.iloc[cand_idx]["patient_id"]),
                "candidate_projection": str(df_test.iloc[cand_idx].get("projection", "")),
                "candidate_image_id": str(df_test.iloc[cand_idx]["image_id"]),
                "candidate_image_path": row_image_path(df_test, img_dir, cand_idx),
                "score": float(score),
                "strict_hit": bool(gt_strict[q_idx, cand_idx].item()),
                "cluster_hit": bool(gt_cluster[q_idx, cand_idx].item()),
            })
        return {
            "query_index": q_idx,
            "query_patient_id": str(df_test.iloc[q_idx]["patient_id"]),
            "query_projection": str(df_test.iloc[q_idx].get("projection", "")),
            "query_image_id": str(df_test.iloc[q_idx]["image_id"]),
            "query_image_path": row_image_path(df_test, img_dir, q_idx),
            "query_report": str(df_test.iloc[q_idx]["org_caption"]),
            "bucket": classify(sim_t2i, q_idx),
            "retrievals": items,
        }

    selected_i2t = []
    selected_t2i = []
    for bucket in ("strict_hit", "cluster_rescue", "miss"):
        selected_i2t.extend(buckets_i2t[bucket][:cases_per_bucket])
        selected_t2i.extend(buckets_t2i[bucket][:cases_per_bucket])

    return {
        "metrics": {
            "strict_i2t": t.recall_at_k(sim_i2t, gt_strict),
            "strict_t2i": t.recall_at_k(sim_t2i, gt_strict.T),
            "cluster_i2t": t.recall_at_k(sim_i2t, gt_cluster),
            "cluster_t2i": t.recall_at_k(sim_t2i, gt_cluster.T),
            "bucket_counts_i2t": {k: len(v) for k, v in buckets_i2t.items()},
            "bucket_counts_t2i": {k: len(v) for k, v in buckets_t2i.items()},
        },
        "image_to_text_cases": [make_i2t_case(i) for i in selected_i2t],
        "text_to_image_cases": [make_t2i_case(i) for i in selected_t2i],
    }


def html_for_report(report: dict) -> str:
    metrics = report["metrics"]

    def render_metric_table():
        rows = []
        for name in ("strict_i2t", "strict_t2i", "cluster_i2t", "cluster_t2i"):
            item = metrics[name]
            rows.append(
                f"<tr><td>{escape(name)}</td><td>{item['R@1']:.2f}</td><td>{item['R@5']:.2f}</td><td>{item['R@10']:.2f}</td></tr>"
            )
        return "\n".join(rows)

    def render_i2t_case(case):
        query_b64 = to_thumb_base64(case["query_image_path"])
        rows = []
        for item in case["retrievals"]:
            rows.append(
                "<tr>"
                f"<td>{item['rank']}</td>"
                f"<td>{item['score']:.4f}</td>"
                f"<td>{escape(item['candidate_patient_id'])}</td>"
                f"<td>{'Y' if item['strict_hit'] else ''}</td>"
                f"<td>{'Y' if item['cluster_hit'] else ''}</td>"
                f"<td>{escape(short_text(item['candidate_report']))}</td>"
                "</tr>"
            )
        return (
            "<section class='case'>"
            f"<h3>Image→Text | {escape(case['bucket'])} | query pid={escape(case['query_patient_id'])}</h3>"
            "<div class='case-grid'>"
            f"<div><img src='data:image/jpeg;base64,{query_b64}' alt='query image'><p><b>Query image:</b> {escape(case['query_image_id'])}<br><b>Projection:</b> {escape(case['query_projection'])}</p></div>"
            f"<div><p><b>Ground-truth report</b><br>{escape(short_text(case['ground_truth_report'], 600))}</p></div>"
            "</div>"
            "<table><thead><tr><th>Rank</th><th>Score</th><th>PID</th><th>Strict</th><th>Cluster</th><th>Retrieved report</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
            "</section>"
        )

    def render_t2i_case(case):
        query_b64 = to_thumb_base64(case["query_image_path"])
        cards = []
        for item in case["retrievals"]:
            cand_b64 = to_thumb_base64(item["candidate_image_path"])
            cards.append(
                "<div class='thumb-card'>"
                f"<img src='data:image/jpeg;base64,{cand_b64}' alt='candidate image'>"
                f"<p><b>#{item['rank']}</b> score={item['score']:.4f}<br>"
                f"pid={escape(item['candidate_patient_id'])}<br>"
                f"proj={escape(item['candidate_projection'])}<br>"
                f"strict={'Y' if item['strict_hit'] else 'N'} | cluster={'Y' if item['cluster_hit'] else 'N'}</p>"
                "</div>"
            )
        return (
            "<section class='case'>"
            f"<h3>Text→Image | {escape(case['bucket'])} | query pid={escape(case['query_patient_id'])}</h3>"
            "<div class='case-grid'>"
            f"<div><img src='data:image/jpeg;base64,{query_b64}' alt='query gt image'><p><b>Ground-truth image:</b> {escape(case['query_image_id'])}<br><b>Projection:</b> {escape(case['query_projection'])}</p></div>"
            f"<div><p><b>Query report</b><br>{escape(short_text(case['query_report'], 600))}</p></div>"
            "</div>"
            f"<div class='thumb-grid'>{''.join(cards)}</div>"
            "</section>"
        )

    i2t_html = "".join(render_i2t_case(c) for c in report["image_to_text_cases"])
    t2i_html = "".join(render_t2i_case(c) for c in report["text_to_image_cases"])
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>IU Xray Retrieval Report</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; background: #f5f7fa; color: #202733; }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    section {{ margin: 24px 0; padding: 18px; background: white; border-radius: 12px; box-shadow: 0 1px 8px rgba(0,0,0,0.08); }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    th, td {{ border: 1px solid #d9e1ea; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #edf3f9; }}
    img {{ max-width: 100%; border-radius: 8px; border: 1px solid #c9d3df; }}
    .case-grid {{ display: grid; grid-template-columns: 280px 1fr; gap: 18px; align-items: start; }}
    .thumb-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }}
    .thumb-card {{ background: #f8fbff; padding: 10px; border-radius: 10px; border: 1px solid #dde6f1; }}
    .meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }}
    .pill {{ padding: 10px 12px; background: #edf3f9; border-radius: 10px; }}
  </style>
</head>
<body>
  <h1>IU X-ray Retrieval Qualitative Report</h1>
  <section>
    <h2>Summary</h2>
    <div class="meta">
      <div class="pill">Image→Text bucket counts: strict_hit={metrics['bucket_counts_i2t']['strict_hit']}, cluster_rescue={metrics['bucket_counts_i2t']['cluster_rescue']}, miss={metrics['bucket_counts_i2t']['miss']}</div>
      <div class="pill">Text→Image bucket counts: strict_hit={metrics['bucket_counts_t2i']['strict_hit']}, cluster_rescue={metrics['bucket_counts_t2i']['cluster_rescue']}, miss={metrics['bucket_counts_t2i']['miss']}</div>
    </div>
    <table>
      <thead><tr><th>Metric</th><th>R@1</th><th>R@5</th><th>R@10</th></tr></thead>
      <tbody>
        {render_metric_table()}
      </tbody>
    </table>
  </section>
  <section>
    <h2>Image→Text Cases</h2>
    {i2t_html}
  </section>
  <section>
    <h2>Text→Image Cases</h2>
    {t2i_html}
  </section>
</body>
</html>"""


def write_csv(report: dict, output_csv: str):
    rows = []
    for direction_key, cases_key in (("i2t", "image_to_text_cases"), ("t2i", "text_to_image_cases")):
        for case in report[cases_key]:
            for item in case["retrievals"]:
                rows.append({
                    "direction": direction_key,
                    "bucket": case["bucket"],
                    "query_index": case["query_index"],
                    "query_patient_id": case["query_patient_id"],
                    "query_image_id": case["query_image_id"],
                    "rank": item["rank"],
                    "candidate_index": item["candidate_index"],
                    "candidate_patient_id": item["candidate_patient_id"],
                    "candidate_image_id": item.get("candidate_image_id", ""),
                    "score": item["score"],
                    "strict_hit": item["strict_hit"],
                    "cluster_hit": item["cluster_hit"],
                    "candidate_projection": item["candidate_projection"],
                })
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def main():
    parser = argparse.ArgumentParser(description="Generate qualitative retrieval report for a checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv-path", default=t.CSV_PATH)
    parser.add_argument("--img-dir", default=t.IMG_DIR)
    parser.add_argument("--seed", type=int, default=t.SEED)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--cases-per-bucket", type=int, default=3)
    parser.add_argument("--output-dir", default="retrieval_report")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(t.TEXT_MODEL)
    df_test = build_test_frame(args.csv_path, args.seed)
    test_loader = build_test_loader(df_test, args.img_dir, args.num_workers)
    model, meta, missing, unexpected = load_model(args.checkpoint, device)

    report = {
        "checkpoint": args.checkpoint,
        "meta": {
            "epoch": meta.get("epoch"),
            "best_r1": meta.get("best_r1"),
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
        },
    }
    report.update(
        build_case_records(
            df_test=df_test,
            img_dir=args.img_dir,
            embeddings=encode_test_set(model, test_loader, tokenizer, device),
            top_k=args.top_k,
            cases_per_bucket=args.cases_per_bucket,
        )
    )

    json_path = os.path.join(args.output_dir, "retrieval_report.json")
    html_path = os.path.join(args.output_dir, "retrieval_report.html")
    csv_path = os.path.join(args.output_dir, "retrieval_report.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_for_report(report))
    write_csv(report, csv_path)
    print(json.dumps({
        "json": json_path,
        "html": html_path,
        "csv": csv_path,
        "checkpoint": args.checkpoint,
        "test_size": int(len(df_test)),
        "strict_i2t_r1": report["metrics"]["strict_i2t"]["R@1"],
        "strict_t2i_r1": report["metrics"]["strict_t2i"]["R@1"],
        "cluster_i2t_r1": report["metrics"]["cluster_i2t"]["R@1"],
        "cluster_t2i_r1": report["metrics"]["cluster_t2i"]["R@1"],
    }, indent=2))


if __name__ == "__main__":
    main()
