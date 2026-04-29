import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from transformers import AutoTokenizer

from benchmark_suite.evaluate_shared import (
    DEFAULT_MANIFEST,
    load_single_model,
    build_single_loader,
)
from tools.generate_retrieval_report import (
    encode_test_set,
    html_for_report,
    write_csv,
)
import train_single_gpu as single


DEFAULT_FIXED_QUERIES = ROOT / "benchmark_suite" / "data" / "iu_xray_fixed_qualitative_queries_seed42.csv"


def main():
    parser = argparse.ArgumentParser(description="Generate a fixed-query qualitative report for a single-image checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--img-dir", default=str(ROOT / "data" / "images_384"))
    parser.add_argument("--manifest-csv", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--fixed-queries-csv", default=str(DEFAULT_FIXED_QUERIES))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", default="benchmark_report")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(single.TEXT_MODEL)
    df_test, loader = build_single_loader(args.manifest_csv, args.img_dir, args.num_workers)
    model, meta, missing, unexpected = load_single_model(args.checkpoint, device)
    embs = encode_test_set(model, loader, tokenizer, device)

    sim_i2t = embs["img_emb"] @ embs["txt_emb"].T
    sim_t2i = embs["txt_emb"] @ embs["img_emb"].T
    pids = embs["pids"]
    gt_strict = torch.from_numpy(pids[:, None] == pids[None, :])
    path = single.labels_to_disease_vecs(embs["labels"].float())
    overlap = (path @ path.T) > 0
    is_normal = path.sum(-1) == 0
    gt_cluster = overlap | (is_normal.unsqueeze(1) & is_normal.unsqueeze(0))

    query_df = pd.read_csv(args.fixed_queries_csv)

    def classify(sim, idx):
        top1 = int(sim[idx].argmax().item())
        if bool(gt_strict[idx, top1].item()):
            return "strict_hit"
        if bool(gt_cluster[idx, top1].item()):
            return "cluster_rescue"
        return "miss"

    image_to_text_cases = []
    text_to_image_cases = []
    for _, q in query_df.iterrows():
        idx = int(q["query_index"])
        top_idx, top_scores = sim_i2t[idx].topk(args.top_k)
        i2t_items = []
        for rank, (cand_idx, score) in enumerate(zip(top_idx.tolist(), top_scores.tolist()), start=1):
            i2t_items.append({
                "rank": rank,
                "candidate_index": cand_idx,
                "candidate_patient_id": str(df_test.iloc[cand_idx]["patient_id"]),
                "candidate_projection": str(df_test.iloc[cand_idx]["projection"]),
                "candidate_image_id": str(df_test.iloc[cand_idx]["image_id"]),
                "candidate_report": str(df_test.iloc[cand_idx]["org_caption"]),
                "score": float(score),
                "strict_hit": bool(gt_strict[idx, cand_idx].item()),
                "cluster_hit": bool(gt_cluster[idx, cand_idx].item()),
            })
        image_to_text_cases.append({
            "query_index": idx,
            "query_patient_id": str(df_test.iloc[idx]["patient_id"]),
            "query_projection": str(df_test.iloc[idx]["projection"]),
            "query_image_id": str(df_test.iloc[idx]["image_id"]),
            "query_image_path": str(Path(args.img_dir) / str(df_test.iloc[idx]["image_id"])),
            "ground_truth_report": str(df_test.iloc[idx]["org_caption"]),
            "bucket": classify(sim_i2t, idx),
            "retrievals": i2t_items,
        })

        top_idx, top_scores = sim_t2i[idx].topk(args.top_k)
        t2i_items = []
        for rank, (cand_idx, score) in enumerate(zip(top_idx.tolist(), top_scores.tolist()), start=1):
            t2i_items.append({
                "rank": rank,
                "candidate_index": cand_idx,
                "candidate_patient_id": str(df_test.iloc[cand_idx]["patient_id"]),
                "candidate_projection": str(df_test.iloc[cand_idx]["projection"]),
                "candidate_image_id": str(df_test.iloc[cand_idx]["image_id"]),
                "candidate_image_path": str(Path(args.img_dir) / str(df_test.iloc[cand_idx]["image_id"])),
                "score": float(score),
                "strict_hit": bool(gt_strict[idx, cand_idx].item()),
                "cluster_hit": bool(gt_cluster[idx, cand_idx].item()),
            })
        text_to_image_cases.append({
            "query_index": idx,
            "query_patient_id": str(df_test.iloc[idx]["patient_id"]),
            "query_projection": str(df_test.iloc[idx]["projection"]),
            "query_image_id": str(df_test.iloc[idx]["image_id"]),
            "query_image_path": str(Path(args.img_dir) / str(df_test.iloc[idx]["image_id"])),
            "query_report": str(df_test.iloc[idx]["org_caption"]),
            "bucket": classify(sim_t2i, idx),
            "retrievals": t2i_items,
        })

    report = {
        "checkpoint": args.checkpoint,
        "meta": {
            "epoch": meta.get("epoch"),
            "best_r1": meta.get("best_r1"),
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
        },
        "metrics": {
            "strict_i2t": single.recall_at_k(sim_i2t, gt_strict),
            "strict_t2i": single.recall_at_k(sim_t2i, gt_strict.T),
            "cluster_i2t": single.recall_at_k(sim_i2t, gt_cluster),
            "cluster_t2i": single.recall_at_k(sim_t2i, gt_cluster.T),
            "bucket_counts_i2t": {
                "strict_hit": sum(1 for c in image_to_text_cases if c["bucket"] == "strict_hit"),
                "cluster_rescue": sum(1 for c in image_to_text_cases if c["bucket"] == "cluster_rescue"),
                "miss": sum(1 for c in image_to_text_cases if c["bucket"] == "miss"),
            },
            "bucket_counts_t2i": {
                "strict_hit": sum(1 for c in text_to_image_cases if c["bucket"] == "strict_hit"),
                "cluster_rescue": sum(1 for c in text_to_image_cases if c["bucket"] == "cluster_rescue"),
                "miss": sum(1 for c in text_to_image_cases if c["bucket"] == "miss"),
            },
        },
        "image_to_text_cases": image_to_text_cases,
        "text_to_image_cases": text_to_image_cases,
    }

    with open(output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(output_dir / "report.html", "w", encoding="utf-8") as f:
        f.write(html_for_report(report))
    write_csv(report, str(output_dir / "report.csv"))
    print(json.dumps({
        "report_json": str(output_dir / "report.json"),
        "report_html": str(output_dir / "report.html"),
        "report_csv": str(output_dir / "report.csv"),
    }, indent=2))


if __name__ == "__main__":
    main()
