"""
silhouette_plot.py  (v3 — Task 3: Clustering-Guided Negative Sampling)
=======================================================================
Mục tiêu Task 3:
  Nếu ảnh A và báo cáo B thuộc cùng một cụm (ví dụ: cùng nhóm bệnh
  "Cardiomegaly") sẽ KHÔNG coi chúng là mẫu âm tính của nhau, ngay cả
  khi chúng đến từ các bệnh nhân khác nhau.

Pipeline đúng:
  Image Encoder (SwinV2) + Text Encoder (ClinicalBERT)
       → Projection Head (MLP) → joint embedding space
       → K-Means Clustering trên FUSED embedding
       → Silhouette Score đánh giá chất lượng cụm
       → cluster_assignments.csv  (dùng cho negative sampling)

Thay đổi so với v2:
  FIX-8  Dùng FUSED projected embedding (image_proj + text_proj) / 2
         thay vì raw SwinV2 pooler output → đúng embedding space Task 3
  FIX-9  K-Means (k=n_clusters) thay vì label-derived IDs → unsupervised
  FIX-10 Export cluster_assignments.csv (uid, filename, label, cluster_id)
  FIX-11 UMAP 2D visualization (fallback PCA nếu umap chưa cài)

Usage:
  !pip install umap-learn -q
  !python plot/silhouette_plot.py \
      --checkpoint /kaggle/.../best_clinical_valid.pt \
      --reports_csv .../indiana_reports.csv \
      --projections_csv .../indiana_projections.csv \
      --img_dir .../images/images_normalized \
      --out_dir /kaggle/working/plots \
      --num_samples 800 --min_per_class 40 --n_clusters 4
"""

import sys, os, argparse, warnings
warnings.filterwarnings("ignore")

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent not in sys.path:
    sys.path.insert(0, parent)

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import (silhouette_score, silhouette_samples,
                              adjusted_rand_score, normalized_mutual_info_score)
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoTokenizer
from scipy.cluster.hierarchy import dendrogram, linkage

import train_proposed as tp

# ── constants ────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE   = 384
BATCH_SIZE = 16

# ── expanded keyword dict ────────────────────────────────────────────────────
MESH_KEYWORDS = {
    "No Finding": ["normal", "no indexing", "negative"],
    "Heart/Mediastinum": [
        "cardiomegaly", "cardiac shadow", "mediastinum/enlarged",
        "mediastinum/widened", "aorta/tortuous", "aorta, thoracic/tortuous",
        "pericardial effusion", "pericardium", "hilar",
        "pulmonary artery/enlarged", "pulmonary hypertension",
        "vascular", "venous congestion", "stents/coronary",
    ],
    "Lung/Parenchyma": [
        "pneumonia", "airspace disease", "consolidation",
        "atelectasis", "pulmonary atelectasis", "edema",
        "pulmonary congestion", "pulmonary edema", "opacity", "shadow",
        "interstitial", "emphysema", "bullous emphysema", "fibrosis",
        "pulmonary fibrosis", "granuloma", "granulomatous disease",
        "nodule", "mass/lung", "lung/hyperdistention", "lung/hypoinflation",
        "diaphragm/flattened", "cicatrix/lung", "density/lung",
        "density/cardiophrenic", "infiltrate", "aspiration", "abscess",
        "cavitation", "hernia/diaphragmatic", "diaphragm/elevated",
    ],
    "Pleural/Space": [
        "pleural effusion", "effusion", "costophrenic", "pneumothorax",
        "thickening/pleura", "pleural thickening", "hydropneumothorax",
        "empyema", "mesothelioma",
    ],
    "Bone/Fracture": [
        "fracture", "fractures", "scoliosis", "kyphosis", "osteophyte",
        "spondylosis", "degenerative", "deformity/ribs", "deformity/thoracic",
        "deformity/spine", "compression fracture", "lytic", "sclerotic",
        "bone and bones/thorax", "osteoporosis", "osteopenia", "rib/",
    ],
}
LABEL_NAMES   = list(MESH_KEYWORDS.keys())
NORMAL_LABELS = {"no finding"}


def assign_label(mesh: str, problems: str) -> str:
    combined = (str(mesh) + " " + str(problems)).lower()
    for label in LABEL_NAMES[1:]:
        if any(kw in combined for kw in MESH_KEYWORDS[label]):
            return label
    return "No Finding"


# ── Dataset ───────────────────────────────────────────────────────────────────
class FrontalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, transform):
        self.records   = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row      = self.records.iloc[idx]
        img_path = os.path.join(self.img_dir, str(row["filename"]))
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), 0)
        caption = (str(row.get("findings", "")) or
                   str(row.get("impression", "")) or "normal")
        return {
            "image":    self.transform(img),
            "caption":  caption,
            "label":    str(row["label"]),
            "uid":      str(row["uid"]),
            "filename": str(row["filename"]),
        }


# ── Stratified sampling ───────────────────────────────────────────────────────
def stratified_sample(dataset, all_labels, num_samples, min_per_class=40):
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    n_classes = len(unique_labels)
    base      = min_per_class
    remaining = max(0, num_samples - base * n_classes)
    total     = counts.sum()
    indices   = []
    rng       = np.random.default_rng(42)
    for lbl, cnt in zip(unique_labels, counts):
        prop_extra = int(remaining * cnt / total)
        n_take     = min(base + prop_extra, cnt)
        idx        = np.where(all_labels == lbl)[0]
        chosen     = rng.choice(idx, n_take, replace=False)
        indices.extend(chosen.tolist())
    rng.shuffle(indices)
    return Subset(dataset, indices)


# ── Model ────────────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str):
    print(f"[*] Loading model from: {checkpoint_path}")
    model = tp.StudyMedicalSwinBERT().to(DEVICE)
    ckpt  = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# ── FIX-8: Extract FUSED projected embeddings ─────────────────────────────────
def extract_fused_embeddings(model, subset: Subset):
    """
    Trả về fused embedding = (image_proj + text_proj) / 2

    Đây là joint embedding space sau MLP Projection Head —
    đúng không gian mà contrastive loss hoạt động.

    Logic ưu tiên:
      1. outputs có >= 4 phần tử → dùng outputs[2] (img_proj) + outputs[3] (txt_proj)
      2. outputs có >= 2 phần tử → avg của outputs[0] và outputs[1]
      3. Fallback: outputs[0] only
    """
    tokenizer = AutoTokenizer.from_pretrained(tp.TEXT_MODEL)

    def collate(batch):
        images    = torch.stack([b["image"]    for b in batch])
        captions  = [b["caption"]  for b in batch]
        labels    = [b["label"]    for b in batch]
        uids      = [b["uid"]      for b in batch]
        filenames = [b["filename"] for b in batch]
        tok = tokenizer(
            captions, padding="max_length", truncation=True,
            max_length=tp.TEXT_MAX_LEN, return_tensors="pt",
        )
        return images, tok, labels, uids, filenames

    loader = DataLoader(
        subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate
    )
    all_fused, all_labels, all_uids, all_filenames = [], [], [], []
    embed_mode = None

    with torch.no_grad():
        for images, tok, labels, uids, filenames in tqdm(
            loader, desc="[*] Extracting fused embeddings"
        ):
            B             = images.size(0)
            images        = images.unsqueeze(1).to(DEVICE)
            view_mask     = torch.ones (B, 1, dtype=torch.bool,  device=DEVICE)
            view_type_ids = torch.zeros(B, 1, dtype=torch.long,  device=DEVICE)
            tok           = {k: v.to(DEVICE) for k, v in tok.items()}

            outputs = model(
                images, view_mask, view_type_ids,
                tok["input_ids"], tok["attention_mask"],
            )

            # Chọn embedding đúng (chỉ in mode 1 lần)
            if len(outputs) >= 4:
                img_proj = outputs[2].detach()
                txt_proj = outputs[3].detach()
                fused    = (img_proj + txt_proj) / 2.0
                if embed_mode is None:
                    embed_mode = "projected (outputs[2]+outputs[3])/2"
            elif len(outputs) >= 2:
                img_feat = outputs[0].detach()
                txt_feat = outputs[1].detach()
                if img_feat.shape[-1] == txt_feat.shape[-1]:
                    img_n = torch.nn.functional.normalize(img_feat, dim=-1)
                    txt_n = torch.nn.functional.normalize(txt_feat, dim=-1)
                    fused = (img_n + txt_n) / 2.0
                else:
                    img_n = torch.nn.functional.normalize(img_feat, dim=-1)
                    txt_n = torch.nn.functional.normalize(txt_feat, dim=-1)
                    fused = torch.cat([img_n, txt_n], dim=-1)
                if embed_mode is None:
                    embed_mode = "avg raw features (outputs[0]+outputs[1])/2"
            else:
                fused = outputs[0].detach()
                if embed_mode is None:
                    embed_mode = "image-only fallback (outputs[0])"

            all_fused.append(fused.cpu().numpy())
            all_labels.extend(labels)
            all_uids.extend(uids)
            all_filenames.extend(filenames)

    print(f"[*] Embedding mode: {embed_mode}")
    embeddings = np.concatenate(all_fused, axis=0)
    return embeddings, all_labels, all_uids, all_filenames


# ── FIX-9: K-Means clustering ────────────────────────────────────────────────
def run_kmeans_clustering(emb_norm, n_clusters, condition_labels):
    print(f"[*] Running K-Means (k={n_clusters}) ...")
    km          = KMeans(n_clusters=n_clusters, random_state=42,
                         n_init=20, max_iter=500)
    cluster_ids = km.fit_predict(emb_norm)
    sil_avg     = silhouette_score  (emb_norm, cluster_ids, metric="cosine")
    sil_samples = silhouette_samples(emb_norm, cluster_ids, metric="cosine")

    unique_labels = sorted(set(condition_labels))
    gt_ids        = np.array([unique_labels.index(l) for l in condition_labels])
    ari  = adjusted_rand_score          (gt_ids, cluster_ids)
    nmi  = normalized_mutual_info_score (gt_ids, cluster_ids)

    print(f"[*] Silhouette Score (cosine) : {sil_avg:.4f}")
    print(f"[*] ARI  (cluster vs GT label): {ari:.4f}")
    print(f"[*] NMI  (cluster vs GT label): {nmi:.4f}")
    return cluster_ids, sil_avg, sil_samples


# ── Plot: Silhouette ──────────────────────────────────────────────────────────
def plot_silhouette(emb_norm, cluster_ids, condition_labels,
                    sil_samples, sil_avg, save_path):
    from collections import Counter
    active = sorted([k for k, v in Counter(cluster_ids).items() if v >= 2])
    if len(active) < 2:
        print("[!] Not enough clusters for silhouette plot."); return

    unique_conds  = sorted(set(condition_labels))
    cond_to_color = {c: plt.cm.tab10(i / max(len(unique_conds), 1))
                     for i, c in enumerate(unique_conds)}

    fig, ax = plt.subplots(figsize=(12, max(6, len(active) * 1.8)))
    y_lower = 10
    for k in active:
        mask      = cluster_ids == k
        ith_vals  = sil_samples[mask]
        ith_conds = [condition_labels[i] for i, m in enumerate(mask) if m]
        order     = np.argsort(ith_vals)[::-1]
        ith_vals  = ith_vals[order]
        ith_conds = [ith_conds[i] for i in order]
        size      = len(ith_vals)
        for j, (val, cond) in enumerate(zip(ith_vals, ith_conds)):
            ax.barh(y_lower + j, val, height=1.0,
                    color=cond_to_color[cond], edgecolor="none", alpha=0.85)
        ax.text(-0.07, y_lower + size / 2, f"Cluster {k}\n(n={size})",
                ha="right", va="center", fontsize=9, fontweight="bold")
        y_lower += size + 8

    ax.axvline(x=sil_avg, color="crimson", linestyle="--", linewidth=1.8)
    patches = [mpatches.Patch(color=cond_to_color[c], label=c)
               for c in unique_conds]
    patches.append(plt.Line2D([0], [0], color="crimson", linestyle="--",
                               linewidth=1.8, label=f"Avg = {sil_avg:.3f}"))
    ax.legend(handles=patches, loc="lower right", fontsize=9,
              framealpha=0.9, ncol=2)
    ax.set_xlim(-0.5, 1.0)
    ax.set_xlabel("Silhouette Coefficient (cosine)", fontsize=13)
    ax.set_yticks([])
    ax.set_title(
        f"Silhouette Analysis — Fused Multimodal Embedding\n"
        f"K-Means k={len(active)}  |  Avg Score (cosine): {sil_avg:.3f}",
        fontsize=14, fontweight="bold", pad=16,
    )
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    sns.despine(ax=ax, left=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✔] Silhouette plot → {save_path}")


# ── Plot: UMAP / PCA 2D (FIX-11) ─────────────────────────────────────────────
def plot_umap_clusters(emb_norm, cluster_ids, condition_labels, save_path):
    try:
        import umap as umap_lib
        reducer = umap_lib.UMAP(n_components=2, random_state=42,
                                metric="cosine", n_neighbors=30, min_dist=0.1)
        coords  = reducer.fit_transform(emb_norm)
        method  = "UMAP"
    except ImportError:
        print("[!] umap-learn not found → PCA 2D fallback.")
        coords = PCA(n_components=2, random_state=42).fit_transform(emb_norm)
        method = "PCA"

    unique_clusters   = sorted(set(cluster_ids))
    unique_conditions = sorted(set(condition_labels))
    cluster_colors    = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
    cond_markers      = ["o", "s", "^", "D", "v", "P", "*", "X"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left: K-Means cluster
    ax = axes[0]
    for k, col in zip(unique_clusters, cluster_colors):
        mask = cluster_ids == k
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[col],
                   s=18, alpha=0.7, label=f"Cluster {k}", edgecolors="none")
    ax.set_title(f"{method} — K-Means Clusters", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, markerscale=1.5)
    ax.set_xlabel(f"{method}-1"); ax.set_ylabel(f"{method}-2")
    ax.grid(True, linestyle=":", alpha=0.3)

    # Right: Ground-truth label
    ax = axes[1]
    cond_colors = plt.cm.Set2(np.linspace(0, 1, len(unique_conditions)))
    for cond, col, mk in zip(unique_conditions, cond_colors, cond_markers):
        mask = np.array([l == cond for l in condition_labels])
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[col], s=18,
                   alpha=0.7, marker=mk, label=cond, edgecolors="none")
    ax.set_title(f"{method} — Clinical Labels", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, markerscale=1.5)
    ax.set_xlabel(f"{method}-1"); ax.set_ylabel(f"{method}-2")
    ax.grid(True, linestyle=":", alpha=0.3)

    plt.suptitle(
        "Fused Multimodal Embedding — Cluster Visualization\n"
        "Task 3: Clustering-Guided Negative Sampling",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✔] {method} plot → {save_path}")


# ── Plot: Hierarchical Dendrogram ────────────────────────────────────────────
def plot_hierarchical_clusters(emb_norm, labels, save_path):
    print("[*] Hierarchical clustering on label centroids ...")
    unique_labels = sorted(set(labels))
    centroids     = [emb_norm[np.array([l == lbl for l in labels])].mean(axis=0)
                     for lbl in unique_labels]
    centroids     = normalize(np.array(centroids), norm="l2")
    linked        = linkage(centroids, method="ward")
    plt.figure(figsize=(10, 7))
    dendrogram(linked, orientation="top", labels=unique_labels,
               distance_sort="descending", show_leaf_counts=True, leaf_font_size=10)
    plt.title("Hierarchical Clustering of Clinical Pathologies\n(Fused Embedding Space)",
              fontsize=14, fontweight="bold", pad=20)
    plt.ylabel("Ward Distance", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✔] Dendrogram → {save_path}")


# ── Plot: Similarity Clustermap ───────────────────────────────────────────────
def plot_similarity_clustermap(emb_norm, labels, save_path):
    unique_labels = sorted(set(labels))
    centroids     = [emb_norm[np.array([l == lbl for l in labels])].mean(axis=0)
                     for lbl in unique_labels]
    centroids     = normalize(np.array(centroids), norm="l2")
    sim_df        = pd.DataFrame(np.dot(centroids, centroids.T),
                                 index=unique_labels, columns=unique_labels)
    g = sns.clustermap(sim_df, annot=True, fmt=".2f", cmap="YlGnBu",
                       figsize=(10, 10), cbar_pos=(0.02, 0.8, 0.05, 0.18))
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45)
    g.fig.suptitle("Clinical Similarity Clustermap (Fused Embedding)\n"
                   "Task 3: Clustering-Guided Negative Sampling",
                   fontsize=14, fontweight="bold", y=1.02)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✔] Clustermap → {save_path}")


# ── FIX-10: Export cluster assignments ────────────────────────────────────────
def export_cluster_assignments(uids, filenames, labels, cluster_ids, save_path):
    """
    Export CSV dùng trong training loop:

    Trong negative sampling loop:
      anchor = (uid_i, cluster_c)
      negative candidates = samples với cluster_id != c   ← hard negatives
      excluded            = samples với cluster_id == c   ← false negative guard
    """
    df = pd.DataFrame({
        "uid":            uids,
        "filename":       filenames,
        "clinical_label": labels,
        "cluster_id":     cluster_ids,
    })
    df.to_csv(save_path, index=False)
    print(f"[✔] Cluster assignments → {save_path}")
    print(f"    Rows: {len(df)}")
    print(f"    Cluster distribution:\n{df['cluster_id'].value_counts().sort_index().to_string()}")
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Task 3: Clustering-Guided Negative Sampling — Silhouette Analysis"
    )
    p.add_argument("--checkpoint",      required=True,
                   help="Path to best_clinical_valid.pt")
    p.add_argument("--reports_csv",     required=True)
    p.add_argument("--projections_csv", required=True)
    p.add_argument("--img_dir",         required=True)
    p.add_argument("--out_dir",         default="plots")
    p.add_argument("--num_samples",     type=int, default=800,
                   help="Total pathology samples (normals excluded before sampling)")
    p.add_argument("--min_per_class",   type=int, default=40,
                   help="Min samples per clinical class")
    p.add_argument("--n_clusters",      type=int, default=4,
                   help="K for K-Means (= số nhóm bệnh lý quan tâm)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. CSVs ─────────────────────────────────────────────────────────────────
    print("[*] Loading CSVs ...")
    reports     = pd.read_csv(args.reports_csv)
    projections = pd.read_csv(args.projections_csv)
    frontal     = (projections[projections["projection"] == "Frontal"]
                   .drop_duplicates("uid").reset_index(drop=True))
    merged      = frontal.merge(
        reports[["uid", "MeSH", "Problems", "findings", "impression"]],
        on="uid", how="left"
    )
    for col in ["MeSH", "Problems", "findings", "impression"]:
        merged[col] = merged[col].fillna("")
    merged["label"] = merged.apply(
        lambda r: assign_label(r["MeSH"], r["Problems"]), axis=1
    )
    print(f"[*] Full dataset: {len(merged)} Frontal studies")
    print("[*] Label distribution (ALL):\n" + merged["label"].value_counts().to_string())

    # 2. Filter normals trước sampling ────────────────────────────────────────
    path_mask  = ~merged["label"].str.lower().isin(NORMAL_LABELS)
    path_df    = merged[path_mask].reset_index(drop=True)
    print(f"\n[*] Excluded {(~path_mask).sum()} No-Finding studies.")
    print(f"[*] Pathology pool: {len(path_df)}")
    print("[*] Pathology distribution:\n" + path_df["label"].value_counts().to_string())

    # 3. Dataset & sampling ───────────────────────────────────────────────────
    from torchvision import transforms as T
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = FrontalDataset(path_df, args.img_dir, transform)
    subset  = stratified_sample(
        dataset, path_df["label"].values,
        num_samples=args.num_samples, min_per_class=args.min_per_class,
    )
    print(f"[*] Subset size: {len(subset)}")

    # 4. Model ────────────────────────────────────────────────────────────────
    model = load_model(args.checkpoint)

    # 5. Extract FUSED embeddings (FIX-8) ─────────────────────────────────────
    embeddings, labels, uids, filenames = extract_fused_embeddings(model, subset)
    labels = [str(l).lower() for l in labels]

    # 6. Normalize + PCA ──────────────────────────────────────────────────────
    emb_norm = normalize(embeddings, norm="l2")
    if emb_norm.shape[1] > 128:
        print(f"[*] PCA: {emb_norm.shape[1]} → 128 dims ...")
        emb_norm = normalize(
            PCA(n_components=128, random_state=42).fit_transform(emb_norm), norm="l2"
        )

    # 7. K-Means (FIX-9) ──────────────────────────────────────────────────────
    cluster_ids, sil_avg, sil_samples = run_kmeans_clustering(
        emb_norm, args.n_clusters, labels
    )

    print(f"\n[*] Cluster breakdown:")
    for k in sorted(set(cluster_ids)):
        mask       = cluster_ids == k
        top        = pd.Series([labels[i] for i, m in enumerate(mask) if m]).value_counts()
        dominant   = top.index[0] if len(top) else "?"
        print(f"    Cluster {k}: n={mask.sum()}, dominant={dominant}")

    # 8. Plots ────────────────────────────────────────────────────────────────
    plot_silhouette(
        emb_norm, cluster_ids, labels, sil_samples, sil_avg,
        os.path.join(args.out_dir, "silhouette_plot.png"),
    )
    plot_umap_clusters(
        emb_norm, cluster_ids, labels,
        os.path.join(args.out_dir, "umap_clusters.png"),
    )
    plot_hierarchical_clusters(
        emb_norm, labels,
        os.path.join(args.out_dir, "hierarchical_dendrogram.png"),
    )
    plot_similarity_clustermap(
        emb_norm, labels,
        os.path.join(args.out_dir, "similarity_clustermap.png"),
    )

    # 9. Export cluster_assignments.csv (FIX-10) ──────────────────────────────
    export_cluster_assignments(
        uids, filenames, labels, cluster_ids,
        os.path.join(args.out_dir, "cluster_assignments.csv"),
    )

    print(f"\n{'='*60}")
    print(f"  Task 3 Pipeline Complete")
    print(f"  Silhouette Score : {sil_avg:.4f}")
    print(f"  cluster_assignments.csv → ready for negative sampling")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()