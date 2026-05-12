"""
silhouette_plot.py  (FIXED VERSION)
=====================================
Standalone script to generate a Silhouette Plot for the IU X-ray model.

Input:
  - best_clinical_valid.pt   : trained model checkpoint
  - indiana_reports.csv      : uid, MeSH, Problems, findings, impression ...
  - indiana_projections.csv  : uid, filename, projection

Output:
  - silhouette_plot.png
  - hierarchical_dendrogram.png
  - similarity_clustermap.png

Usage (on Kaggle):
  !python plot/silhouette_plot.py \
      --checkpoint /kaggle/input/duong-dan-model/best_clinical_valid.pt \
      --reports_csv /kaggle/input/chest-xrays-indiana-university/indiana_reports.csv \
      --projections_csv /kaggle/input/chest-xrays-indiana-university/indiana_projections.csv \
      --img_dir /kaggle/input/chest-xrays-indiana-university/images/images_normalized \
      --out_dir /kaggle/working/plots \
      --num_samples 800 \
      --min_per_class 40

FIXES APPLIED (v2):
  FIX-1  model output unpacked explicitly with .detach() before .numpy()
  FIX-2  tokenizer moved inside collate_fn (closure captured correctly)
  FIX-3  pathology_mask uses np.isin + consistent lowercase, no stale list
  FIX-4  stratified sampling ensures every class has >= min_per_class samples
  FIX-5  MESH_KEYWORDS greatly expanded — catches emphysema, granuloma,
         nodule, scoliosis, degenerative, thickening/pleura, hernia, etc.
         (was silently dumping ~700 real pathology rows into "No Finding")
  FIX-6  Dataset built from pathology-only rows BEFORE sampling — avoids
         wasting the num_samples budget on normals that get discarded later
  FIX-7  MeSH uses full-path matching (e.g. "cardiac shadow" catches both
         "cardiac shadow/enlarged" and "cardiac shadow/borderline" without
         needing separate entries)
"""

import sys, os, argparse, warnings
warnings.filterwarnings("ignore")

# ── path setup ──────────────────────────────────────────────────────────────
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent not in sys.path:
    sys.path.insert(0, parent)

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, silhouette_samples
from tqdm import tqdm
from transformers import AutoTokenizer
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

import train_proposed as tp

# ── constants ────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE   = 384
BATCH_SIZE = 16

# ── FIX-5: expanded keyword dictionary ──────────────────────────────────────
# Strategy: match MeSH "Term/location/qualifier" paths using prefix substrings.
# E.g. "cardiac shadow" matches both "cardiac shadow/enlarged" and
# "cardiac shadow/borderline" — no need to list every variant separately.
MESH_KEYWORDS = {
    # ── NOTE: "No Finding" is checked LAST (fallback) ──────────────────────
    "No Finding": [
        "normal",
        "no indexing",
        "negative",
    ],

    # ── Cardiac / Mediastinal ───────────────────────────────────────────────
    "Heart/Mediastinum": [
        "cardiomegaly",
        "cardiac shadow",            # covers /enlarged and /borderline
        "mediastinum/enlarged",
        "mediastinum/widened",
        "aorta/tortuous",
        "aorta, thoracic/tortuous",
        "pericardial effusion",
        "pericardium",
        "hilar",
        "pulmonary artery/enlarged",
        "pulmonary hypertension",
        "vascular",
        "venous congestion",
        "stents/coronary",           # post-surgical cardiac marker
    ],

    # ── Lung Parenchyma / Airspace ──────────────────────────────────────────
    "Lung/Parenchyma": [
        # original
        "pneumonia",
        "airspace disease",
        "consolidation",
        "atelectasis",
        "pulmonary atelectasis",
        "edema",
        "pulmonary congestion",
        "pulmonary edema",
        "opacity",
        "shadow",
        "interstitial",
        # FIX-5 additions
        "emphysema",
        "bullous emphysema",
        "fibrosis",
        "pulmonary fibrosis",
        "granuloma",                 # Calcified Granuloma/lung → 395 cases
        "granulomatous disease",
        "nodule",
        "mass/lung",
        "lung/hyperdistention",      # COPD / hyperinflation
        "lung/hypoinflation",
        "diaphragm/flattened",       # sign of hyperinflation
        "cicatrix/lung",             # scarring
        "density/lung",
        "density/cardiophrenic",
        "infiltrate",
        "aspiration",
        "abscess",
        "cavitation",
        "hernia/diaphragmatic",
        "diaphragm/elevated",        # sub-phrenic / phrenic palsy
    ],

    # ── Pleural / Chest Wall ────────────────────────────────────────────────
    "Pleural/Space": [
        # original
        "pleural effusion",
        "effusion",
        "costophrenic",
        "pneumothorax",
        # FIX-5 additions
        "thickening/pleura",         # pleural thickening — 46+ cases missed
        "pleural thickening",
        "hydropneumothorax",
        "empyema",
        "mesothelioma",
    ],

    # ── Bone / Musculoskeletal ──────────────────────────────────────────────
    "Bone/Fracture": [
        # original
        "fracture",
        "fractures",
        # FIX-5 additions — spine & rib pathology
        "scoliosis",                 # 88 cases, 63 were missed
        "kyphosis",                  # 28 cases
        "osteophyte",                # degenerative spine — very common
        "spondylosis",               # 25+ cases
        "degenerative",              # covers thoracic vertebrae/degenerative
        "deformity/ribs",
        "deformity/thoracic",
        "deformity/spine",
        "compression fracture",
        "lytic",
        "sclerotic",
        "bone and bones/thorax",
        "osteoporosis",
        "osteopenia",
        "rib/",                      # rib lesions (rib/fracture, rib/lesion …)
    ],
}

LABEL_NAMES = list(MESH_KEYWORDS.keys())

# Labels treated as "normal" — excluded before silhouette analysis
NORMAL_LABELS = {"no finding"}


def assign_label(mesh: str, problems: str) -> str:
    """
    Return the primary clinical label for one row.

    FIX-7: checks pathology categories first (priority order), then falls
    back to "No Finding". Uses lowercase substring matching so that
    MeSH hierarchical paths like "Pulmonary Atelectasis/base/bilateral"
    are caught by the keyword "atelectasis".
    """
    combined = (str(mesh) + " " + str(problems)).lower()

    # Check pathologies in priority order (skip index 0 = "No Finding")
    for label in LABEL_NAMES[1:]:
        if any(kw in combined for kw in MESH_KEYWORDS[label]):
            return label

    return "No Finding"


# ── simple single-image dataset ──────────────────────────────────────────────
class FrontalDataset(Dataset):
    """Loads one Frontal image per study (uid) with its caption."""

    def __init__(self, df: pd.DataFrame, img_dir: str, transform):
        self.records   = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        img_path = os.path.join(self.img_dir, str(row["filename"]))
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), 0)

        caption = (str(row.get("findings", "")) or
                   str(row.get("impression", "")) or
                   "normal")
        return {
            "image":   self.transform(img),
            "caption": caption,
            "label":   str(row["label"]),
        }


# ── stratified sampling ───────────────────────────────────────────────────────
def stratified_sample(dataset: Dataset,
                      all_labels: np.ndarray,
                      num_samples: int,
                      min_per_class: int = 40) -> Subset:
    """
    Sample indices so that:
      - every class gets at least min_per_class samples (or all available)
      - total samples ≈ num_samples (distributed proportionally after the minimum)

    FIX-4: replaces the plain random choice that could leave rare classes
            (e.g. Bone/Fracture) with fewer than 2 samples, crashing
            silhouette_score().

    FIX-6: this function now receives a pathology-only dataset/labels array,
            so none of the num_samples budget is wasted on "No Finding" rows
            that would be discarded anyway.
    """
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    n_classes = len(unique_labels)

    # Budget per class: at least min_per_class, rest distributed proportionally
    base      = min_per_class
    remaining = max(0, num_samples - base * n_classes)
    total     = counts.sum()

    indices = []
    rng = np.random.default_rng(42)

    for lbl, cnt in zip(unique_labels, counts):
        prop_extra = int(remaining * cnt / total)
        n_take     = min(base + prop_extra, cnt)
        idx        = np.where(all_labels == lbl)[0]
        chosen     = rng.choice(idx, n_take, replace=False)
        indices.extend(chosen.tolist())

    rng.shuffle(indices)
    return Subset(dataset, indices)


# ── model loading ─────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str):
    print(f"[*] Loading model from: {checkpoint_path}")
    model = tp.StudyMedicalSwinBERT().to(DEVICE)
    ckpt  = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# ── feature extraction ────────────────────────────────────────────────────────
def extract_embeddings(model, subset: Subset):
    """
    Extract image embeddings from the model.

    FIX-1: model output is unpacked explicitly; .detach() is called before
            .cpu().numpy() to avoid RuntimeError when grad is attached.
    FIX-2: tokenizer is instantiated once and captured inside collate_fn
            as a proper closure — no more tokenizer-out-of-scope risk.
    """
    tokenizer = AutoTokenizer.from_pretrained(tp.TEXT_MODEL)

    # ── collate: tokenise captions here so the closure captures tokenizer ──
    def collate(batch):
        images   = torch.stack([b["image"]   for b in batch])
        captions = [b["caption"] for b in batch]
        labels   = [b["label"]   for b in batch]
        tok = tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=tp.TEXT_MAX_LEN,
            return_tensors="pt",
        )
        return images, tok, labels

    loader = DataLoader(
        subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate
    )
    all_embs, all_labels = [], []

    with torch.no_grad():
        for images, tok, labels in tqdm(loader, desc="[*] Extracting embeddings"):
            B = images.size(0)
            images        = images.unsqueeze(1).to(DEVICE)          # (B,1,C,H,W)
            view_mask     = torch.ones (B, 1, dtype=torch.bool,  device=DEVICE)
            view_type_ids = torch.zeros(B, 1, dtype=torch.long,  device=DEVICE)

            tok = {k: v.to(DEVICE) for k, v in tok.items()}

            # FIX-1 — unpack output explicitly; index 0 = img_features
            outputs  = model(
                images, view_mask, view_type_ids,
                tok["input_ids"], tok["attention_mask"],
            )
            img_feat = outputs[0]           # shape: (B, hidden_dim)
            # .detach() prevents "can't convert a tensor with requires_grad"
            all_embs.append(img_feat.detach().cpu().numpy())
            all_labels.extend(labels)

    embeddings = np.concatenate(all_embs, axis=0)
    return embeddings, all_labels


# ── Silhouette Plot ───────────────────────────────────────────────────────────
def plot_silhouette(embeddings, cluster_ids, condition_labels, save_path: str):
    """
    Draw a Silhouette Plot grouped by active model prototypes.
    Empty clusters (n < 2) are automatically filtered out.
    """
    from collections import Counter

    counts          = Counter(cluster_ids)
    active_clusters = sorted([k for k, v in counts.items() if v >= 2])

    if len(active_clusters) < 2:
        print("[!] Not enough active clusters for silhouette analysis.")
        print(f"    Active clusters: {active_clusters}")
        return

    active_mask = np.array([c in active_clusters for c in cluster_ids])
    emb_active  = embeddings[active_mask]
    ids_active  = cluster_ids[active_mask]
    cond_active = [condition_labels[i] for i, m in enumerate(active_mask) if m]

    remap        = {old: new for new, old in enumerate(active_clusters)}
    ids_remapped = np.array([remap[c] for c in ids_active])
    n_clusters   = len(active_clusters)

    print(f"[*] Active clusters : {n_clusters} / {int(cluster_ids.max()) + 1} total")
    print(f"[*] Samples in active clusters: {len(emb_active)} / {len(embeddings)}")
    print(f"    Cluster sizes: {dict(Counter(ids_remapped))}")

    unique_conditions = sorted(set(cond_active))
    cond_to_color = {
        c: plt.cm.tab20(i / max(len(unique_conditions), 1))
        for i, c in enumerate(unique_conditions)
    }

    emb_norm = normalize(emb_active, norm="l2")
    if emb_norm.shape[1] > 64:
        print(f"[*] PCA: {emb_norm.shape[1]} → 64 dims ...")
        pca      = PCA(n_components=64, random_state=42)
        emb_norm = pca.fit_transform(emb_norm)

    sil_vals = silhouette_samples(emb_norm, ids_remapped, metric="cosine")
    sil_avg  = silhouette_score (emb_norm, ids_remapped, metric="cosine")
    print(f"[*] Average Silhouette Score (cosine): {sil_avg:.4f}")

    fig, ax = plt.subplots(figsize=(12, max(6, n_clusters * 1.5)))
    y_lower = 10

    for k in range(n_clusters):
        mask      = ids_remapped == k
        ith_vals  = sil_vals[mask]
        ith_conds = [cond_active[i] for i, m in enumerate(mask) if m]

        order     = np.argsort(ith_vals)[::-1]
        ith_vals  = ith_vals[order]
        ith_conds = [ith_conds[i] for i in order]

        size    = len(ith_vals)
        y_upper = y_lower + size

        for j, (val, cond) in enumerate(zip(ith_vals, ith_conds)):
            ax.barh(y_lower + j, val, height=1.0,
                    color=cond_to_color[cond], edgecolor="none", alpha=0.85)

        orig_id = active_clusters[k]
        ax.text(-0.06, y_lower + size / 2,
                f"Cluster {orig_id}\n(n={size})",
                ha="right", va="center", fontsize=9, fontweight="bold")

        y_lower = y_upper + 8

    ax.axvline(x=sil_avg, color="crimson", linestyle="--", linewidth=1.8,
               label=f"Avg = {sil_avg:.3f}")

    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, color=cond_to_color[c], label=c)
        for c in unique_conditions
    ]
    ax.legend(
        handles=legend_patches + [
            plt.Line2D([0], [0], color="crimson", linestyle="--",
                       linewidth=1.8, label=f"Avg (cosine) = {sil_avg:.3f}")
        ],
        loc="lower right", fontsize=9, framealpha=0.9, ncol=2,
    )

    ax.set_xlim(-0.5, 1.0)
    ax.set_xlabel("Silhouette Coefficient (cosine)", fontsize=13)
    ax.set_yticks([])
    ax.set_title(
        f"Silhouette Analysis — Image Latent Space\n"
        f"Active Prototypes: {n_clusters}  |  "
        f"Avg Score (cosine): {sil_avg:.3f}",
        fontsize=14, fontweight="bold", pad=16,
    )
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    sns.despine(ax=ax, left=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✔] Silhouette plot saved → {save_path}")


# ── Hierarchical Dendrogram ───────────────────────────────────────────────────
def plot_hierarchical_clusters(embeddings, labels, save_path: str):
    """
    Perform Hierarchical Clustering on class centroids and plot a Dendrogram.
    Shows how the model relates different pathologies to each other.
    """
    print("[*] Performing Hierarchical Clustering analysis ...")

    unique_labels = sorted(set(labels))
    centroids = []
    for lbl in unique_labels:
        mask = np.array([l == lbl for l in labels])
        centroids.append(embeddings[mask].mean(axis=0))

    centroids = normalize(np.array(centroids), norm="l2")
    linked    = linkage(centroids, method="ward")

    plt.figure(figsize=(10, 7))
    dendrogram(linked,
               orientation="top",
               labels=unique_labels,
               distance_sort="descending",
               show_leaf_counts=True,
               leaf_font_size=10)

    plt.title(
        "Hierarchical Clustering of Clinical Pathologies\n"
        "(Latent Space Relationship Tree)",
        fontsize=14, fontweight="bold", pad=20,
    )
    plt.ylabel("Distance (Similarity)", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✔] Hierarchical Dendrogram saved → {save_path}")


# ── Similarity Clustermap ─────────────────────────────────────────────────────
def plot_similarity_clustermap(embeddings, labels, save_path: str):
    """
    Generate a Heatmap + Dendrogram (Clustermap) of group-to-group similarity.
    """
    unique_labels = sorted(set(labels))
    centroids = []
    for lbl in unique_labels:
        mask = np.array([l == lbl for l in labels])
        centroids.append(embeddings[mask].mean(axis=0))

    centroids  = normalize(np.array(centroids), norm="l2")
    sim_matrix = np.dot(centroids, centroids.T)
    sim_df     = pd.DataFrame(sim_matrix, index=unique_labels, columns=unique_labels)

    g = sns.clustermap(sim_df, annot=True, fmt=".2f", cmap="YlGnBu",
                       figsize=(10, 10), cbar_pos=(0.02, 0.8, 0.05, 0.18))
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45)
    g.fig.suptitle(
        "Clinical Similarity Clustermap\n(Hierarchical Relationship)",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✔] Similarity Clustermap saved → {save_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",      required=True,
                   help="Path to best_clinical_valid.pt")
    p.add_argument("--reports_csv",     required=True,
                   help="Path to indiana_reports.csv")
    p.add_argument("--projections_csv", required=True,
                   help="Path to indiana_projections.csv")
    p.add_argument("--img_dir",         required=True,
                   help="Directory containing .png images")
    p.add_argument("--out_dir",         default="plots")
    p.add_argument("--num_samples",     type=int, default=800,
                   help="Total pathology images to sample (stratified). "
                        "Normal images are excluded before sampling (FIX-6).")
    p.add_argument("--min_per_class",   type=int, default=40,
                   help="Minimum samples guaranteed per pathology class (FIX-4). "
                        "Raised default from 15→40 to improve silhouette stability.")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load and merge CSVs ──────────────────────────────────────────────────
    print("[*] Loading CSVs ...")
    reports     = pd.read_csv(args.reports_csv)
    projections = pd.read_csv(args.projections_csv)

    frontal = (projections[projections["projection"] == "Frontal"]
               .drop_duplicates("uid")
               .reset_index(drop=True))

    merged = frontal.merge(
        reports[["uid", "MeSH", "Problems", "findings", "impression"]],
        on="uid", how="left",
    )
    for col in ["MeSH", "Problems", "findings", "impression"]:
        merged[col] = merged[col].fillna("")

    # FIX-5: assign_label now uses the expanded MESH_KEYWORDS
    merged["label"] = merged.apply(
        lambda r: assign_label(r["MeSH"], r["Problems"]), axis=1
    )
    print(f"[*] Full dataset: {len(merged)} Frontal studies")
    print("[*] Label distribution (ALL):")
    print(merged["label"].value_counts().to_string())

    # 2. FIX-6: filter normals BEFORE building dataset & sampling ─────────────
    # This prevents num_samples budget from being spent on rows that will be
    # discarded anyway. Previously: sample 800 → discard 540 normals → 260 left.
    # Now: sample 800 from ~1000 pathology rows → keep all 800.
    labels_lower    = merged["label"].str.lower()
    pathology_mask  = ~labels_lower.isin(NORMAL_LABELS)
    pathology_df    = merged[pathology_mask].reset_index(drop=True)
    n_normal        = (~pathology_mask).sum()
    n_patho         = pathology_mask.sum()

    print(f"\n[*] Excluded {n_normal} 'No Finding' studies before sampling.")
    print(f"[*] Pathology pool: {n_patho} studies")
    print("[*] Label distribution (PATHOLOGY ONLY):")
    print(pathology_df["label"].value_counts().to_string())

    if n_patho < 5:
        raise ValueError(
            f"Too few pathology samples ({n_patho}). "
            "Check your CSV or expand MESH_KEYWORDS."
        )

    # 3. Build dataset from pathology-only rows ────────────────────────────────
    from torchvision import transforms as T
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])
    dataset = FrontalDataset(pathology_df, args.img_dir, transform)

    # 4. FIX-4 + FIX-6: stratified sample from pathology pool only ────────────
    all_labels_arr = pathology_df["label"].values
    subset = stratified_sample(
        dataset, all_labels_arr,
        num_samples=args.num_samples,
        min_per_class=args.min_per_class,
    )
    print(f"[*] Stratified subset size: {len(subset)}")

    # 5. Load model ────────────────────────────────────────────────────────────
    model = load_model(args.checkpoint)

    # 6. Extract embeddings (FIX-1 & FIX-2 inside) ────────────────────────────
    embeddings, labels = extract_embeddings(model, subset)
    labels_np = np.array([str(l).lower() for l in labels])

    # Sanity check — should be 0 normals since we filtered before sampling
    n_normal_remaining = np.isin(labels_np, list(NORMAL_LABELS)).sum()
    if n_normal_remaining > 0:
        print(f"[!] Warning: {n_normal_remaining} 'normal' samples slipped through — removing.")
        keep = ~np.isin(labels_np, list(NORMAL_LABELS))   # FIX-3 style
        embeddings = embeddings[keep]
        labels_np  = labels_np[keep]

    labels = labels_np.tolist()

    # 7. Map clinical labels → numeric cluster IDs ─────────────────────────────
    unique_labels = sorted(set(labels))
    label_to_id   = {l: i for i, l in enumerate(unique_labels)}
    cluster_ids   = np.array([label_to_id[l] for l in labels])

    print(f"\n[*] Final {len(unique_labels)} clinical groups for Silhouette:")
    for l, i in label_to_id.items():
        n = (cluster_ids == i).sum()
        print(f"    Cluster {i}: {l}  (n={n})")

    # 8. Draw Silhouette Plot ──────────────────────────────────────────────────
    plot_silhouette(
        embeddings, cluster_ids, labels,
        os.path.join(args.out_dir, "silhouette_plot.png"),
    )

    # 9. Hierarchical Visualizations ───────────────────────────────────────────
    plot_hierarchical_clusters(
        embeddings, labels,
        os.path.join(args.out_dir, "hierarchical_dendrogram.png"),
    )

    plot_similarity_clustermap(
        embeddings, labels,
        os.path.join(args.out_dir, "similarity_clustermap.png"),
    )


if __name__ == "__main__":
    main()