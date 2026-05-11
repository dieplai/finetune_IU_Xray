"""
silhouette_plot.py
==================
Standalone script to generate a Silhouette Plot for the IU X-ray model.

Input:
  - best_clinical_valid.pt   : trained model checkpoint
  - indiana_reports.csv      : uid, MeSH, Problems, findings, impression ...
  - indiana_projections.csv  : uid, filename, projection

Output:
  - silhouette_plot.png

Usage (on Kaggle):
  !python plot/silhouette_plot.py \
      --checkpoint /kaggle/input/duong-dan-model/best_clinical_valid.pt \
      --reports_csv /kaggle/input/chest-xrays-indiana-university/indiana_reports.csv \
      --projections_csv /kaggle/input/chest-xrays-indiana-university/indiana_projections.csv \
      --img_dir /kaggle/input/chest-xrays-indiana-university/images/images_normalized \
      --out_dir /kaggle/working/plots
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

import train_proposed as tp

# ── constants ────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 384
BATCH_SIZE = 16

# ── label extraction from MeSH / Problems ────────────────────────────────────
MESH_KEYWORDS = {
    "No Finding":                 ["normal", "no indexing", "negative"],
    "Cardiomegaly":               ["cardiomegaly", "cardiac shadow/enlarged", "cardiac shadow/borderline"],
    "Pleural Effusion":           ["pleural effusion", "effusion", "costophrenic"],
    "Atelectasis":                ["atelectasis", "pulmonary atelectasis"],
    "Pneumonia":                  ["pneumonia", "airspace disease", "consolidation"],
    "Edema":                      ["edema", "pulmonary congestion", "pulmonary edema"],
    "Pneumothorax":               ["pneumothorax"],
    "Fracture":                   ["fracture", "fractures"],
    "Lung Opacity":               ["opacity", "shadow", "interstitial"],
    "Enlarged Cardiomediastinum": ["mediastinum/enlarged"],
}
LABEL_NAMES = list(MESH_KEYWORDS.keys())


def assign_label(mesh: str, problems: str) -> str:
    """Return the primary clinical label for one row."""
    combined = (str(mesh) + " " + str(problems)).lower()
    # Check specific pathologies first (skip 'No Finding')
    for label in LABEL_NAMES[1:]:
        if any(kw in combined for kw in MESH_KEYWORDS[label]):
            return label
    return "No Finding"


# ── simple single-image dataset ──────────────────────────────────────────────
class FrontalDataset(Dataset):
    """Loads one Frontal image per study (uid) with its caption."""

    def __init__(self, df: pd.DataFrame, img_dir: str, transform):
        self.records = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        img_path = os.path.join(self.img_dir, str(row["filename"]))

        # Load image (graceful fallback for missing files)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), 0)

        return {
            "image":   self.transform(img),
            "caption": str(row.get("findings", "") or row.get("impression", "") or "normal"),
            "label":   str(row["label"]),
        }


# ── model loading ─────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str):
    print(f"[*] Loading model from: {checkpoint_path}")
    model = tp.StudyMedicalSwinBERT().to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# ── feature extraction ────────────────────────────────────────────────────────
def extract_embeddings(model, dataset, num_samples: int):
    """Extract image embeddings from the model."""
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    subset  = Subset(dataset, indices)

    tokenizer = AutoTokenizer.from_pretrained(tp.TEXT_MODEL)

    def collate(batch):
        return (
            torch.stack([b["image"] for b in batch]),
            [b["caption"] for b in batch],
            [b["label"] for b in batch],
        )

    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    all_embs, all_labels = [], []

    with torch.no_grad():
        for images, captions, labels in tqdm(loader, desc="[*] Extracting embeddings"):
            B = images.size(0)
            images        = images.unsqueeze(1).to(DEVICE)   # (B,1,C,H,W)
            view_mask     = torch.ones(B, 1, dtype=torch.bool,  device=DEVICE)
            view_type_ids = torch.zeros(B, 1, dtype=torch.long, device=DEVICE)

            tok = tokenizer(
                captions, padding="max_length", truncation=True,
                max_length=tp.TEXT_MAX_LEN, return_tensors="pt",
            ).to(DEVICE)

            img_feat = model(
                images, view_mask, view_type_ids,
                tok["input_ids"], tok["attention_mask"],
            )[0]  # Take img_features (index 0)

            all_embs.append(img_feat.cpu().numpy())
            all_labels.extend(labels)

    embeddings = np.concatenate(all_embs, axis=0)
    return embeddings, all_labels


# ── Silhouette Plot ───────────────────────────────────────────────────────────
def plot_silhouette(embeddings, cluster_ids, condition_labels, save_path: str):
    """
    Draw a Silhouette Plot grouped by active model prototypes.
    Empty clusters (n=0 or n=1) are automatically filtered out.
    """
    from collections import Counter

    # ── Filter: keep only clusters with >= 2 samples ─────────────────────────
    counts = Counter(cluster_ids)
    active_clusters = sorted([k for k, v in counts.items() if v >= 2])
    
    if len(active_clusters) < 2:
        print("[!] Not enough active clusters for silhouette analysis.")
        print(f"    Active clusters: {active_clusters}")
        return
    
    # Keep only samples from active clusters
    active_mask = np.array([c in active_clusters for c in cluster_ids])
    emb_active  = embeddings[active_mask]
    ids_active  = cluster_ids[active_mask]
    cond_active = [condition_labels[i] for i, m in enumerate(active_mask) if m]
    
    # Re-map cluster IDs to contiguous 0, 1, 2, ...
    remap = {old: new for new, old in enumerate(active_clusters)}
    ids_remapped = np.array([remap[c] for c in ids_active])
    n_clusters = len(active_clusters)
    
    print(f"[*] Active clusters: {n_clusters} / {int(cluster_ids.max())+1} total prototypes")
    print(f"[*] Samples in active clusters: {len(emb_active)} / {len(embeddings)}")
    print(f"    Cluster sizes: {dict(Counter(ids_remapped))}")
    
    unique_conditions = sorted(set(cond_active))
    cond_to_color = {c: plt.cm.tab20(i / max(len(unique_conditions), 1))
                     for i, c in enumerate(unique_conditions)}

    # ── Silhouette computation with cosine metric ─────────────────────────────
    emb_norm = normalize(emb_active, norm='l2')
    if emb_norm.shape[1] > 64:
        print(f"[*] PCA: {emb_norm.shape[1]} → 64 dims ...")
        pca = PCA(n_components=64, random_state=42)
        emb_norm = pca.fit_transform(emb_norm)

    sil_vals = silhouette_samples(emb_norm, ids_remapped, metric='cosine')
    sil_avg  = silhouette_score(emb_norm, ids_remapped, metric='cosine')
    print(f"[*] Average Silhouette Score (cosine): {sil_avg:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
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

    # Average score line
    ax.axvline(x=sil_avg, color="crimson", linestyle="--", linewidth=1.8,
               label=f"Avg = {sil_avg:.3f}")

    # Legend for conditions
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


# ── main ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",      required=True,
                   help="Path to best_clinical_valid.pt")
    p.add_argument("--reports_csv",     required=True,
                   help="Path to indiana_reports.csv")
    p.add_argument("--projections_csv", required=True,
                   help="Path to indiana_projections.csv")
    p.add_argument("--img_dir",         required=True,
                   help="Directory containing .dcm.png images")
    p.add_argument("--out_dir",         default="plots")
    p.add_argument("--num_samples",     type=int, default=800,
                   help="Number of images to sample")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load and merge CSVs ──────────────────────────────────────────────────
    print("[*] Loading CSVs ...")
    reports     = pd.read_csv(args.reports_csv)      # uid, MeSH, Problems, findings, impression
    projections = pd.read_csv(args.projections_csv)  # uid, filename, projection

    # Keep only Frontal views (one per study)
    frontal = (projections[projections["projection"] == "Frontal"]
               .drop_duplicates("uid")
               .reset_index(drop=True))

    # Merge to get MeSH / Problems + filename
    merged = frontal.merge(
        reports[["uid", "MeSH", "Problems", "findings", "impression"]],
        on="uid", how="left",
    )
    merged["MeSH"]     = merged["MeSH"].fillna("")
    merged["Problems"] = merged["Problems"].fillna("")
    merged["findings"] = merged["findings"].fillna("")
    merged["impression"] = merged["impression"].fillna("")

    # Assign primary clinical label
    merged["label"] = merged.apply(
        lambda r: assign_label(r["MeSH"], r["Problems"]), axis=1
    )
    print(f"[*] Dataset: {len(merged)} Frontal studies")
    print("[*] Label distribution:")
    print(merged["label"].value_counts().to_string())

    # 2. Build dataset ─────────────────────────────────────────────────────────
    from torchvision import transforms as T
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    dataset = FrontalDataset(merged, args.img_dir, transform)

    # 3. Load model ────────────────────────────────────────────────────────────
    model = load_model(args.checkpoint)

    # 4. Extract embeddings ────────────────────────────────────────────────────
    embeddings, labels = extract_embeddings(model, dataset, args.num_samples)

    # 5. Convert clinical labels → numeric cluster IDs ─────────────────────────
    # This measures: "Does the embedding space separate clinical conditions?"
    unique_labels = sorted(set(labels))
    label_to_id   = {l: i for i, l in enumerate(unique_labels)}
    cluster_ids   = np.array([label_to_id[l] for l in labels])
    print(f"[*] Using {len(unique_labels)} clinical groups as clusters:")
    for l, i in label_to_id.items():
        print(f"    [{i}] {l}: {(cluster_ids == i).sum()} samples")

    # 6. Draw Silhouette Plot ──────────────────────────────────────────────────
    save_path = os.path.join(args.out_dir, "silhouette_plot.png")
    plot_silhouette(embeddings, cluster_ids, labels, save_path)


if __name__ == "__main__":
    main()
