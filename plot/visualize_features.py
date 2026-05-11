import sys
import os

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score
import umap
from collections import Counter
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

import train_proposed as tp
from src.dataset import get_val_transform

# ─── Constants ────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "plot/best_clinical_valid.pt"
OUTPUT_DIR      = "plot"
BATCH_SIZE      = 16
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid", palette="muted")
COLOR_PALETTE = "Spectral"

# ─── Target labels (Inherited from training config) ────────────────────────────
PATH_COLS = tp.PATH_COLS

# ─── Dynamic Keyword mapping ──────────────────────────────────────────────────
def get_keywords_for_col(col_name):
    """Dynamically generate keywords based on column name."""
    col_lower = col_name.lower()
    if col_lower == "no finding": return ['no indexing', 'normal', 'negative']
    if col_lower == "pleural effusion": return ['effusion']
    if col_lower == "enlarged cardiomediastinum": return ['mediastinum/enlarged']
    # Default: use the column name itself as the keyword
    return [col_lower]

def extract_labels_from_mesh(mesh_str: str, problems_str: str) -> np.ndarray:
    combined = (str(mesh_str) + ';' + str(problems_str)).lower()
    vec = np.zeros(len(PATH_COLS), dtype=np.float32)

    for i, col in enumerate(PATH_COLS):
        keywords = get_keywords_for_col(col)
        if any(kw in combined for kw in keywords):
            vec[i] = 1.0

    if vec.sum() == 0:
        # Fallback to 'No Finding' if it exists in PATH_COLS
        if 'No Finding' in PATH_COLS:
            vec[PATH_COLS.index('No Finding')] = 1.0
        elif 'Normal' in PATH_COLS:
            vec[PATH_COLS.index('Normal')] = 1.0

    return vec


def build_label_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply extract_labels_from_mesh to the whole reports dataframe and return
    a dataframe with uid + one binary column per PATH_COLS entry.
    """
    rows = df.apply(
        lambda r: extract_labels_from_mesh(r.get('MeSH', ''), r.get('Problems', '')),
        axis=1,
    )
    label_df = pd.DataFrame(rows.tolist(), columns=PATH_COLS)
    label_df.insert(0, 'uid', df['uid'].values)
    return label_df


# ─── Model helpers ────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str):
    print(f"[*] Loading model from {checkpoint_path}...")
    model = tp.StudyMedicalSwinBERT().to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def get_features(model, loader):
    print("[*] Extracting embeddings from dataset...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tp.TEXT_MODEL)

    all_img_embs, all_txt_embs, all_labels = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Feature Extraction"):
            images        = batch["images"].to(DEVICE)
            view_mask     = batch["view_mask"].to(DEVICE)
            view_type_ids = batch["view_type_ids"].to(DEVICE)

            tok = tokenizer(
                batch["caption"],
                padding="max_length",
                truncation=True,
                max_length=tp.TEXT_MAX_LEN,
                return_tensors="pt",
            ).to(DEVICE)

            img_emb, txt_emb, *_ = model(
                images, view_mask, view_type_ids,
                tok["input_ids"], tok["attention_mask"],
            )

            all_img_embs.append(img_emb.cpu().numpy())
            all_txt_embs.append(txt_emb.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())

    return (
        np.concatenate(all_img_embs),
        np.concatenate(all_txt_embs),
        np.concatenate(all_labels),
    )


# ─── Label helpers ────────────────────────────────────────────────────────────

def get_primary_labels(labels: np.ndarray, label_names) -> list[str]:
    """
    For each sample pick the *highest-confidence* positive label.

    Unlike the original code, we rank by label index priority so that
    more-specific conditions (Cardiomegaly, Pneumonia, …) take precedence
    over the catch-all 'No Finding' / 'Normal' when both are set.

    Priority order (higher index = higher priority):
        No Finding < Normal < Fracture < Atelectasis < Cardiomegaly
        < Pneumonia < Consolidation < Pleural Effusion < Pneumothorax < Edema
    """
    PRIORITY = list(range(len(label_names)))   # last index wins

    primary = []
    for row in labels:
        pos = np.where(row > 0.5)[0]
        if len(pos) == 0:
            primary.append("Normal")
        elif len(pos) == 1:
            primary.append(label_names[pos[0]])
        else:
            # pick the one with highest priority (= highest index in PATH_COLS)
            best = max(pos, key=lambda i: PRIORITY[i])
            primary.append(label_names[best])
    return primary


# ─── Plot functions ───────────────────────────────────────────────────────────

def plot_pca_biplot(features, labels, label_names, title, filename):
    print(f"[+] Generating PCA Biplot: {filename}...")
    pca      = PCA(n_components=2)
    pca_feat = pca.fit_transform(features)
    var_exp  = pca.explained_variance_ratio_

    df = pd.DataFrame({
        'PC1':       pca_feat[:, 0],
        'PC2':       pca_feat[:, 1],
        'Condition': get_primary_labels(labels, label_names),
    })

    plt.figure(figsize=(12, 9))
    sns.scatterplot(
        data=df, x='PC1', y='PC2', hue='Condition',
        palette=COLOR_PALETTE, alpha=0.8, edgecolor='w', s=60,
    )
    plt.title(
        f"{title}\nVariance Explained: PC1={var_exp[0]:.2%}, PC2={var_exp[1]:.2%}",
        fontsize=15, fontweight='bold', pad=20,
    )
    plt.xlabel(f"Principal Component 1 ({var_exp[0]:.2%})", fontsize=12)
    plt.ylabel(f"Principal Component 2 ({var_exp[1]:.2%})", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Clinical Condition")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_dimensionality_reduction(features, labels, label_names,
                                  method='tsne', dims=2, title="", filename=""):
    print(f"[+] Generating {method.upper()} ({dims}D): {filename}...")

    if method == 'tsne':
        reducer = TSNE(
            n_components=dims, random_state=42,
            perplexity=30, init='pca', learning_rate='auto',
        )
    else:
        reducer = umap.UMAP(
            n_components=dims, n_neighbors=15,
            min_dist=0.1, random_state=42,
        )

    feat_reduced = reducer.fit_transform(features)
    conditions   = get_primary_labels(labels, label_names)

    if dims == 2:
        df = pd.DataFrame({
            'Dim 1':     feat_reduced[:, 0],
            'Dim 2':     feat_reduced[:, 1],
            'Condition': conditions,
        })
        plt.figure(figsize=(12, 9))
        sns.scatterplot(
            data=df, x='Dim 1', y='Dim 2', hue='Condition',
            palette=COLOR_PALETTE, alpha=0.8, s=60, edgecolor='w',
        )
        plt.title(title, fontsize=15, fontweight='bold', pad=20)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Clinical Condition")
    else:
        fig = plt.figure(figsize=(12, 9))
        ax  = fig.add_subplot(111, projection='3d')
        unique_labels = sorted(set(conditions))

        for label in unique_labels:
            mask = np.array(conditions) == label
            ax.scatter(
                feat_reduced[mask, 0], feat_reduced[mask, 1], feat_reduced[mask, 2],
                label=label, alpha=0.7, s=40, edgecolors='w', linewidth=0.5,
            )
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Clinical Condition")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_silhouette(features, labels, label_names, title, filename):
    print(f"[+] Generating Silhouette Plot: {filename}...")
    
    conditions = get_primary_labels(labels, label_names)
    unique_labels = sorted(set(conditions))
    
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_to_idx[l] for l in conditions])
    
    counts = Counter(conditions)
    valid_indices = [i for i, label in enumerate(conditions) if counts[label] > 1]
    
    if not valid_indices:
        print("[!] Warning: Not enough samples per cluster for Silhouette analysis.")
        return
        
    X = features[valid_indices]
    y = y[valid_indices]
    y_labels = [conditions[i] for i in valid_indices]
    
    avg_score = silhouette_score(X, y)
    sample_values = silhouette_samples(X, y)
    
    plt.figure(figsize=(12, 10))
    y_lower = 10
    colors = sns.color_palette(COLOR_PALETTE, len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        ith_cluster_values = sample_values[np.array(y_labels) == label]
        ith_cluster_values.sort()
        
        size_cluster_i = ith_cluster_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = colors[i]
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, label)
        y_lower = y_upper + 10
        
    plt.axvline(x=avg_score, color="red", linestyle="--", label=f"Avg Score: {avg_score:.3f}")
    plt.title(f"{title}\nAverage Silhouette Score: {avg_score:.3f}", fontsize=15, fontweight='bold')
    plt.xlabel("Silhouette Coefficient Values")
    plt.ylabel("Cluster Label")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize latent space of IU-Xray model"
    )
    parser.add_argument("--checkpoint",   default=CHECKPOINT_PATH)
    parser.add_argument("--csv_path",     default=tp.CSV_PATH)
    parser.add_argument("--img_dir",      default=tp.IMG_DIR)
    parser.add_argument("--out_dir",      default=OUTPUT_DIR)
    parser.add_argument("--num_samples",  type=int, default=800)
    # Paths to the raw IU X-ray CSVs for proper label extraction
    parser.add_argument(
        "--reports_csv",
        default="indiana_reports.csv",
        help="Path to indiana_reports.csv (for MeSH-based label extraction)",
    )
    parser.add_argument(
        "--projections_csv",
        default="indiana_projections.csv",
        help="Path to indiana_projections.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── 1. Load raw IU X-ray CSVs and build proper labels ─────────────────────
    print("[*] Loading IU X-ray CSVs for label extraction...")
    reports_df     = pd.read_csv(args.reports_csv)
    projections_df = pd.read_csv(args.projections_csv)

    # Build uid → label vector mapping using MeSH/Problems
    label_df = build_label_dataframe(reports_df)
    print("[*] Label distribution from MeSH extraction:")
    print(label_df[PATH_COLS].sum().sort_values(ascending=False).to_string())

    # ── 2. Merge labels into the main dataset CSV ──────────────────────────────
    print("\n[*] Preparing main dataset CSV...")
    if not os.path.exists(args.csv_path):
        print(f"[!] Error: CSV file not found at {args.csv_path}")
        return

    df = pd.read_csv(args.csv_path)

    # Column name normalisation (handles common Kaggle naming variants)
    column_mapping = {
        'uid':        'patient_id',
        'filename':   'image_id',
        'findings':   'org_caption',
        'impression': 'org_caption',
        'report':     'org_caption',
        'caption':    'org_caption',
    }
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            print(f"[*] Mapping column '{old_col}' → '{new_col}'")
            df[new_col] = df[old_col]

    if 'patient_id' not in df.columns:
        if 'image_id' in df.columns:
            df['patient_id'] = df['image_id'].apply(lambda x: str(x).split('_')[0])
        else:
            print(f"[!] Error: required columns missing. Found: {list(df.columns)}")
            return

    # ── 3. Attach labels from label_df (keyed on uid / patient_id) ────────────
    # Prefer joining via uid when available; fall back to patient_id
    uid_col_main = 'uid' if 'uid' in df.columns else 'patient_id'
    merged = df.merge(
        label_df.rename(columns={'uid': uid_col_main}),
        on=uid_col_main,
        how='left',
    )

    # Fill labels that couldn't be matched (e.g. uid mismatch) with No Finding
    for col in PATH_COLS:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = merged[col].fillna(0.0)

    # Rows still without any label → mark as No Finding
    no_label_mask = merged[PATH_COLS].sum(axis=1) == 0
    merged.loc[no_label_mask, 'No Finding'] = 1.0
    print(f"[*] Samples without matched label (defaulted to No Finding): {no_label_mask.sum()}")

    df = merged  # use enriched dataframe going forward

    # ── 4. Dataset & DataLoader ────────────────────────────────────────────────
    print(f"\n[*] Building dataset (image size 384×384)...")
    from torchvision import transforms as T
    transform = T.Compose([
        T.Resize((384, 384), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = tp.StudyIUXrayDataset(
        df=df,
        img_dir=args.img_dir,
        transform=transform,
        train_mode=False,
    )

    num_samples    = min(len(dataset), args.num_samples)
    indices        = np.random.choice(len(dataset), num_samples, replace=False)
    subset_dataset = torch.utils.data.Subset(dataset, indices)
    loader         = DataLoader(
        subset_dataset, batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=tp.collate_study,
    )

    # ── 5. Load model ─────────────────────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        print(f"[!] Error: Checkpoint not found at {args.checkpoint}")
        return
    model = load_model(args.checkpoint)

    # ── 6. Extract features ───────────────────────────────────────────────────
    img_embs, txt_embs, labels = get_features(model, loader)
    label_names = np.array(PATH_COLS)

    # ── 7. Generate visualizations ────────────────────────────────────────────
    print("\n" + "=" * 40)
    print("  GENERATING VISUALIZATIONS")
    print("=" * 40)

    out = args.out_dir

    # PCA
    plot_pca_biplot(img_embs, labels, label_names,
                    "PCA Analysis - Image Embeddings",
                    os.path.join(out, "pca_img_2d.png"))
    plot_pca_biplot(txt_embs, labels, label_names,
                    "PCA Analysis - Text Embeddings",
                    os.path.join(out, "pca_txt_2d.png"))

    # t-SNE 2D & 3D
    plot_dimensionality_reduction(img_embs, labels, label_names,
                                  method='tsne', dims=2,
                                  title="t-SNE Visualization (2D) - Image Latent Space",
                                  filename=os.path.join(out, "tsne_img_2d.png"))
    plot_dimensionality_reduction(img_embs, labels, label_names,
                                  method='tsne', dims=3,
                                  title="t-SNE Visualization (3D) - Image Latent Space",
                                  filename=os.path.join(out, "tsne_img_3d.png"))

    # UMAP 2D & 3D
    plot_dimensionality_reduction(img_embs, labels, label_names,
                                  method='umap', dims=2,
                                  title="UMAP Visualization (2D) - Image Latent Space",
                                  filename=os.path.join(out, "umap_img_2d.png"))
    plot_dimensionality_reduction(img_embs, labels, label_names,
                                  method='umap', dims=3,
                                  title="UMAP Visualization (3D) - Image Latent Space",
                                  filename=os.path.join(out, "umap_img_3d.png"))

    # Silhouette Plot
    plot_silhouette(img_embs, labels, label_names,
                    "Silhouette Analysis - Image Latent Space",
                    os.path.join(out, "silhouette_img.png"))

    print(f"\n[✔] All plots saved to: {out}")


if __name__ == "__main__":
    main()