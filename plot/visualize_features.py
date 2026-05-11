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
from torch.utils.data import DataLoader, Subset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')

# Import your existing modules
import train_proposed as tp
from src.dataset import get_val_transform

# ──────────────────────────────────────────────────────────────────────────────
# Constants & Paths
# ──────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid", palette="muted")
COLOR_PALETTE = "tab10"   # For clusters

# ──────────────────────────────────────────────────────────────────────────────
# Helper: extract ground truth labels (using your existing function)
# ──────────────────────────────────────────────────────────────────────────────
def get_keywords_for_col(col_name):
    col_lower = col_name.lower()
    if col_lower == "no finding": return ['no indexing', 'normal', 'negative']
    if col_lower == "pleural effusion": return ['effusion']
    if col_lower == "enlarged cardiomediastinum": return ['mediastinum/enlarged']
    return [col_lower]

def extract_labels_from_mesh(mesh_str: str, problems_str: str) -> np.ndarray:
    combined = (str(mesh_str) + ';' + str(problems_str)).lower()
    vec = np.zeros(len(tp.PATH_COLS), dtype=np.float32)
    for i, col in enumerate(tp.PATH_COLS):
        keywords = get_keywords_for_col(col)
        if any(kw in combined for kw in keywords):
            vec[i] = 1.0
    if vec.sum() == 0:
        if 'No Finding' in tp.PATH_COLS:
            vec[tp.PATH_COLS.index('No Finding')] = 1.0
        elif 'Normal' in tp.PATH_COLS:
            vec[tp.PATH_COLS.index('Normal')] = 1.0
    return vec

def build_label_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply extract_labels_from_mesh to the whole reports dataframe and return
    a dataframe with uid + one binary column per tp.PATH_COLS entry.
    """
    rows = df.apply(
        lambda r: extract_labels_from_mesh(r.get('MeSH', ''), r.get('Problems', '')),
        axis=1,
    )
    label_df = pd.DataFrame(rows.tolist(), columns=tp.PATH_COLS)
    label_df.insert(0, 'uid', df['uid'].values)
    return label_df

def get_primary_label(labels: np.ndarray, label_names) -> str:
    """Return the primary condition for a multi-label vector (simplified: pick first positive)"""
    pos = np.where(labels > 0.5)[0]
    if len(pos) == 0:
        return "Normal"
    # Use a heuristic priority (you can adjust)
    # For simplicity, return the first positive
    return label_names[pos[0]]

# ──────────────────────────────────────────────────────────────────────────────
# Model & feature extraction
# ──────────────────────────────────────────────────────────────────────────────
def load_model(checkpoint_path):
    print(f"[*] Loading model from {checkpoint_path}")
    model = tp.StudyMedicalSwinBERT().to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def extract_features(model, loader):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tp.TEXT_MODEL)
    img_embs, txt_embs, labels = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features"):
            images = batch["images"].to(DEVICE)
            view_mask = batch["view_mask"].to(DEVICE)
            view_type_ids = batch["view_type_ids"].to(DEVICE)
            tok = tokenizer(batch["caption"], padding="max_length", truncation=True,
                            max_length=tp.TEXT_MAX_LEN, return_tensors="pt").to(DEVICE)
            img_emb, txt_emb, _ = model(images, view_mask, view_type_ids,
                                        tok["input_ids"], tok["attention_mask"])
            img_embs.append(img_emb.cpu().numpy())
            txt_embs.append(txt_emb.cpu().numpy())
            labels.append(batch["labels"].cpu().numpy())
    return np.concatenate(img_embs), np.concatenate(txt_embs), np.concatenate(labels)

# ──────────────────────────────────────────────────────────────────────────────
# Clustering and evaluation
# ──────────────────────────────────────────────────────────────────────────────
def find_optimal_k(embeddings, max_k=20):
    inertias = []
    K_range = range(2, max_k+1)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        inertias.append(kmeans.inertia_)
    # Elbow plot
    plt.figure(figsize=(8,5))
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal K')
    plt.grid(True)
    plt.savefig('plots/elbow_curve.png', dpi=300)
    plt.close()
    # Heuristic: return K where inertia drop slows (can be automated, but user chooses)
    # For simplicity, return the K that gives best silhouette later
    return K_range[np.argmin(np.diff(inertias, 2)) + 1] if len(inertias)>2 else 5

def perform_clustering(embeddings, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_
    return cluster_labels, centroids

def compute_silhouette(embeddings, cluster_labels):
    sil_avg = silhouette_score(embeddings, cluster_labels)
    sil_per_sample = silhouette_samples(embeddings, cluster_labels)
    return sil_avg, sil_per_sample

def plot_silhouette(sil_per_sample, cluster_labels, n_clusters, save_path):
    fig, ax = plt.subplots(figsize=(10, 7))
    y_lower = 10
    for i in range(n_clusters):
        ith_sil = sil_per_sample[cluster_labels == i]
        ith_sil.sort()
        size = len(ith_sil)
        y_upper = y_lower + size
        color = plt.cm.tab10(i / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_sil, facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + size/2, f'Cluster {i}', fontsize=10)
        y_lower = y_upper + 10
    ax.axvline(x=np.mean(sil_per_sample), color='red', linestyle='--', label=f'Average = {np.mean(sil_per_sample):.3f}')
    ax.set_xlabel('Silhouette coefficient')
    ax.set_ylabel('Cluster')
    ax.set_title('Silhouette Plot')
    ax.legend()
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# Dimensionality reduction plots (colored by cluster)
# ──────────────────────────────────────────────────────────────────────────────
def plot_reduction(embeddings, labels, title, filename, method='tsne', dims=2):
    if method == 'tsne':
        reducer = TSNE(n_components=dims, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    else:
        reducer = umap.UMAP(n_components=dims, n_neighbors=15, min_dist=0.1, random_state=42)
    reduced = reducer.fit_transform(embeddings)
    if dims == 2:
        df = pd.DataFrame({'x': reduced[:,0], 'y': reduced[:,1], 'cluster': labels.astype(str)})
        plt.figure(figsize=(12,9))
        sns.scatterplot(data=df, x='x', y='y', hue='cluster', palette=COLOR_PALETTE, alpha=0.7, s=40, edgecolor='w')
        plt.title(title, fontsize=15)
        plt.legend(title='Cluster')
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12,9))
        ax = fig.add_subplot(111, projection='3d')
        for i in np.unique(labels):
            mask = labels == i
            ax.scatter(reduced[mask,0], reduced[mask,1], reduced[mask,2], label=f'Cluster {i}', alpha=0.6, s=30)
        ax.set_title(title)
        ax.legend()
        plt.savefig(filename, dpi=300)
        plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# Negative sampling based on distance to cluster centroid
# ──────────────────────────────────────────────────────────────────────────────
def negative_sampling(embeddings, true_labels, cluster_labels, centroids, pos_per_cluster_ratio=1.0, neg_ratio=2.0):
    """
    For each cluster, treat samples of the majority class (or a chosen positive condition) as positive,
    and others as negative. Select negatives furthest from centroid (easy negatives) or closest to centroid (hard negatives)?
    Here we select negatives that are closest to centroid of their own cluster -> "hard negatives".
    Adjust logic according to your method.
    """
    # For demonstration, we assume positive class is "Cardiomegaly" (or any condition). 
    # You can modify to use ground truth labels.
    # Here we'll use the primary label from multi-label vectors as proxy.
    # In practice, you should define a binary target (e.g., presence of any pathology vs no finding).
    pos_mask = (true_labels[:, tp.PATH_COLS.index('Cardiomegaly')] > 0.5)  # example
    neg_mask = ~pos_mask

    selected_neg_indices = []
    discarded_neg_indices = []
    # For each cluster, select negatives that are closest to centroid (hard negatives)
    for c in np.unique(cluster_labels):
        cluster_pts = np.where(cluster_labels == c)[0]
        # within cluster, get positive and negative indices
        pos_in_cluster = cluster_pts[pos_mask[cluster_pts]]
        neg_in_cluster = cluster_pts[neg_mask[cluster_pts]]
        if len(neg_in_cluster) == 0:
            continue
        # distances to centroid of this cluster
        centroid = centroids[c]
        dists = np.linalg.norm(embeddings[neg_in_cluster] - centroid, axis=1)
        # select top k closest (hard negatives) – you can tune k
        k = min(int(len(neg_in_cluster) * neg_ratio), len(neg_in_cluster))
        closest_idx = np.argsort(dists)[:k]
        selected = neg_in_cluster[closest_idx]
        discarded = np.setdiff1d(neg_in_cluster, selected)
        selected_neg_indices.extend(selected)
        discarded_neg_indices.extend(discarded)
    return pos_mask, selected_neg_indices, discarded_neg_indices

def plot_negative_sampling_scatter(embeddings_2d, pos_mask, selected_neg, discarded_neg, save_path):
    plt.figure(figsize=(10,8))
    plt.scatter(embeddings_2d[~pos_mask,0], embeddings_2d[~pos_mask,1], c='lightgray', s=20, label='Negative (not selected)', alpha=0.5)
    plt.scatter(embeddings_2d[pos_mask,0], embeddings_2d[pos_mask,1], c='blue', s=40, label='Positive', edgecolor='k')
    plt.scatter(embeddings_2d[selected_neg,0], embeddings_2d[selected_neg,1], c='red', s=40, label='Selected Negative', edgecolor='k')
    plt.title('Negative Sampling Strategy')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_distance_distribution(embeddings, centroids, cluster_labels, pos_mask, selected_neg, discarded_neg, save_path):
    # Compute distance to own cluster centroid for each sample
    dist_to_centroid = np.zeros(len(embeddings))
    for i, c in enumerate(cluster_labels):
        dist_to_centroid[i] = np.linalg.norm(embeddings[i] - centroids[c])
    data = [dist_to_centroid[pos_mask], dist_to_centroid[selected_neg], dist_to_centroid[discarded_neg]]
    labels = ['Positive', 'Selected Negative', 'Discarded Negative']
    plt.figure(figsize=(8,6))
    plt.hist(data, bins=30, alpha=0.6, label=labels)
    plt.xlabel('Distance to cluster centroid')
    plt.ylabel('Frequency')
    plt.title('Distribution of distances by sample type')
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# Similarity heatmap
# ──────────────────────────────────────────────────────────────────────────────
def plot_similarity_heatmap(embeddings, true_labels, cluster_labels, num_samples=200, save_path='plots/similarity_heatmap.png'):
    # Subsample for clarity
    idx = np.random.choice(len(embeddings), min(num_samples, len(embeddings)), replace=False)
    sub_emb = embeddings[idx]
    sub_labels = true_labels[idx]
    sub_clusters = cluster_labels[idx]
    # Cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(sub_emb)
    # Sort by cluster then by true label (optional)
    order = np.argsort(sub_clusters)
    sim_sorted = sim[order][:, order]
    plt.figure(figsize=(12,10))
    sns.heatmap(sim_sorted, cmap='coolwarm', xticklabels=False, yticklabels=False, cbar_kws={'label': 'Cosine similarity'})
    # Add cluster boundary lines
    boundaries = np.cumsum(np.bincount(sub_clusters[order]))
    for b in boundaries[:-1]:
        plt.axhline(b, color='black', linewidth=1)
        plt.axvline(b, color='black', linewidth=1)
    plt.title('Similarity Heatmap (samples sorted by cluster)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# Evaluation metrics after training classifier using selected negatives
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_classifier(embeddings, pos_mask, selected_neg, test_size=0.3):
    # Create balanced dataset: positives + selected negatives
    pos_idx = np.where(pos_mask)[0]
    neg_idx = np.array(selected_neg)
    X = np.vstack([embeddings[pos_idx], embeddings[neg_idx]])
    y = np.hstack([np.ones(len(pos_idx)), np.zeros(len(neg_idx))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred_prob = clf.predict_proba(X_test)[:,1]
    y_pred = clf.predict(X_test)
    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    # PR
    prec, rec, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(rec, prec)
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    axes[0].plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    axes[0].plot([0,1], [0,1], 'k--')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend()

    axes[1].plot(rec, prec, label=f'PR AUC = {pr_auc:.3f}')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend()

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2])
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('True')
    axes[2].set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('plots/evaluation_metrics.png', dpi=300)
    plt.close()
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='plot/best_clinical_valid.pt')
    parser.add_argument('--csv_path', default=tp.CSV_PATH)
    parser.add_argument('--img_dir', default=tp.IMG_DIR)
    parser.add_argument('--reports_csv', default='indiana_reports.csv')
    parser.add_argument('--projections_csv', default='indiana_projections.csv')
    parser.add_argument('--out_dir', default='plots')
    parser.add_argument('--num_samples', type=int, default=800)
    parser.add_argument('--n_clusters', type=int, default=None, help='Set fixed K, else auto-detect')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load data and build ground truth labels
    print("[*] Loading IU X-ray CSVs...")
    reports_df = pd.read_csv(args.reports_csv)
    # Build label dataframe using our local function
    label_df = build_label_dataframe(reports_df)
    # If not, use the one defined here (but note: need to replicate)
    # We'll use the function we defined above
    # Actually, your original code had 'build_label_dataframe' inside the script; we'll use that.

    # For brevity, we assume you have a merged dataset with multi-hot labels.
    # I'll simulate the main dataset loading.
    # But to avoid duplication, I'll quickly implement the dataset creation as in your original code.
    from torchvision import transforms as T
    transform = T.Compose([
        T.Resize((384,384)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    df = pd.read_csv(args.csv_path)
    
    # Column mapping to ensure compatibility with Kaggle names
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
            df[new_col] = df[old_col]
            
    if 'patient_id' not in df.columns and 'image_id' in df.columns:
        df['patient_id'] = df['image_id'].apply(lambda x: str(x).split('_')[0])

    # Use your original dataset class
    dataset = tp.StudyIUXrayDataset(df, args.img_dir, transform, train_mode=False)
    indices = np.random.choice(len(dataset), min(args.num_samples, len(dataset)), replace=False)
    loader = DataLoader(Subset(dataset, indices), batch_size=16, shuffle=False, collate_fn=tp.collate_study)

    # 2. Load model and extract features
    model = load_model(args.checkpoint)
    img_embs, txt_embs, labels = extract_features(model, loader)  # labels shape (N, num_classes)

    # Use only image embeddings for clustering (or combine with text)
    embeddings = img_embs  # you could concat: np.hstack([img_embs, txt_embs])

    # 3. Determine optimal K and perform clustering
    if args.n_clusters is None:
        optimal_k = find_optimal_k(embeddings, max_k=15)
        print(f"[*] Optimal K estimated: {optimal_k}")
    else:
        optimal_k = args.n_clusters
    cluster_labels, centroids = perform_clustering(embeddings, optimal_k)

    # 4. Silhouette analysis
    sil_avg, sil_per_sample = compute_silhouette(embeddings, cluster_labels)
    print(f"[*] Average silhouette score: {sil_avg:.4f}")
    plot_silhouette(sil_per_sample, cluster_labels, optimal_k, os.path.join(args.out_dir, 'silhouette_plot.png'))

    # 5. Dimensionality reduction plots (by cluster)
    plot_reduction(embeddings, cluster_labels, 't-SNE (2D) colored by cluster', os.path.join(args.out_dir, 'tsne_clusters_2d.png'), method='tsne', dims=2)
    plot_reduction(embeddings, cluster_labels, 'UMAP (2D) colored by cluster', os.path.join(args.out_dir, 'umap_clusters_2d.png'), method='umap', dims=2)

    # 6. Negative sampling (using ground truth from multi-hot labels)
    # Need to define a positive condition. Example: any pathology (exclude 'No Finding')
    no_finding_idx = tp.PATH_COLS.index('No Finding') if 'No Finding' in tp.PATH_COLS else None
    if no_finding_idx is not None:
        pos_any = (labels.sum(axis=1) - labels[:, no_finding_idx]) > 0
    else:
        pos_any = labels.sum(axis=1) > 0
    # Let's use 'pos_any' as positive label (presence of any pathology)
    pos_mask = pos_any
    # Perform negative sampling
    pos_mask_np, selected_neg, discarded_neg = negative_sampling(embeddings, labels, cluster_labels, centroids,
                                                                  pos_per_cluster_ratio=1.0, neg_ratio=1.5)
    # Get 2D coordinates for visualization (using UMAP)
    reducer = umap.UMAP(random_state=42)
    emb_2d = reducer.fit_transform(embeddings)
    plot_negative_sampling_scatter(emb_2d, pos_mask_np, selected_neg, discarded_neg,
                                   os.path.join(args.out_dir, 'negative_sampling_scatter.png'))
    plot_distance_distribution(embeddings, centroids, cluster_labels, pos_mask_np, selected_neg, discarded_neg,
                               os.path.join(args.out_dir, 'distance_distribution.png'))

    # 7. Similarity heatmap
    plot_similarity_heatmap(embeddings, labels, cluster_labels, num_samples=300,
                            save_path=os.path.join(args.out_dir, 'similarity_heatmap.png'))

    # 8. Classifier evaluation using selected negatives
    evaluate_classifier(embeddings, pos_mask_np, selected_neg, test_size=0.3)

    print(f"[✔] All plots and evaluations saved to {args.out_dir}")

if __name__ == '__main__':
    main()