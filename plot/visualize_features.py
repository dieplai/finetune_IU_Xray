import sys
import os

# Ensure the root directory is in sys.path so we can import train_proposed and src
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
# Also check if we are running from inside the 'plot' folder
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
import umap
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

# Import model and dataset from the training script
import train_proposed as tp
from src.dataset import get_val_transform

# Constants
CHECKPOINT_PATH = "plot/best_clinical_valid.pt"
OUTPUT_DIR = "plot"
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Professional Plot Styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid", palette="muted")
COLOR_PALETTE = "Spectral" # Vibrant and professional

def load_model(checkpoint_path):
    print(f"[*] Loading model from {checkpoint_path}...")
    model = tp.StudyMedicalSwinBERT().to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def get_features(model, loader):
    print("[*] Extracting embeddings from dataset...")
    all_img_embs = []
    all_txt_embs = []
    all_labels = []
    
    # We need the tokenizer to process captions
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tp.TEXT_MODEL)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Feature Extraction"):
            images = batch["images"].to(DEVICE)
            view_mask = batch["view_mask"].to(DEVICE)
            view_type_ids = batch["view_type_ids"].to(DEVICE)
            
            tok = tokenizer(
                batch["caption"],
                padding="max_length",
                truncation=True,
                max_length=tp.TEXT_MAX_LEN,
                return_tensors="pt",
            ).to(DEVICE)
            
            img_emb, txt_emb, _, _, _, _, _, _, _ = model(
                images,
                view_mask,
                view_type_ids,
                tok["input_ids"],
                tok["attention_mask"],
            )
            
            all_img_embs.append(img_emb.cpu().numpy())
            all_txt_embs.append(txt_emb.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())
            
    return (
        np.concatenate(all_img_embs),
        np.concatenate(all_txt_embs),
        np.concatenate(all_labels)
    )

def get_primary_labels(labels, label_names):
    primary_labels = []
    for row in labels:
        pos = np.where(row > 0.5)[0]
        if len(pos) > 0:
            primary_labels.append(label_names[pos[0]])
        else:
            primary_labels.append("Normal")
    return primary_labels

def plot_pca_biplot(features, labels, label_names, title, filename):
    print(f"[+] Generating PCA Biplot: {filename}...")
    pca = PCA(n_components=2)
    pca_feat = pca.fit_transform(features)
    var_exp = pca.explained_variance_ratio_
    
    df = pd.DataFrame({
        'PC1': pca_feat[:, 0],
        'PC2': pca_feat[:, 1],
        'Condition': get_primary_labels(labels, label_names)
    })
    
    plt.figure(figsize=(12, 9))
    scatter = sns.scatterplot(
        data=df, x='PC1', y='PC2', hue='Condition', 
        palette=COLOR_PALETTE, alpha=0.8, edgecolor='w', s=60
    )
    
    plt.title(f"{title}\nVariance Explained: PC1={var_exp[0]:.2%}, PC2={var_exp[1]:.2%}", 
              fontsize=15, fontweight='bold', pad=20)
    plt.xlabel(f"Principal Component 1 ({var_exp[0]:.2%})", fontsize=12)
    plt.ylabel(f"Principal Component 2 ({var_exp[1]:.2%})", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Clinical Condition")
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_dimensionality_reduction(features, labels, label_names, method='tsne', dims=2, title="", filename=""):
    print(f"[+] Generating {method.upper()} ({dims}D): {filename}...")
    
    if method == 'tsne':
        reducer = TSNE(n_components=dims, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    else:
        reducer = umap.UMAP(n_components=dims, n_neighbors=15, min_dist=0.1, random_state=42)
        
    feat_reduced = reducer.fit_transform(features)
    conditions = get_primary_labels(labels, label_names)
    
    if dims == 2:
        df = pd.DataFrame({
            'Dim 1': feat_reduced[:, 0],
            'Dim 2': feat_reduced[:, 1],
            'Condition': conditions
        })
        plt.figure(figsize=(12, 9))
        sns.scatterplot(
            data=df, x='Dim 1', y='Dim 2', hue='Condition', 
            palette=COLOR_PALETTE, alpha=0.8, s=60, edgecolor='w'
        )
        plt.title(title, fontsize=15, fontweight='bold', pad=20)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Clinical Condition")
    else:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        unique_labels = sorted(list(set(conditions)))
        colors = sns.color_palette(COLOR_PALETTE, len(unique_labels))
        label_to_color = dict(zip(unique_labels, colors))
        
        for label in unique_labels:
            mask = np.array(conditions) == label
            ax.scatter(
                feat_reduced[mask, 0], feat_reduced[mask, 1], feat_reduced[mask, 2],
                label=label, alpha=0.7, s=40, edgecolors='w', linewidth=0.5
            )
            
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Clinical Condition")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize latent space of IU-Xray model")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH, help="Path to .pt model")
    parser.add_argument("--csv_path", default=tp.CSV_PATH, help="Path to dataset CSV")
    parser.add_argument("--img_dir", default=tp.IMG_DIR, help="Path to image directory")
    parser.add_argument("--out_dir", default=OUTPUT_DIR, help="Output directory for plots")
    parser.add_argument("--num_samples", type=int, default=800, help="Number of samples to visualize")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. Setup Data
    print("[*] Preparing dataset and dataloader...")
    if not os.path.exists(args.csv_path):
        print(f"[!] Error: CSV file not found at {args.csv_path}")
        return

    df = pd.read_csv(args.csv_path)
    
    # Mapping common Kaggle column names to expected names
    column_mapping = {
        'uid': 'patient_id',
        'filename': 'image_id',
        'findings': 'org_caption',
        'impression': 'org_caption',
        'report': 'org_caption',
        'caption': 'org_caption'
    }
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            print(f"[*] Mapping column '{old_col}' to '{new_col}'")
            df[new_col] = df[old_col]

    # Preprocessing for Kaggle datasets that might lack 'patient_id'
    if 'patient_id' not in df.columns:
        if 'image_id' in df.columns:
            print("[*] 'patient_id' column missing. Attempting to parse from 'image_id'...")
            df['patient_id'] = df['image_id'].apply(lambda x: str(x).split('_')[0])
        else:
            print(f"[!] Error: Required columns not found. Available columns: {list(df.columns)}")
            return

    # Add pathology labels: Check if they exist, if not, try to extract from 'Problems'/'MeSH'
    label_names = tp.CHEXPERT_COLS
    missing_labels = [col for col in label_names if col not in df.columns]
    
    if missing_labels:
        print(f"[*] Extracting clinical labels from 'Problems' and 'MeSH' columns...")
        for col in label_names:
            if col not in df.columns:
                df[col] = 0.0 # Default
        
        # Simple keyword matching for common pathologies
        keywords = {
            "Cardiomegaly": ["cardiomegaly", "enlarged heart"],
            "Pleural Effusion": ["effusion", "pleural effusion"],
            "Edema": ["edema", "pulmonary edema"],
            "Pneumonia": ["pneumonia", "infection"],
            "Atelectasis": ["atelectasis", "collapse"],
            "Pneumothorax": ["pneumothorax", "collapsed lung"],
            "Consolidation": ["consolidation"],
            "Fracture": ["fracture", "broken"],
            "No Finding": ["normal", "no finding", "negative"]
        }
        
        for idx, row in df.iterrows():
            text = (str(row.get('Problems', '')) + " " + str(row.get('MeSH', ''))).lower()
            for col, keys in keywords.items():
                if any(k in text for k in keys) and col in df.columns:
                    df.at[idx, col] = 1.0

    print(f"[*] Visualizing with fixed image size: 384x384")
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
        train_mode=False
    )
    
    num_samples = min(len(dataset), args.num_samples)
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    subset_dataset = torch.utils.data.Subset(dataset, indices)
        
    loader = DataLoader(subset_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=tp.collate_study)
    
    # 2. Load Model
    if not os.path.exists(args.checkpoint):
        print(f"[!] Error: Checkpoint not found at {args.checkpoint}")
        return
        
    model = load_model(args.checkpoint)
    
    # 3. Extract Features
    img_embs, txt_embs, labels = get_features(model, loader)
    label_names = tp.CHEXPERT_COLS
    
    # 4. Generate Visualizations
    print("\n" + "="*30)
    print("  GENERATING VISUALIZATIONS")
    print("="*30)
    
    # PCA
    plot_pca_biplot(img_embs, labels, label_names, "PCA Analysis - Image Embeddings", os.path.join(args.out_dir, "pca_img_2d.png"))
    plot_pca_biplot(txt_embs, labels, label_names, "PCA Analysis - Text Embeddings", os.path.join(args.out_dir, "pca_txt_2d.png"))
    
    # t-SNE (2D & 3D)
    plot_dimensionality_reduction(img_embs, labels, label_names, method='tsne', dims=2, 
                                  title="t-SNE Visualization (2D) - Image Latent Space", filename=os.path.join(args.out_dir, "tsne_img_2d.png"))
    plot_dimensionality_reduction(img_embs, labels, label_names, method='tsne', dims=3, 
                                  title="t-SNE Visualization (3D) - Image Latent Space", filename=os.path.join(args.out_dir, "tsne_img_3d.png"))
    
    # UMAP (2D & 3D)
    plot_dimensionality_reduction(img_embs, labels, label_names, method='umap', dims=2, 
                                  title="UMAP Visualization (2D) - Image Latent Space", filename=os.path.join(args.out_dir, "umap_img_2d.png"))
    plot_dimensionality_reduction(img_embs, labels, label_names, method='umap', dims=3, 
                                  title="UMAP Visualization (3D) - Image Latent Space", filename=os.path.join(args.out_dir, "umap_img_3d.png"))
    
    print(f"\n[✔] Success! All plots saved to: {args.out_dir}")

if __name__ == "__main__":
    main()
