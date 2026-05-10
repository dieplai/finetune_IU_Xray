#!/usr/bin/env python3
"""
train_single_gpu.py
===================
Single-GPU version of the Hierarchical Clustering-Guided Medical VLM.

This file provides the shared baseline utilities used by train_proposed.py:
  [C1] IDF-Weighted Jaccard Similarity   - rare diseases weighted more
  [C2] Healthy Cluster                   - No Finding patients form a cluster

Model: SwinV2-Base (384x384) + Bio_ClinicalBERT + MLP projection heads
Data : v8_clean.csv  (7,322 rows, 3,772 patients, 3,328 paired F+L)
       - 14 CheXpert columns as hard labels (No Finding inclusive)
       - projection: "Frontal" / "Lateral"

Run:
  python train_single_gpu.py
  python train_single_gpu.py --batch_size 6 --grad_accum 22 --epochs 100
"""

import os
import sys
import time
import json
import math
import argparse

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from PIL import Image


# ==============================================================================
#  PATHS  (edit to match your environment)
# ==============================================================================
CSV_PATH = "data/v8_clean.csv"
IMG_DIR  = "data/images_384"
OUT_DIR  = "outputs"

# ==============================================================================
#  MODEL ARCHITECTURE
# ==============================================================================
VISION_MODEL = "microsoft/swinv2-base-patch4-window12to24-192to384-22kto1k-ft"
TEXT_MODEL   = "emilyalsentzer/Bio_ClinicalBERT"
IMG_SIZE     = 384    # SwinV2-Base native resolution; do NOT change
EMBED_DIM    = 512    # joint embedding space dimension
PROJ_HID_DIM = 1024   # projection head hidden dim
DROPOUT      = 0.2

# ==============================================================================
#  TRAINING HYPERPARAMETERS
# ==============================================================================
# Memory note: 15 GB VRAM comfortably fits BS=4.
# To match the 2-GPU (2xBS=4xGRAD_ACCUM=16 = effective 128) experiment:
#   single GPU -> BS=4, GRAD_ACCUM=32   (effective batch = 128)
NUM_EPOCHS   = 150
BATCH_SIZE   = 4      # per-step batch; increase only if VRAM allows
GRAD_ACCUM   = 32     # effective batch = BATCH_SIZE x GRAD_ACCUM = 128
FREEZE_EP    = 10     # epochs of heads-only warm-up before unfreezing encoders
TEXT_MAX_LEN = 128
NUM_WORKERS  = 4
SEED         = 42
EVAL_EVERY   = 5      # run full val evaluation every N epochs

LR_HEAD      = 2e-4   # projection heads (faster adaptation)
LR_ENC       = 1e-5   # SwinV2-Base backbone (conservative; ImageNet pretrained)
LR_TEXT      = 1e-4   # Bio_ClinicalBERT (needs stronger adaptation than vision)
WEIGHT_DECAY = 0.01
MAX_GRAD     = 1.0    # gradient norm clipping

# ==============================================================================
#  TASK 3: CLUSTERING-GUIDED LOSS  [C1][C2]
# ==============================================================================
CLUSTER_START  = 11   # epoch to activate IDF-Jaccard clustering (after FREEZE_EP)
CLUSTER_ALPHA  = 0.4  # blend: (1-alpha)*hard_label + alpha*idf_jaccard_soft
CLUSTER_TEMP   = 1.0  # soft label temperature (higher -> smoother boundaries)

# ==============================================================================
#  DATA SCHEMA
# ==============================================================================
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD  = (0.229, 0.224, 0.225)

# Full 14-column order as stored in v8_clean.csv
PATH_COLS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Lesion", "Lung Opacity", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]

# 12 disease dimensions used by the clustering loss (No Finding excluded;
# Enlarged Cardiomediastinum merged into Cardiomegaly via max())
CHEXPERT_COLS = [
    "Cardiomegaly", "Lung Lesion", "Lung Opacity", "Edema",
    "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]

# IU-Xray training-set label frequencies -> IDF weights for [C1]
_DISEASE_FREQ = [
    0.090, 0.138, 0.129, 0.025, 0.055, 0.022,
    0.100, 0.020, 0.057, 0.009, 0.018, 0.031,
]


# ==============================================================================
#  LOGGING
# ==============================================================================
_log_file = None

def log(msg: str) -> None:
    """Thread-safe timestamped logging to stdout + file."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _log_file is not None:
        _log_file.write(line + "\n")
        _log_file.flush()


# ==============================================================================
#  DATA
# ==============================================================================
def patient_split(
    df: pd.DataFrame,
    split: str,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Deterministic patient-level train/val/test split.

    Splits on unique patient_id so no patient leaks across sets.
    Proportions: test=10%, val=10%, train=80% (by default).
    """
    pats = df["patient_id"].astype(str).unique()
    rng  = np.random.default_rng(seed)
    # Convert to list first to avoid numpy warning on non-array shuffle
    pats = list(pats)
    rng.shuffle(pats)
    n  = len(pats)
    nt = int(n * test)
    nv = int(n * val)
    mapping = {
        "test" : set(pats[:nt]),
        "val"  : set(pats[nt : nt + nv]),
        "train": set(pats[nt + nv :]),
    }
    return df[df["patient_id"].astype(str).isin(mapping[split])].reset_index(drop=True)


def get_train_transform() -> T.Compose:
    """
    Standard augmentation pipeline for chest X-ray training.
    Images are already 384x384 from preprocessing -> no resize needed.
    """
    return T.Compose([
        # Avoid horizontal flips for chest X-rays: reports often mention left/right.
        T.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.95, 1.05)),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(IMG_MEAN, IMG_STD),
    ])


def get_val_transform() -> T.Compose:
    """Minimal transform for validation/test (no augmentation)."""
    return T.Compose([
        T.ToTensor(),
        T.Normalize(IMG_MEAN, IMG_STD),
    ])


class IUXrayDataset(Dataset):
    """
    IU-Xray dataset loader.

    Each sample returns:
      image   : (3, 384, 384) float tensor, normalized
      caption : raw clinical report string (org_caption column)
      pid     : patient_id as string
      proj    : first character of projection ('f' for Frontal, 'l' for Lateral)
      labels  : (14,) float tensor - full 14-dim CheXpert PATH_COLS labels
    """

    def __init__(self, df: pd.DataFrame, img_dir: str, transform: T.Compose):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row  = self.df.iloc[idx]
        path = os.path.join(self.img_dir, row["image_id"])
        img  = Image.open(path).convert("RGB")
        labs = np.array([row.get(c, 0) for c in PATH_COLS], dtype=np.float32)
        return {
            "image"  : self.transform(img),
            "caption": str(row["org_caption"]),
            "pid"    : str(row["patient_id"]),
            "proj"   : str(row.get("projection", "")).strip().lower()[:1],
            "labels" : torch.from_numpy(labs),   # shape (14,)
        }


def collate_fn(batch: list[dict]) -> dict:
    return {
        "image"  : torch.stack([b["image"]   for b in batch]),
        "caption": [b["caption"] for b in batch],
        "pid"    : [b["pid"]     for b in batch],
        "proj"   : [b["proj"]    for b in batch],
        "labels" : torch.stack([b["labels"]  for b in batch]),
    }


# ==============================================================================
#  MODEL
# ==============================================================================
class ProjectionHead(nn.Module):
    """
    2-layer MLP projection head: in_dim -> hid_dim -> out_dim.

    Architecture follows SimCLR/CLIP: Linear -> BN -> GELU -> Dropout -> Linear.
    Output is L2-normalized for cosine similarity computation.
    Spatial: img_proj maps SwinV2 pooler_output (1024) -> 512.
             txt_proj maps BERT [CLS] token (768) -> 512.
    """

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        x = self.net(x)
        return F.normalize(x, dim=-1) if normalize else x


class MedicalSwinBERT(nn.Module):
    """
    Dual-encoder model for medical image-text retrieval.

    Vision encoder : SwinV2-Base pretrained on ImageNet-22K at 384x384.
                     Gradient checkpointing enabled to reduce activation memory.
    Text encoder   : Bio_ClinicalBERT pretrained on clinical notes (PubMed + MIMIC-III).
                     Gradient checkpointing enabled.
    Projection     : Both modalities projected to EMBED_DIM=512 joint space.
    Temperature    : Learnable logit_scale (clamped to [log(1), log(100)]).
    """

    def __init__(self):
        super().__init__()

        # -- Vision encoder -------------------------------------------------
        self.image_encoder = AutoModel.from_pretrained(VISION_MODEL)
        # Gradient checkpointing: trades compute for memory.
        # use_reentrant=False: avoids interaction issues with custom autograd functions.
        try:
            self.image_encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            # Older transformers versions do not accept kwargs
            self.image_encoder.gradient_checkpointing_enable()

        # Detect pooler output dimension dynamically (should be 1024 for SwinV2-Base)
        with torch.no_grad():
            dummy    = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            img_dim  = self.image_encoder(pixel_values=dummy).pooler_output.shape[-1]
        log(f"  SwinV2 pooler dim: {img_dim}")

        # -- Text encoder ---------------------------------------------------
        self.text_encoder = AutoModel.from_pretrained(TEXT_MODEL)
        try:
            self.text_encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            self.text_encoder.gradient_checkpointing_enable()

        txt_dim = self.text_encoder.config.hidden_size  # 768 for BERT-base
        log(f"  BERT hidden dim: {txt_dim}")

        # -- Projection heads -----------------------------------------------
        self.img_proj = ProjectionHead(img_dim, PROJ_HID_DIM, EMBED_DIM)
        self.txt_proj = ProjectionHead(txt_dim, PROJ_HID_DIM, EMBED_DIM)

        # -- Learnable temperature ------------------------------------------
        # Initialized to CLIP default: exp(log(1/0.07)) approx 14.3
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images -> L2-normalized embeddings of shape (B, EMBED_DIM)."""
        feat = self.image_encoder(pixel_values=images).pooler_output
        return self.img_proj(feat)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode tokenized text -> L2-normalized embeddings of shape (B, EMBED_DIM)."""
        cls = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]   # [CLS] token
        return self.txt_proj(cls)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          img_emb   : (B, EMBED_DIM) normalized image embeddings
          txt_emb   : (B, EMBED_DIM) normalized text embeddings
          scale     : scalar temperature (exp of learnable logit_scale)
        """
        # Clamp temperature inside forward() so gradient always flows
        self.logit_scale.data.clamp_(math.log(1.0), math.log(100.0))
        img_emb = self.encode_image(images)
        txt_emb = self.encode_text(input_ids, attention_mask)
        scale   = self.logit_scale.exp()
        return img_emb, txt_emb, scale

    def backbone_params(self) -> list:
        """All encoder parameters (vision + text)."""
        return list(self.image_encoder.parameters()) + list(self.text_encoder.parameters())

    def head_params(self) -> list:
        """All projection head parameters + temperature."""
        return (
            list(self.img_proj.parameters())
            + list(self.txt_proj.parameters())
            + [self.logit_scale]
        )


# ==============================================================================
#  HELPERS
# ==============================================================================
def labels_to_disease_vecs(labels: torch.Tensor) -> torch.Tensor:
    """
    Convert 14-dim PATH_COLS labels -> 12-dim CHEXPERT_COLS disease vectors.

    Merge step: Enlarged Cardiomediastinum (index 1) is merged into
    Cardiomegaly (index 2) via element-wise max, then column 1 is dropped.
    No Finding (index 0) is excluded entirely - it is handled separately
    in [C2] as its own "healthy" cluster.

    Input : labels (B, 14)  - full PATH_COLS order
    Output: disease_vecs (B, 12) - CHEXPERT_COLS order
    """
    card_merged  = torch.maximum(labels[:, 1], labels[:, 2])   # (B,)
    disease_vecs = torch.cat(
        [card_merged.unsqueeze(1), labels[:, 3:]], dim=1
    )  # (B, 12)
    return disease_vecs


def _idf_weights(device: torch.device) -> torch.Tensor:
    """
    Compute IDF weights from training-set disease frequencies.
    Rare diseases (e.g., Pneumothorax 2%) get higher weight than
    common ones (e.g., Cardiomegaly 9%).
    Returns normalized weights that sum to 1.
    """
    freq = torch.tensor(_DISEASE_FREQ, dtype=torch.float32, device=device)
    idf  = torch.log(1.0 / freq.clamp(min=1e-4))
    return idf / idf.sum()


# ==============================================================================
#  LOSS FUNCTIONS
# ==============================================================================
def build_positive_mask(patient_ids: list[str], device: torch.device) -> torch.Tensor:
    """
    Build boolean mask where mask[i, j] = True iff patient_ids[i] == patient_ids[j].

    Hard positives must be defined by patient identity, not caption equality:
    many IU-Xray reports are text-identical across different patients.
    Shape: (B, B) bool tensor.
    """
    B    = len(patient_ids)
    mask = torch.zeros(B, B, dtype=torch.bool, device=device)
    for i in range(B):
        for j in range(B):
            if patient_ids[i] == patient_ids[j]:
                mask[i, j] = True
    return mask


def idf_weighted_jaccard(dv: torch.Tensor, idf_w: torch.Tensor) -> torch.Tensor:
    """
    [C1] IDF-Weighted Jaccard similarity for binary disease label vectors.

    J_w(A, B) = sum_k w_k * (A_k AND B_k) / sum_k w_k * (A_k OR B_k)

    - A_k AND B_k (intersection): both patients have disease k
    - A_k OR  B_k (union)       : at least one patient has disease k
    - w_k = IDF weight (rare diseases contribute more)

    Implementation trick: for binary vectors, (A*w)*(B*w) approximates
    weighted intersection via sqrt(w) scaling:
      inter = sum_k w_k * A_k * B_k  via  (A*sqrt(w)) @ (B*sqrt(w))^T

    Input:
      dv    : (B, 12) binary disease vectors (float)
      idf_w : (12,) IDF weights (normalized, sum=1)
    Output:
      (B, B) similarity matrix in [0, 1]
    """
    sqrt_w = idf_w.sqrt()
    inter  = (dv * sqrt_w) @ (dv * sqrt_w).T     # (B, B) - weighted intersection
    a_sum  = (dv * idf_w).sum(-1)                 # (B,)   - weighted row sum
    union  = a_sum.unsqueeze(1) + a_sum.unsqueeze(0) - inter
    return inter / (union + 1e-8)                 # (B, B) in [0.0, 1.0]


class MultiPositiveInfoNCE(nn.Module):
    """
    InfoNCE loss supporting multiple positives per anchor.

    Used in Phase 1 (epochs < CLUSTER_START) before clustering activates.
    Correctly handles the IU-Xray peculiarity that frontal and lateral images
    of the same patient are both valid positives.

    Mathematical form:
      L_i2t = -1/|P_i| sum_{j in P_i} log[ exp(l_ij) / sum_k exp(l_ik) ]
    where P_i = {j : patient_id[j] == patient_id[i]}.
    """

    def forward(self, logits: torch.Tensor, patient_ids: list[str]) -> torch.Tensor:
        device   = logits.device
        pos_mask = build_positive_mask(patient_ids, device).float()

        # Image -> text direction
        n_pos_i = pos_mask.sum(-1).clamp(min=1)
        l_i2t   = -(pos_mask * F.log_softmax(logits, dim=-1)).sum(-1) / n_pos_i

        # Text -> image direction
        n_pos_t = pos_mask.T.sum(-1).clamp(min=1)
        l_t2i   = -(pos_mask.T * F.log_softmax(logits.T, dim=-1)).sum(-1) / n_pos_t

        return (l_i2t.mean() + l_t2i.mean()) / 2


def build_cluster_positive_mask(
    patient_ids: list[str],
    disease_vecs: torch.Tensor,
    include_healthy_cluster: bool = True,
) -> torch.Tensor:
    """Build hard cluster positives for the basic clustering-guided ablation.

    Positive pairs are same-patient pairs plus disease-overlap pairs. Healthy
    both-normal pairs can be enabled separately so the Healthy Cluster
    contribution can be ablated cleanly.
    """
    device = disease_vecs.device
    dv = disease_vecs.float()
    patient_mask = build_positive_mask(patient_ids, device)
    overlap = (dv @ dv.T) > 0
    if include_healthy_cluster:
        is_normal = dv.sum(-1) == 0
        overlap = overlap | (is_normal.unsqueeze(1) & is_normal.unsqueeze(0))
    return patient_mask | overlap


class HierarchicalClusterLoss(nn.Module):
    """
    [C1] + [C2]: IDF-Weighted Jaccard soft labels + Healthy cluster.

    Soft target = blend of hard same-patient label and IDF-Jaccard soft label:
      target = (1 - alpha) * hard_label + alpha * idf_jaccard_soft

    [C2] Healthy cluster: patients with no pathology (No Finding = 1, all
    disease dims = 0) are assigned mutual similarity of 0.4; they share clinical
    similarity even though their disease_vecs are all-zero (Jaccard = 0/0 = 0).

    Args:
      alpha       : IDF-Jaccard blend weight (0=hard only, 1=soft only)
      temperature : softmax temperature for soft label smoothing
    """

    def __init__(
        self,
        alpha: float = CLUSTER_ALPHA,
        temperature: float = CLUSTER_TEMP,
        use_idf_jaccard: bool = True,
        use_healthy_cluster: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.use_idf_jaccard = use_idf_jaccard
        self.use_healthy_cluster = use_healthy_cluster

    def forward(
        self,
        logits: torch.Tensor,
        patient_ids: list[str],
        disease_vecs: torch.Tensor,
    ) -> torch.Tensor:
        device = logits.device
        dv     = disease_vecs.float()

        # Hard positives: same patient_id (includes study-level paired views)
        pos_mask = build_positive_mask(patient_ids, device).float()
        n_pos    = pos_mask.sum(-1, keepdim=True).clamp(min=1)
        hard     = pos_mask / n_pos          # normalized hard target

        if not self.use_idf_jaccard:
            cluster_mask = build_cluster_positive_mask(
                patient_ids,
                dv,
                include_healthy_cluster=self.use_healthy_cluster,
            ).float()
            target = cluster_mask / cluster_mask.sum(-1, keepdim=True).clamp(min=1)
            l_i2t = -(target   * F.log_softmax(logits,   dim=-1)).sum(-1).mean()
            l_t2i = -(target.T * F.log_softmax(logits.T, dim=-1)).sum(-1).mean()
            return (l_i2t + l_t2i) / 2

        # [C1] IDF-Weighted Jaccard soft similarity
        idf_w   = _idf_weights(device)
        sim_mat = idf_weighted_jaccard(dv, idf_w)

        if self.use_healthy_cluster:
            # [C2] Healthy Cluster: both-normal pairs get bonus similarity 0.4
            is_normal   = (dv.sum(-1) == 0)                                # (B,)
            both_normal = (is_normal.unsqueeze(1) & is_normal.unsqueeze(0)).float()
            sim_mat     = (sim_mat + both_normal * 0.4).clamp(max=1.0)

        soft   = F.softmax(sim_mat / self.temperature, dim=-1)
        target = (1.0 - self.alpha) * hard + self.alpha * soft

        l_i2t = -(target   * F.log_softmax(logits,   dim=-1)).sum(-1).mean()
        l_t2i = -(target.T * F.log_softmax(logits.T, dim=-1)).sum(-1).mean()
        return (l_i2t + l_t2i) / 2


class CombinedLoss(nn.Module):
    """
    Two-stage loss schedule used by the proposed paper run:

    Phase 1 (epoch < cluster_start):
      MultiPositiveInfoNCE - strict contrastive loss with multi-positive support.
      Purpose: warm up projection heads before noisy soft labels are introduced.

    Phase 2 (epoch >= cluster_start):
      HierarchicalClusterLoss [C1][C2] - IDF-Jaccard soft labels + Healthy Cluster.
      Purpose: encode disease similarity structure into the embedding space.
    """

    def __init__(
        self,
        cluster_start: int  = CLUSTER_START,
        alpha: float        = CLUSTER_ALPHA,
        temperature: float  = CLUSTER_TEMP,
        use_idf_jaccard: bool = True,
        use_healthy_cluster: bool = True,
    ):
        super().__init__()
        self.cluster_start = cluster_start
        self.mp_loss       = MultiPositiveInfoNCE()
        self.cluster_loss  = HierarchicalClusterLoss(
            alpha=alpha,
            temperature=temperature,
            use_idf_jaccard=use_idf_jaccard,
            use_healthy_cluster=use_healthy_cluster,
        )

    def forward(
        self,
        logits: torch.Tensor,
        img_emb: torch.Tensor,
        txt_emb: torch.Tensor,
        patient_ids: list[str],
        disease_vecs: torch.Tensor,
        epoch: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          total_loss : combined loss (backprop target)
          main_loss  : primary contrastive/clustering loss (for logging)
        """
        if epoch < self.cluster_start:
            main_loss  = self.mp_loss(logits, patient_ids)
        else:
            main_loss = self.cluster_loss(logits, patient_ids, disease_vecs)

        return main_loss, main_loss


# ==============================================================================
#  EVALUATION
# ==============================================================================
def recall_at_k(
    sim: torch.Tensor,
    gt_mask: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """
    Compute Recall@K for retrieval evaluation.

    A query is a "hit" at R@K if any of its top-K retrieved items is
    a ground-truth positive.

    Args:
      sim     : (N_queries, N_gallery) cosine similarity matrix
      gt_mask : (N_queries, N_gallery) bool - True where sample is a positive
      ks      : tuple of K values to evaluate
    Returns:
      dict {"R@1": float, "R@5": float, "R@10": float} in percentage [0, 100]
    """
    results: dict[str, float] = {}
    N = sim.shape[0]
    for k in ks:
        topk  = sim.topk(k, dim=-1).indices   # (N, k)
        n_hit = sum(1 for i in range(N) if gt_mask[i][topk[i]].any())
        results[f"R@{k}"] = n_hit / N * 100
    return results


@torch.no_grad()
def evaluate(
    model: MedicalSwinBERT,
    loader: DataLoader,
    tokenizer,
    device: torch.device,
) -> tuple[dict, dict, dict, dict, float, float]:
    """
    Full evaluation on a DataLoader. Returns recall metrics for two protocols:

    STRICT  - ground truth = exact same patient_id (traditional retrieval)
    CLUSTER - ground truth = any shared pathological label (clinically meaningful)

    Both i2t (image->text) and t2i (text->image) directions are evaluated.

    Returns:
      (strict_i2t, strict_t2i, cluster_i2t, cluster_t2i,
       mean_strict_r1, mean_cluster_r1)
    """
    model.eval()
    all_ie, all_te, all_pids, all_labs = [], [], [], []

    for batch in loader:
        imgs = batch["image"].to(device)
        tok  = tokenizer(
            batch["caption"],
            padding="max_length",
            truncation=True,
            max_length=TEXT_MAX_LEN,
            return_tensors="pt",
        ).to(device)
        ie, te, _ = model(imgs, tok["input_ids"], tok["attention_mask"])
        all_ie.append(ie.cpu())
        all_te.append(te.cpu())
        all_pids.extend(batch["pid"])
        all_labs.append(batch["labels"])

    model.train()

    ie   = torch.cat(all_ie)                    # (N, 512)
    te   = torch.cat(all_te)                    # (N, 512)
    labs = torch.cat(all_labs).numpy()          # (N, 14)
    pids = np.array(all_pids)                   # (N,)
    N    = ie.shape[0]

    sim_i2t = ie @ te.T    # (N, N)
    sim_t2i = te @ ie.T    # (N, N)

    # Ground truth 1 - STRICT: exact same patient_id
    gt_strict = torch.from_numpy(pids[:, None] == pids[None, :])   # (N, N) bool

    # Ground truth 2 - CLUSTER: any shared pathological finding.
    # Use the same 12-dim merged disease representation as training.
    path = labels_to_disease_vecs(torch.tensor(labs, dtype=torch.float32))
    overlap     = (path @ path.T) > 0
    is_normal   = (path.sum(-1) == 0)
    both_normal = is_normal.unsqueeze(1) & is_normal.unsqueeze(0)
    gt_cluster  = overlap | both_normal                             # (N, N) bool

    strict_i2t  = recall_at_k(sim_i2t, gt_strict)
    strict_t2i  = recall_at_k(sim_t2i, gt_strict.T)
    cluster_i2t = recall_at_k(sim_i2t, gt_cluster)
    cluster_t2i = recall_at_k(sim_t2i, gt_cluster.T)

    sr1 = (strict_i2t["R@1"]  + strict_t2i["R@1"])  / 2
    cr1 = (cluster_i2t["R@1"] + cluster_t2i["R@1"]) / 2

    return strict_i2t, strict_t2i, cluster_i2t, cluster_t2i, sr1, cr1


def log_eval_results(
    strict_i2t: dict,
    strict_t2i: dict,
    cluster_i2t: dict,
    cluster_t2i: dict,
    sr1: float,
    cr1: float,
) -> None:
    """Pretty-print evaluation results."""
    log(f"  [STRICT  - same patient]")
    log(f"  i2t | R@1={strict_i2t['R@1']:6.2f}%  R@5={strict_i2t['R@5']:6.2f}%  R@10={strict_i2t['R@10']:6.2f}%")
    log(f"  t2i | R@1={strict_t2i['R@1']:6.2f}%  R@5={strict_t2i['R@5']:6.2f}%  R@10={strict_t2i['R@10']:6.2f}%")
    log(f"  Mean Strict  R@1={sr1:.2f}%")
    log(f"")
    log(f"  [CLUSTER - shared pathology]")
    log(f"  i2t | R@1={cluster_i2t['R@1']:6.2f}%  R@5={cluster_i2t['R@5']:6.2f}%  R@10={cluster_i2t['R@10']:6.2f}%")
    log(f"  t2i | R@1={cluster_t2i['R@1']:6.2f}%  R@5={cluster_t2i['R@5']:6.2f}%  R@10={cluster_t2i['R@10']:6.2f}%")
    log(f"  Mean Cluster R@1={cr1:.2f}%")


# ==============================================================================
#  TRAINING
# ==============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-GPU training for Medical Image-Text Retrieval"
    )
    parser.add_argument("--csv_path",    default=CSV_PATH)
    parser.add_argument("--img_dir",     default=IMG_DIR)
    parser.add_argument("--out_dir",     default=OUT_DIR)
    parser.add_argument("--epochs",      type=int,   default=NUM_EPOCHS)
    parser.add_argument("--batch_size",  type=int,   default=BATCH_SIZE)
    parser.add_argument("--grad_accum",  type=int,   default=GRAD_ACCUM)
    parser.add_argument("--freeze_ep",   type=int,   default=FREEZE_EP)
    parser.add_argument("--eval_every",  type=int,   default=EVAL_EVERY)
    parser.add_argument("--seed",        type=int,   default=SEED)
    parser.add_argument("--num_workers", type=int,   default=NUM_WORKERS)
    parser.add_argument("--resume",      default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def main() -> None:
    global _log_file

    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    _log_file = open(os.path.join(args.out_dir, "train.log"), "w", encoding="utf-8")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log("=" * 65)
    log("  Single-GPU Medical VLM Training")
    log("  Contributions: [C1] IDF-Jaccard  [C2] Healthy Cluster")
    log("=" * 65)
    log(f"  Device       : {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        log(f"  GPU          : {props.name}")
        log(f"  VRAM         : {props.total_memory // 1024**2} MB")
    log(f"  Vision model : {VISION_MODEL}")
    log(f"  Text model   : {TEXT_MODEL}")
    log(f"  Embed dim    : {EMBED_DIM}")
    log(f"  Proj hid dim : {PROJ_HID_DIM}")
    log(f"  Batch size   : {args.batch_size}  (per step)")
    log(f"  Grad accum   : {args.grad_accum}")
    log(f"  Eff batch    : {args.batch_size * args.grad_accum}")
    log(f"  Epochs       : {args.epochs}")
    log(f"  Freeze ep    : {args.freeze_ep}")
    log(f"  Cluster start: {CLUSTER_START}")
    log(f"  Output dir   : {args.out_dir}")
    log("=" * 65)

    # Save run config for reproducibility
    config = {k: v for k, v in vars(args).items()}
    config.update({
        "VISION_MODEL": VISION_MODEL, "TEXT_MODEL": TEXT_MODEL,
        "EMBED_DIM": EMBED_DIM, "PROJ_HID_DIM": PROJ_HID_DIM,
        "LR_HEAD": LR_HEAD, "LR_ENC": LR_ENC, "LR_TEXT": LR_TEXT,
        "WEIGHT_DECAY": WEIGHT_DECAY, "MAX_GRAD": MAX_GRAD,
        "CLUSTER_START": CLUSTER_START, "CLUSTER_ALPHA": CLUSTER_ALPHA,
        "CLUSTER_TEMP": CLUSTER_TEMP,
        "effective_batch": args.batch_size * args.grad_accum,
    })
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # -- Data ------------------------------------------------------------------
    log("\n[DATA]")
    df = pd.read_csv(args.csv_path)
    df["patient_id"] = df["patient_id"].astype(str)
    df_train = patient_split(df, "train", seed=args.seed)
    df_val   = patient_split(df, "val",   seed=args.seed)
    df_test  = patient_split(df, "test",  seed=args.seed)
    log(f"  Train: {len(df_train):5d} imgs | {df_train['patient_id'].nunique()} patients")
    log(f"  Val  : {len(df_val):5d} imgs | {df_val['patient_id'].nunique()} patients")
    log(f"  Test : {len(df_test):5d} imgs | {df_test['patient_id'].nunique()} patients")
    config.update({
        "train_images": int(len(df_train)),
        "val_images": int(len(df_val)),
        "test_images": int(len(df_test)),
        "train_patients": int(df_train["patient_id"].nunique()),
        "val_patients": int(df_val["patient_id"].nunique()),
        "test_patients": int(df_test["patient_id"].nunique()),
    })
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    train_ds = IUXrayDataset(df_train, args.img_dir, get_train_transform())
    val_ds   = IUXrayDataset(df_val,   args.img_dir, get_val_transform())
    test_ds  = IUXrayDataset(df_test,  args.img_dir, get_val_transform())

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader  = DataLoader(
        val_ds,  batch_size=16, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=16, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn,
    )

    # -- Model -----------------------------------------------------------------
    log("\n[MODEL]")
    model     = MedicalSwinBERT().to(device)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)

    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    log(f"  Total params  : {n_total:.1f}M")

    # -- Loss ------------------------------------------------------------------
    criterion = CombinedLoss(cluster_start=CLUSTER_START).to(device)
    log(f"\n[LOSS] cluster_start={CLUSTER_START}")
    log(f"       alpha={CLUSTER_ALPHA} | temp={CLUSTER_TEMP}")

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda")

    # -- Phase 1: Freeze encoders -----------------------------------------------
    log(f"\n[PHASE 1] Freeze encoders for {args.freeze_ep} epochs (heads-only warm-up)")
    for p in model.backbone_params():
        p.requires_grad = False
    opt = AdamW(model.head_params(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)

    # -- Resume from checkpoint -------------------------------------------------
    start_epoch = 1
    best_r1     = 0.0
    best_epoch  = 0
    history: list[dict] = []
    sched = None  # will be created in Phase 2

    if args.resume and os.path.isfile(args.resume):
        log(f"\n[RESUME] Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_r1     = ckpt.get("best_r1", 0.0)
        best_epoch  = ckpt.get("epoch", 0)
        log(f"  Resuming from epoch {start_epoch} | best R@1={best_r1:.2f}%")

        # If resuming past freeze epoch, rebuild Phase 2 optimizer/scheduler
        if start_epoch > args.freeze_ep + 1:
            log(f"  Re-building Phase 2 optimizer (already past freeze)")
            for p in model.backbone_params():
                p.requires_grad = True
            opt = AdamW([
                {"params": model.image_encoder.parameters(), "lr": LR_ENC},
                {"params": model.text_encoder.parameters(),  "lr": LR_TEXT},
                {"params": model.head_params(),               "lr": LR_HEAD},
            ], weight_decay=WEIGHT_DECAY)
            remain_steps = len(train_loader) * (args.epochs - args.freeze_ep) // args.grad_accum
            warmup_steps = max(1, int(remain_steps * 0.05))
            sched = SequentialLR(opt, [
                LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
                CosineAnnealingLR(opt, T_max=remain_steps - warmup_steps, eta_min=1e-8),
            ], milestones=[warmup_steps])

    log("\n" + "=" * 65)
    log(f"  TRAINING: {args.epochs} epochs  (start={start_epoch})")
    log(f"  Effective batch = {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    log("=" * 65)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        # -- Phase 2: Unfreeze encoders -------------------------------------
        if epoch == args.freeze_ep + 1:
            log(f"\n[PHASE 2] Unfreeze encoders at epoch {epoch}")
            torch.cuda.empty_cache()
            for p in model.backbone_params():
                p.requires_grad = True
            opt = AdamW([
                {"params": model.image_encoder.parameters(), "lr": LR_ENC},
                {"params": model.text_encoder.parameters(),  "lr": LR_TEXT},
                {"params": model.head_params(),               "lr": LR_HEAD},
            ], weight_decay=WEIGHT_DECAY)
            remain_steps = len(train_loader) * (args.epochs - args.freeze_ep) // args.grad_accum
            warmup_steps = max(1, int(remain_steps * 0.05))
            sched = SequentialLR(opt, [
                LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
                CosineAnnealingLR(opt, T_max=remain_steps - warmup_steps, eta_min=1e-8),
            ], milestones=[warmup_steps])
            log(f"  LR: enc={LR_ENC:.0e}  text={LR_TEXT:.0e}  head={LR_HEAD:.0e}")
            log(f"  Warmup {warmup_steps} steps -> CosineAnnealing ({remain_steps - warmup_steps} steps)")

        # -- Training Loop -------------------------------------------------
        model.train()
        L_total, L_main = [], []
        opt.zero_grad()

        for step, batch in enumerate(train_loader):
            imgs   = batch["image"].to(device)
            labels = batch["labels"].to(device)
            pids   = batch["pid"]
            caps   = batch["caption"]

            tok = tokenizer(
                caps,
                padding="max_length",
                truncation=True,
                max_length=TEXT_MAX_LEN,
                return_tensors="pt",
            ).to(device)

            with torch.amp.autocast("cuda"):
                img_emb, txt_emb, scale = model(imgs, tok["input_ids"], tok["attention_mask"])

                # Convert labels: 14-dim PATH_COLS -> 12-dim CHEXPERT_COLS
                disease_vecs = labels_to_disease_vecs(labels)

                # [C1][C2]: IDF-Jaccard + Healthy Cluster loss
                logits = scale * (img_emb @ txt_emb.T)
                loss_task3, loss_main = criterion(
                    logits, img_emb, txt_emb, pids, disease_vecs, epoch
                )

                loss = loss_task3 / args.grad_accum

            scaler.scale(loss).backward()

            L_total.append(loss_task3.item())
            L_main.append(loss_main.item())

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                if epoch > args.freeze_ep and sched is not None:
                    sched.step()

        elapsed = time.time() - t0
        temp    = model.logit_scale.exp().item()
        phase   = (
            "Phase1" if epoch <= args.freeze_ep
            else "Phase2"
        )

        log(
            f"Ep[{epoch:03d}/{args.epochs}] {phase} "
            f"loss={np.mean(L_total):.4f} "
            f"(main={np.mean(L_main):.4f}) "
            f"T={temp:.1f}  t={elapsed:.0f}s"
        )

        # -- Evaluation ----------------------------------------------------
        do_eval = (
            epoch % args.eval_every == 0
            or epoch == args.epochs
            or epoch <= 3
        )
        if do_eval:
            log(f"\n--- Eval epoch {epoch} ---")
            si, st, ci, ct, sr1, cr1 = evaluate(model, val_loader, tokenizer, device)
            log_eval_results(si, st, ci, ct, sr1, cr1)

            row = dict(
                epoch=epoch,
                phase=phase,
                elapsed_s=float(elapsed),
                loss=float(np.mean(L_total)),
                loss_main=float(np.mean(L_main)),
                temp=temp,
                batch_size=args.batch_size,
                grad_accum=args.grad_accum,
                effective_batch=args.batch_size * args.grad_accum,
                strict_r1=sr1,
                cluster_r1=cr1,
                strict_i2t_r1=si["R@1"],  strict_i2t_r5=si["R@5"],  strict_i2t_r10=si["R@10"],
                strict_t2i_r1=st["R@1"],  strict_t2i_r5=st["R@5"],  strict_t2i_r10=st["R@10"],
                cluster_i2t_r1=ci["R@1"], cluster_i2t_r5=ci["R@5"], cluster_i2t_r10=ci["R@10"],
                cluster_t2i_r1=ct["R@1"], cluster_t2i_r5=ct["R@5"], cluster_t2i_r10=ct["R@10"],
            )
            history.append(row)

            is_best = sr1 > best_r1
            if is_best:
                best_r1    = sr1
                best_epoch = epoch
                torch.save(
                    {
                        "epoch"   : epoch,
                        "model"   : model.state_dict(),
                        "best_r1" : best_r1,
                        "strict"  : {"i2t": si, "t2i": st},
                        "cluster" : {"i2t": ci, "t2i": ct},
                        "config"  : config,
                    },
                    os.path.join(args.out_dir, "best.pt"),
                )
                best_summary = {
                    "best_epoch": epoch,
                    "best_val_strict_r1": best_r1,
                    "best_val_cluster_r1": cr1,
                    "strict": {"i2t": si, "t2i": st},
                    "cluster": {"i2t": ci, "t2i": ct},
                    "batch_size": args.batch_size,
                    "grad_accum": args.grad_accum,
                    "effective_batch": args.batch_size * args.grad_accum,
                    "freeze_ep": args.freeze_ep,
                }
                with open(os.path.join(args.out_dir, "best_summary.json"), "w") as f:
                    json.dump(best_summary, f, indent=2)
                log(f"  * NEW BEST Strict R@1={best_r1:.2f}% @ ep{epoch}")
            else:
                log(f"  (best={best_r1:.2f}% @ ep{best_epoch})")

            pd.DataFrame(history).to_csv(
                os.path.join(args.out_dir, "history.csv"), index=False
            )
            compat_rows = []
            for h in history:
                compat_rows.append({
                    "epoch": h["epoch"],
                    "train_loss": h["loss"],
                    "r1_strict": h["strict_r1"],
                    "r5_strict": (h["strict_i2t_r5"] + h["strict_t2i_r5"]) / 2,
                    "r10_strict": (h["strict_i2t_r10"] + h["strict_t2i_r10"]) / 2,
                    "r1_cluster": h["cluster_r1"],
                    "r5_cluster": (h["cluster_i2t_r5"] + h["cluster_t2i_r5"]) / 2,
                    "r10_cluster": (h["cluster_i2t_r10"] + h["cluster_t2i_r10"]) / 2,
                    "best_r1_strict": max(x["strict_r1"] for x in history if x["epoch"] <= h["epoch"]),
                    "is_best": int(h["epoch"] == best_epoch),
                })
            pd.DataFrame(compat_rows).to_csv(
                os.path.join(args.out_dir, "training_history_lvtm_compat.csv"),
                index=False,
            )
            latest_summary = {
                "epoch": epoch,
                "phase": phase,
                "best_epoch": best_epoch,
                "best_val_strict_r1": best_r1,
                "current_val_strict_r1": sr1,
                "current_val_cluster_r1": cr1,
                "strict_i2t": si,
                "strict_t2i": st,
                "cluster_i2t": ci,
                "cluster_t2i": ct,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "effective_batch": args.batch_size * args.grad_accum,
            }
            with open(os.path.join(args.out_dir, "latest_metrics.json"), "w") as f:
                json.dump(latest_summary, f, indent=2)
            with open(os.path.join(args.out_dir, "progress.log"), "a", encoding="utf-8") as f:
                f.write(
                    " | ".join([
                        f"epoch={epoch}",
                        f"phase={phase}",
                        f"loss={np.mean(L_total):.4f}",
                        f"strict_r1={sr1:.2f}",
                        f"strict_r5={((si['R@5'] + st['R@5']) / 2):.2f}",
                        f"strict_r10={((si['R@10'] + st['R@10']) / 2):.2f}",
                        f"cluster_r1={cr1:.2f}",
                        f"cluster_r5={((ci['R@5'] + ct['R@5']) / 2):.2f}",
                        f"cluster_r10={((ci['R@10'] + ct['R@10']) / 2):.2f}",
                        f"best_strict_r1={best_r1:.2f}",
                    ]) + "\n"
                )

            # Periodic checkpoint every 10 epochs
            if epoch % 10 == 0:
                torch.save(
                    {"epoch": epoch, "model": model.state_dict(), "best_r1": best_r1},
                    os.path.join(args.out_dir, f"ckpt_ep{epoch:03d}.pt"),
                )
                log(f"  Saved checkpoint ckpt_ep{epoch:03d}.pt")

    # -- Final Test Evaluation ------------------------------------------------
    log("\n" + "=" * 55)
    log("  FINAL TEST EVALUATION")
    log("=" * 55)

    best_ckpt_path = os.path.join(args.out_dir, "best.pt")
    if os.path.isfile(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        log(f"Loaded best checkpoint (ep{ckpt['epoch']}, val Strict R@1={ckpt['best_r1']:.2f}%)")
    else:
        log("WARNING: No best.pt found - evaluating current model weights.")

    si, st, ci, ct, sr1, cr1 = evaluate(model, test_loader, tokenizer, device)
    log("\n  [TEST RESULTS]")
    log_eval_results(si, st, ci, ct, sr1, cr1)

    # Persist results
    pd.DataFrame(history).to_csv(os.path.join(args.out_dir, "history.csv"), index=False)
    compat_rows = []
    for h in history:
        compat_rows.append({
            "epoch": h["epoch"],
            "train_loss": h["loss"],
            "r1_strict": h["strict_r1"],
            "r5_strict": (h["strict_i2t_r5"] + h["strict_t2i_r5"]) / 2,
            "r10_strict": (h["strict_i2t_r10"] + h["strict_t2i_r10"]) / 2,
            "r1_cluster": h["cluster_r1"],
            "r5_cluster": (h["cluster_i2t_r5"] + h["cluster_t2i_r5"]) / 2,
            "r10_cluster": (h["cluster_i2t_r10"] + h["cluster_t2i_r10"]) / 2,
            "best_r1_strict": max(x["strict_r1"] for x in history if x["epoch"] <= h["epoch"]),
            "is_best": int(h["epoch"] == best_epoch),
        })
    pd.DataFrame(compat_rows).to_csv(
        os.path.join(args.out_dir, "training_history_lvtm_compat.csv"),
        index=False,
    )
    test_results = {
        "test_strict" : {"i2t": si, "t2i": st},
        "test_cluster": {"i2t": ci, "t2i": ct},
        "best_epoch"  : best_epoch,
        "best_val_strict_r1": best_r1,
        "mean_strict_r1_test": sr1,
        "mean_cluster_r1_test": cr1,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "freeze_ep": args.freeze_ep,
        "epochs": args.epochs,
    }
    with open(os.path.join(args.out_dir, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)

    with open(os.path.join(args.out_dir, "run_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"out_dir={args.out_dir}\n")
        f.write(f"batch_size={args.batch_size}\n")
        f.write(f"grad_accum={args.grad_accum}\n")
        f.write(f"effective_batch={args.batch_size * args.grad_accum}\n")
        f.write(f"freeze_ep={args.freeze_ep}\n")
        f.write(f"epochs={args.epochs}\n")
        f.write(f"best_epoch={best_epoch}\n")
        f.write(f"best_val_strict_r1={best_r1:.4f}\n")
        f.write(f"test_mean_strict_r1={sr1:.4f}\n")
        f.write(f"test_mean_cluster_r1={cr1:.4f}\n")
        f.write(f"test_strict_i2t_r1={si['R@1']:.4f}\n")
        f.write(f"test_strict_i2t_r5={si['R@5']:.4f}\n")
        f.write(f"test_strict_i2t_r10={si['R@10']:.4f}\n")
        f.write(f"test_strict_t2i_r1={st['R@1']:.4f}\n")
        f.write(f"test_strict_t2i_r5={st['R@5']:.4f}\n")
        f.write(f"test_strict_t2i_r10={st['R@10']:.4f}\n")
        f.write(f"test_cluster_i2t_r1={ci['R@1']:.4f}\n")
        f.write(f"test_cluster_i2t_r5={ci['R@5']:.4f}\n")
        f.write(f"test_cluster_i2t_r10={ci['R@10']:.4f}\n")
        f.write(f"test_cluster_t2i_r1={ct['R@1']:.4f}\n")
        f.write(f"test_cluster_t2i_r5={ct['R@5']:.4f}\n")
        f.write(f"test_cluster_t2i_r10={ct['R@10']:.4f}\n")

    log(f"\nDONE. All outputs saved to: {args.out_dir}")
    if _log_file:
        _log_file.close()


# ==============================================================================
#  ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    main()
