#!/usr/bin/env python3
"""
# Proposed method: slow-curriculum clinical-cluster training for paper runs

This variant treats one patient/study as one paired sample:
  - image side = frontal+lateral study embedding
  - text side  = single report for that patient

It keeps the thesis direction (cluster-aware false-negative mitigation) and
uses a slow curriculum so strict retrieval can stabilize before clinical
same-disease supervision becomes strong.
"""

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoModel, AutoTokenizer

import random
import train_single_gpu as base


CSV_PATH = base.CSV_PATH
IMG_DIR = base.IMG_DIR
OUT_DIR = "outputs_study"

VISION_MODEL = base.VISION_MODEL
TEXT_MODEL = base.TEXT_MODEL
IMG_SIZE = base.IMG_SIZE
EMBED_DIM = base.EMBED_DIM
PROJ_HID_DIM = base.PROJ_HID_DIM
DROPOUT = base.DROPOUT
TEXT_MAX_LEN = base.TEXT_MAX_LEN
PATH_COLS = base.PATH_COLS
CHEXPERT_COLS = base.CHEXPERT_COLS

NUM_EPOCHS = 150
BATCH_SIZE = 32
GRAD_ACCUM = 4
FREEZE_EP = 5
NUM_WORKERS = 4
SEED = 42
EVAL_EVERY = 5

LR_HEAD = 2e-4
LR_ENC = 1e-5
LR_TEXT = 1e-4
WEIGHT_DECAY = base.WEIGHT_DECAY
MAX_GRAD = base.MAX_GRAD

CLUSTER_START = 12
CLUSTER_ALPHA = 0.35
CLUSTER_TEMP = 1.0
AUX_WEIGHT = 0.02
CLINICAL_START = 15
CLINICAL_WEIGHT = 0.12
BALANCED_CLINICAL_WEIGHT = 0.10
MIN_STRICT_FOR_BALANCED = 4.00
CLINICAL_BEST_MIN_STRICT = 3.50
MAX_VIEWS = 2

IMG_MEAN = base.IMG_MEAN
IMG_STD = base.IMG_STD
VIEW_TO_ID = {"f": 0, "l": 1, "o": 2, "pad": 3}

_log_file = None


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _log_file is not None:
        _log_file.write(line + "\n")
        _log_file.flush()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class StudyIUXrayDataset(Dataset):
    """
    Groups rows by patient_id so one patient/study becomes one training sample.

    For each patient:
      - sample up to one frontal and one lateral image
      - keep one report
      - aggregate pathology labels with max() across images
    """

    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        transform,
        train_mode: bool,
        max_views: int = MAX_VIEWS,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.train_mode = train_mode
        self.max_views = max_views
        self.samples = self._build_samples()

    def _build_samples(self) -> list[dict]:
        samples = []
        for pid, g in self.df.groupby(self.df["patient_id"].astype(str), sort=True):
            g = g.reset_index(drop=True)
            labels = np.array(
                [g[col].max() if col in g.columns else 0.0 for col in PATH_COLS],
                dtype=np.float32,
            )
            report = str(g.iloc[0]["org_caption"])

            frontal = []
            lateral = []
            other = []
            for _, row in g.iterrows():
                proj = str(row.get("projection", "")).strip().lower()
                img_id = str(row["image_id"])
                if proj.startswith("f"):
                    frontal.append(img_id)
                elif proj.startswith("l"):
                    lateral.append(img_id)
                else:
                    other.append(img_id)

            samples.append({
                "pid": str(pid),
                "caption": report,
                "labels": labels,
                "views": {
                    "f": sorted(frontal),
                    "l": sorted(lateral),
                    "o": sorted(other),
                },
            })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _choose_image_id(self, candidates: list[str]) -> str | None:
        if not candidates:
            return None
        if not self.train_mode or len(candidates) == 1:
            return candidates[0]
        idx = np.random.randint(0, len(candidates))
        return candidates[idx]

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        chosen_ids = []
        chosen_views = []
        for view_key in ("f", "l"):
            image_id = self._choose_image_id(sample["views"][view_key])
            if image_id is not None:
                chosen_ids.append(image_id)
                chosen_views.append(view_key)

        if not chosen_ids:
            image_id = self._choose_image_id(sample["views"]["o"])
            if image_id is None:
                # Fallback: there must be at least one image per patient.
                all_ids = sample["views"]["f"] + sample["views"]["l"] + sample["views"]["o"]
                image_id = all_ids[0]
            chosen_ids.append(image_id)
            chosen_views.append("o")

        images = []
        for image_id in chosen_ids[: self.max_views]:
            path = os.path.join(self.img_dir, image_id)
            img = Image.open(path).convert("RGB")
            images.append(self.transform(img))

        while len(images) < self.max_views:
            images.append(torch.zeros_like(images[0]))
            chosen_views.append("pad")

        view_mask = torch.tensor(
            [1 if v != "pad" else 0 for v in chosen_views[: self.max_views]],
            dtype=torch.bool,
        )
        view_type_ids = torch.tensor(
            [VIEW_TO_ID[v] for v in chosen_views[: self.max_views]],
            dtype=torch.long,
        )
        return {
            "images": torch.stack(images),
            "view_mask": view_mask,
            "view_type_ids": view_type_ids,
            "caption": sample["caption"],
            "pid": sample["pid"],
            "labels": torch.from_numpy(sample["labels"]),
        }


def collate_study(batch: list[dict]) -> dict:
    return {
        "images": torch.stack([b["images"] for b in batch]),
        "view_mask": torch.stack([b["view_mask"] for b in batch]),
        "view_type_ids": torch.stack([b["view_type_ids"] for b in batch]),
        "caption": [b["caption"] for b in batch],
        "pid": [b["pid"] for b in batch],
        "labels": torch.stack([b["labels"] for b in batch]),
    }


class ProjectionHead(nn.Module):
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


class StudyMedicalSwinBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = AutoModel.from_pretrained(VISION_MODEL)
        try:
            self.image_encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            self.image_encoder.gradient_checkpointing_enable()

        self.text_encoder = AutoModel.from_pretrained(TEXT_MODEL)
        try:
            self.text_encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            self.text_encoder.gradient_checkpointing_enable()

        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            img_dim = self.image_encoder(pixel_values=dummy).pooler_output.shape[-1]
        txt_dim = self.text_encoder.config.hidden_size

        self.img_proj = ProjectionHead(img_dim, PROJ_HID_DIM, EMBED_DIM)
        self.txt_proj = ProjectionHead(txt_dim, PROJ_HID_DIM, EMBED_DIM)
        self.view_type_embed = nn.Embedding(4, EMBED_DIM, padding_idx=VIEW_TO_ID["pad"])
        self.view_attn = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM // 2),
            nn.GELU(),
            nn.Linear(EMBED_DIM // 2, 1),
        )
        self.img_pathology_head = nn.Linear(EMBED_DIM, len(CHEXPERT_COLS))
        self.img_normal_head = nn.Linear(EMBED_DIM, 1)
        self.txt_pathology_head = nn.Linear(EMBED_DIM, len(CHEXPERT_COLS))
        self.txt_normal_head = nn.Linear(EMBED_DIM, 1)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

        log(f"  SwinV2 pooler dim: {img_dim}")
        log(f"  BERT hidden dim: {txt_dim}")

    def encode_study_images(
        self,
        images: torch.Tensor,
        view_mask: torch.Tensor,
        view_type_ids: torch.Tensor,
    ):
        bsz, n_views, c, h, w = images.shape
        flat_images = images.view(bsz * n_views, c, h, w)
        flat_mask = view_mask.view(-1)
        valid_images = flat_images[flat_mask]

        feat = self.image_encoder(pixel_values=valid_images).pooler_output
        proj = self.img_proj(feat, normalize=False)

        full = torch.zeros(bsz * n_views, EMBED_DIM, device=images.device, dtype=proj.dtype)
        full[flat_mask] = proj
        full = full.view(bsz, n_views, EMBED_DIM)

        view_bias = self.view_type_embed(view_type_ids)
        attn_scores = self.view_attn(full + view_bias).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~view_mask, -1e4)
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = (attn_weights.unsqueeze(-1) * full).sum(dim=1)
        return F.normalize(pooled, dim=-1), pooled

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        cls = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state[:, 0, :]
        proj = self.txt_proj(cls, normalize=False)
        return F.normalize(proj, dim=-1), proj

    def forward(
        self,
        images: torch.Tensor,
        view_mask: torch.Tensor,
        view_type_ids: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        self.logit_scale.data.clamp_(math.log(1.0), math.log(100.0))
        img_emb, img_raw = self.encode_study_images(images, view_mask, view_type_ids)
        txt_emb, txt_raw = self.encode_text(input_ids, attention_mask)
        scale = self.logit_scale.exp()
        img_pathology_logits = self.img_pathology_head(img_emb)
        img_normal_logits = self.img_normal_head(img_emb).squeeze(-1)
        txt_pathology_logits = self.txt_pathology_head(txt_emb)
        txt_normal_logits = self.txt_normal_head(txt_emb).squeeze(-1)
        return (
            img_emb,
            txt_emb,
            scale,
            img_pathology_logits,
            img_normal_logits,
            txt_pathology_logits,
            txt_normal_logits,
            img_raw,
            txt_raw,
        )

    def backbone_params(self) -> list:
        return list(self.image_encoder.parameters()) + list(self.text_encoder.parameters())

    def head_params(self) -> list:
        return (
            list(self.img_proj.parameters())
            + list(self.txt_proj.parameters())
            + list(self.view_type_embed.parameters())
            + list(self.view_attn.parameters())
            + list(self.img_pathology_head.parameters())
            + list(self.img_normal_head.parameters())
            + list(self.txt_pathology_head.parameters())
            + list(self.txt_normal_head.parameters())
            + [self.logit_scale]
        )


def build_cluster_mask(disease_vecs: torch.Tensor) -> torch.Tensor:
    path = disease_vecs.float()
    overlap = (path @ path.T) > 0
    is_normal = path.sum(-1) == 0
    both_normal = is_normal.unsqueeze(1) & is_normal.unsqueeze(0)
    return overlap | both_normal



def build_clinical_cluster_mask(disease_vecs: torch.Tensor) -> torch.Tensor:
    """Clinical-positive mask: only non-normal pairs with disease overlap.
    Both-normal pairs are excluded because they are too broad for clinical-valid
    supervised contrastive learning.
    """
    path = disease_vecs.float()
    overlap = (path @ path.T) > 0
    is_normal = path.sum(-1) == 0
    both_non_normal = (~is_normal).unsqueeze(1) & (~is_normal).unsqueeze(0)
    return overlap & both_non_normal


class ClinicalSupervisedContrastiveLoss(nn.Module):
    """
    Pull non-normal disease-overlap peers together across image/text embeddings.

    Same-clinical-cluster peers become positives instead of ordinary negatives,
    matching the thesis objective.
    The diagonal exact match is excluded here because strict retrieval is already
    handled by the main contrastive objective.
    """

    def forward(self, logits: torch.Tensor, disease_vecs: torch.Tensor) -> torch.Tensor:
        device = logits.device
        n = logits.shape[0]
        diag = torch.eye(n, dtype=torch.bool, device=device)
        peer_mask = build_clinical_cluster_mask(disease_vecs.to(device)) & ~diag

        if not peer_mask.any():
            return logits.sum() * 0.0

        losses = []
        for sim, mask in ((logits, peer_mask), (logits.T, peer_mask.T)):
            valid = mask.any(dim=-1)
            if not valid.any():
                continue
            pos = mask.float()
            denom = pos.sum(dim=-1).clamp(min=1.0)
            log_prob = F.log_softmax(sim, dim=-1)
            per_query = -(pos * log_prob).sum(dim=-1) / denom
            losses.append(per_query[valid])

        if not losses:
            return logits.sum() * 0.0
        return torch.cat(losses).mean()


class StudyLoss(nn.Module):
    def __init__(
        self,
        cluster_start: int = CLUSTER_START,
        aux_weight: float = AUX_WEIGHT,
        clinical_start: int = CLINICAL_START,
        clinical_weight: float = CLINICAL_WEIGHT,
        use_idf_jaccard: bool = True,
        use_healthy_cluster: bool = True,
    ):
        super().__init__()
        self.cluster_start = cluster_start
        self.aux_weight = aux_weight
        self.clinical_start = clinical_start
        self.clinical_weight = clinical_weight
        self.base_loss = base.CombinedLoss(
            cluster_start=cluster_start,
            alpha=CLUSTER_ALPHA, temperature=CLUSTER_TEMP,
            use_idf_jaccard=use_idf_jaccard,
            use_healthy_cluster=use_healthy_cluster,
        )
        self.clinical_loss = ClinicalSupervisedContrastiveLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def set_clinical_weight(self, value: float) -> None:
        self.clinical_weight = float(value)

    def forward(
        self,
        logits: torch.Tensor,
        img_emb: torch.Tensor,
        txt_emb: torch.Tensor,
        patient_ids: list[str],
        disease_vecs: torch.Tensor,
        img_pathology_logits: torch.Tensor,
        img_normal_logits: torch.Tensor,
        txt_pathology_logits: torch.Tensor,
        txt_normal_logits: torch.Tensor,
        epoch: int,
    ):
        loss_main, main_component = self.base_loss(
            logits, img_emb, txt_emb, patient_ids, disease_vecs, epoch
        )

        if epoch >= self.clinical_start:
            clinical_component = self.clinical_loss(logits, disease_vecs)
        else:
            clinical_component = logits.sum() * 0.0

        target_path = disease_vecs.float()
        target_normal = (target_path.sum(dim=-1) == 0).float()
        aux_img_path = self.bce(img_pathology_logits, target_path)
        aux_img_norm = self.bce(img_normal_logits, target_normal)
        aux_txt_path = self.bce(txt_pathology_logits, target_path)
        aux_txt_norm = self.bce(txt_normal_logits, target_normal)
        consistency_path = F.mse_loss(
            torch.sigmoid(img_pathology_logits),
            torch.sigmoid(txt_pathology_logits),
        )
        consistency_norm = F.mse_loss(
            torch.sigmoid(img_normal_logits),
            torch.sigmoid(txt_normal_logits),
        )
        aux_component = (
            aux_img_path + aux_img_norm + aux_txt_path + aux_txt_norm
        ) / 4 + 0.25 * (consistency_path + consistency_norm)

        total = (
            loss_main
            + self.clinical_weight * clinical_component
            + self.aux_weight * aux_component
        )
        return total, main_component, aux_component, clinical_component


def clinical_weight_for_epoch(epoch: int, args: argparse.Namespace) -> float:
    """Slow curriculum for clinical same-disease supervision.

    The schedule lets strict pair alignment stabilize first, introduces
    clinical supervision gradually, then decays it late so exact retrieval can
    recover instead of drifting into broad disease grouping.
    """
    if not getattr(args, "use_clinical_schedule", True):
        return float(args.clinical_weight if epoch >= args.clinical_start else 0.0)

    if epoch < args.clinical_start:
        return 0.0

    ramp_end = max(args.clinical_start, args.clinical_ramp_end)
    if epoch <= ramp_end:
        span = max(1, ramp_end - args.clinical_start)
        ratio = (epoch - args.clinical_start) / span
        return float(
            args.clinical_ramp_start_weight
            + ratio * (args.clinical_peak_weight - args.clinical_ramp_start_weight)
        )

    if epoch <= args.clinical_hold_end:
        return float(args.clinical_peak_weight)
    if epoch <= args.clinical_decay_end:
        return float(args.clinical_mid_weight)
    return float(args.clinical_final_weight)


class StudyBatchSampler(torch.utils.data.Sampler):
    """Deterministic study-level batch sampler used by the final proposed run.

    Each epoch shuffles patient/study IDs with a fixed seed and fills the last
    batch to the configured batch size. This preserves stable batch geometry
    for gradient accumulation without any hard-negative mining branch.
    """

    def __init__(self, dataset: StudyIUXrayDataset, batch_size: int, seed: int = SEED):
        self.pid_to_idx = {s["pid"]: i for i, s in enumerate(dataset.samples)}
        self.all_pids = [s["pid"] for s in dataset.samples]
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return int(math.ceil(len(self.all_pids) / self.batch_size))

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        shuffled = rng.permutation(self.all_pids).tolist()

        for i in range(0, len(shuffled), self.batch_size):
            batch_pids = list(shuffled[i: i + self.batch_size])
            used = set(batch_pids)

            if len(batch_pids) < self.batch_size:
                for pid in rng.permutation(self.all_pids).tolist():
                    if len(batch_pids) >= self.batch_size:
                        break
                    if pid not in used:
                        batch_pids.append(pid)
                        used.add(pid)

            if len(batch_pids) == self.batch_size:
                yield [self.pid_to_idx[p] for p in batch_pids]


def mean_reciprocal_rank(sim_matrix: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """MRR (%) for study-level retrieval; diagonal is the same-patient match."""
    rr = []
    for i in range(sim_matrix.shape[0]):
        gt = gt_mask[i]          # include diagonal: correct match = same patient
        if not gt.any():
            continue
        order = sim_matrix[i].argsort(descending=True)
        for rank, idx in enumerate(order.tolist(), 1):
            if gt[idx]:
                rr.append(1.0 / rank)
                break
    return float(np.mean(rr) * 100) if rr else 0.0


def recall_at_k_valid_queries(
    sim: torch.Tensor,
    gt_mask: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10),
) -> tuple[dict[str, float], int]:
    """Recall@K over queries that have at least one valid positive."""
    valid = gt_mask.any(dim=1)
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return {f"R@{k}": 0.0 for k in ks}, 0
    return base.recall_at_k(sim[valid], gt_mask[valid], ks), n_valid


@torch.no_grad()
def evaluate_study(model: StudyMedicalSwinBERT, loader: DataLoader, tokenizer, device: torch.device):
    model.eval()
    all_ie, all_te, all_pids, all_labs = [], [], [], []

    for batch in loader:
        images = batch["images"].to(device)
        view_mask = batch["view_mask"].to(device)
        view_type_ids = batch["view_type_ids"].to(device)
        tok = tokenizer(
            batch["caption"],
            padding="max_length",
            truncation=True,
            max_length=TEXT_MAX_LEN,
            return_tensors="pt",
        ).to(device)
        ie, te, _, _, _, _, _, _, _ = model(
            images,
            view_mask,
            view_type_ids,
            tok["input_ids"],
            tok["attention_mask"],
        )
        all_ie.append(ie.cpu())
        all_te.append(te.cpu())
        all_pids.extend(batch["pid"])
        all_labs.append(batch["labels"])

    model.train()

    ie = torch.cat(all_ie)
    te = torch.cat(all_te)
    labs = torch.cat(all_labs).float()
    pids = np.array(all_pids)

    sim_i2t = ie @ te.T
    sim_t2i = te @ ie.T
    gt_strict = torch.from_numpy(pids[:, None] == pids[None, :])
    dv = base.labels_to_disease_vecs(labs)
    gt_cluster = build_cluster_mask(dv)
    gt_clinical = build_clinical_cluster_mask(dv)

    strict_i2t   = base.recall_at_k(sim_i2t, gt_strict)
    strict_t2i   = base.recall_at_k(sim_t2i, gt_strict.T)
    cluster_i2t  = base.recall_at_k(sim_i2t, gt_cluster)
    cluster_t2i  = base.recall_at_k(sim_t2i, gt_cluster.T)
    clinical_i2t = base.recall_at_k(sim_i2t, gt_clinical)
    clinical_t2i = base.recall_at_k(sim_t2i, gt_clinical.T)
    clinical_valid_i2t, clinical_valid_i2t_n = recall_at_k_valid_queries(sim_i2t, gt_clinical)
    clinical_valid_t2i, clinical_valid_t2i_n = recall_at_k_valid_queries(sim_t2i, gt_clinical.T)

    sr1  = (strict_i2t["R@1"]   + strict_t2i["R@1"])   / 2
    cr1  = (cluster_i2t["R@1"]  + cluster_t2i["R@1"])  / 2
    ccr1 = (clinical_i2t["R@1"] + clinical_t2i["R@1"]) / 2
    cvcr1 = (clinical_valid_i2t["R@1"] + clinical_valid_t2i["R@1"]) / 2
    mrr  = (mean_reciprocal_rank(sim_i2t, gt_strict) + mean_reciprocal_rank(sim_t2i, gt_strict.T)) / 2
    return (
        strict_i2t,
        strict_t2i,
        cluster_i2t,
        cluster_t2i,
        clinical_i2t,
        clinical_t2i,
        clinical_valid_i2t,
        clinical_valid_t2i,
        clinical_valid_i2t_n,
        clinical_valid_t2i_n,
        sr1,
        cr1,
        ccr1,
        cvcr1,
        mrr,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study-level training for IU-Xray retrieval")
    parser.add_argument("--csv_path", default=CSV_PATH)
    parser.add_argument("--img_dir", default=IMG_DIR)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad_accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--freeze_ep", type=int, default=FREEZE_EP)
    parser.add_argument("--eval_every", type=int, default=EVAL_EVERY)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--split_seed", type=int, default=SEED)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--max_views", type=int, default=MAX_VIEWS)
    parser.add_argument("--cluster_start", type=int, default=CLUSTER_START)
    parser.add_argument("--use_idf_jaccard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_healthy_cluster", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aux_weight", type=float, default=AUX_WEIGHT)
    parser.add_argument("--clinical_start", type=int, default=CLINICAL_START)
    parser.add_argument("--clinical_weight", type=float, default=CLINICAL_WEIGHT)
    parser.add_argument("--use_clinical_schedule", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clinical_ramp_end", type=int, default=25)
    parser.add_argument("--clinical_ramp_start_weight", type=float, default=0.03)
    parser.add_argument("--clinical_peak_weight", type=float, default=0.12)
    parser.add_argument("--clinical_hold_end", type=int, default=60)
    parser.add_argument("--clinical_mid_weight", type=float, default=0.08)
    parser.add_argument("--clinical_decay_end", type=int, default=105)
    parser.add_argument("--clinical_final_weight", type=float, default=0.04)
    parser.add_argument("--clinical_best_min_strict", type=float, default=CLINICAL_BEST_MIN_STRICT)
    parser.add_argument("--balanced_clinical_weight", type=float, default=BALANCED_CLINICAL_WEIGHT)
    parser.add_argument("--min_strict_for_balanced", type=float, default=MIN_STRICT_FOR_BALANCED)
    parser.add_argument("--lr_head", type=float, default=LR_HEAD)
    parser.add_argument("--lr_enc", type=float, default=LR_ENC)
    parser.add_argument("--lr_text", type=float, default=LR_TEXT)
    return parser.parse_args()


def save_history(args, history, best_epoch, best_r1, out_dir):
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
    compat_rows = []
    best_so_far = 0.0
    for h in history:
        best_so_far = max(best_so_far, h["strict_r1"])
        compat_rows.append({
            "epoch": h["epoch"],
            "train_loss": h["loss"],
            "r1_strict": h["strict_r1"],
            "r5_strict": (h["strict_i2t_r5"] + h["strict_t2i_r5"]) / 2,
            "r10_strict": (h["strict_i2t_r10"] + h["strict_t2i_r10"]) / 2,
            "r1_cluster": h["cluster_r1"],
            "r5_cluster": (h["cluster_i2t_r5"] + h["cluster_t2i_r5"]) / 2,
            "r10_cluster": (h["cluster_i2t_r10"] + h["cluster_t2i_r10"]) / 2,
            "r1_clinical_all": h.get("clinical_all_r1", 0.0),
            "r1_clinical_valid": h.get("clinical_valid_r1", 0.0),
            "best_r1_strict": best_so_far,
            "is_best": int(h["epoch"] == best_epoch),
        })
    pd.DataFrame(compat_rows).to_csv(
        os.path.join(out_dir, "training_history_lvtm_compat.csv"),
        index=False,
    )
    with open(os.path.join(out_dir, "run_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"best_epoch={best_epoch}\n")
        f.write(f"best_val_strict_r1={best_r1:.4f}\n")
        f.write(f"batch_size={args.batch_size}\n")
        f.write(f"grad_accum={args.grad_accum}\n")
        f.write(f"effective_batch={args.batch_size * args.grad_accum}\n")


def main():
    global _log_file

    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    _log_file = open(os.path.join(args.out_dir, "train.log"), "w", encoding="utf-8")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log("=" * 72)
    log("Study-level IU-Xray retrieval training")
    log("Goal: strict patient retrieval + cluster-aware false-negative mitigation")
    log("=" * 72)
    log(f"Device       : {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        log(f"GPU          : {props.name}")
        log(f"VRAM         : {props.total_memory // 1024**2} MB")
    log(f"Batch size   : {args.batch_size}")
    log(f"Grad accum   : {args.grad_accum}")
    log(f"Eff batch    : {args.batch_size * args.grad_accum}")
    log(f"Epochs       : {args.epochs}")
    log(f"Seed         : {args.seed}")
    log(f"Split seed   : {args.split_seed}")
    log(f"Freeze ep    : {args.freeze_ep}")
    log(f"Max views    : {args.max_views}")
    log(f"Cluster start: {args.cluster_start}")
    log(f"IDF-Jaccard  : {args.use_idf_jaccard}")
    log(f"Healthy clus.: {args.use_healthy_cluster}")
    log(f"Clinical start: {args.clinical_start}")
    log(f"Clinical w   : {args.clinical_weight}")
    log(f"Clinical schedule: {args.use_clinical_schedule}")
    if args.use_clinical_schedule:
        log(
            "Clinical schedule params: "
            f"start={args.clinical_start}, ramp_end={args.clinical_ramp_end}, "
            f"ramp_start_w={args.clinical_ramp_start_weight}, peak_w={args.clinical_peak_weight}, "
            f"hold_end={args.clinical_hold_end}, mid_w={args.clinical_mid_weight}, "
            f"decay_end={args.clinical_decay_end}, final_w={args.clinical_final_weight}"
        )
    log(f"Out dir      : {args.out_dir}")

    config = {k: v for k, v in vars(args).items()}
    config.update({
        "effective_batch": args.batch_size * args.grad_accum,
        "CLUSTER_START": args.cluster_start,
        "MAX_VIEWS": args.max_views,
        "SPLIT_SEED": args.split_seed,
        "USE_IDF_JACCARD": args.use_idf_jaccard,
        "USE_HEALTHY_CLUSTER": args.use_healthy_cluster,
        "AUX_WEIGHT": args.aux_weight,
        "CLINICAL_START": args.clinical_start,
        "CLINICAL_WEIGHT": args.clinical_weight,
        "USE_CLINICAL_SCHEDULE": args.use_clinical_schedule,
        "CLINICAL_RAMP_END": args.clinical_ramp_end,
        "CLINICAL_RAMP_START_WEIGHT": args.clinical_ramp_start_weight,
        "CLINICAL_PEAK_WEIGHT": args.clinical_peak_weight,
        "CLINICAL_HOLD_END": args.clinical_hold_end,
        "CLINICAL_MID_WEIGHT": args.clinical_mid_weight,
        "CLINICAL_DECAY_END": args.clinical_decay_end,
        "CLINICAL_FINAL_WEIGHT": args.clinical_final_weight,
        "CLINICAL_BEST_MIN_STRICT": args.clinical_best_min_strict,
        "BALANCED_CLINICAL_WEIGHT": args.balanced_clinical_weight,
        "MIN_STRICT_FOR_BALANCED": args.min_strict_for_balanced,
        "CLUSTER_ALPHA": CLUSTER_ALPHA,
        "LR_HEAD": args.lr_head,
        "LR_ENC": args.lr_enc,
        "LR_TEXT": args.lr_text,
    })

    df = pd.read_csv(args.csv_path)
    df["patient_id"] = df["patient_id"].astype(str)
    df_train = base.patient_split(df, "train", seed=args.split_seed)
    df_val = base.patient_split(df, "val", seed=args.split_seed)
    df_test = base.patient_split(df, "test", seed=args.split_seed)
    log(f"Train rows/studies: {len(df_train)} / {df_train['patient_id'].nunique()}")
    log(f"Val rows/studies  : {len(df_val)} / {df_val['patient_id'].nunique()}")
    log(f"Test rows/studies : {len(df_test)} / {df_test['patient_id'].nunique()}")

    train_ds = StudyIUXrayDataset(
        df_train,
        args.img_dir,
        base.get_train_transform(),
        train_mode=True,
        max_views=args.max_views,
    )
    val_ds = StudyIUXrayDataset(
        df_val,
        args.img_dir,
        base.get_val_transform(),
        train_mode=False,
        max_views=args.max_views,
    )
    test_ds = StudyIUXrayDataset(
        df_test,
        args.img_dir,
        base.get_val_transform(),
        train_mode=False,
        max_views=args.max_views,
    )
    log(f"Train/val/test studies: {len(train_ds)} / {len(val_ds)} / {len(test_ds)}")

    study_sampler = StudyBatchSampler(train_ds, args.batch_size, seed=args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=study_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_study,
    )
    log(f"Study batches: {len(train_loader)}")
    config.update({
        "study_batches": len(train_loader),
    })
    val_loader = DataLoader(
        val_ds,
        batch_size=16,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_study,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=16,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_study,
    )

    model = StudyMedicalSwinBERT().to(device)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    criterion = StudyLoss(
        cluster_start=args.cluster_start,
        aux_weight=args.aux_weight,
        clinical_start=args.clinical_start,
        clinical_weight=args.clinical_weight,
        use_idf_jaccard=args.use_idf_jaccard,
        use_healthy_cluster=args.use_healthy_cluster,
    ).to(device)
    scaler = torch.amp.GradScaler("cuda")

    for p in model.backbone_params():
        p.requires_grad = False
    opt = AdamW(model.head_params(), lr=args.lr_head, weight_decay=WEIGHT_DECAY)
    sched = None

    start_epoch = 1
    best_r1 = 0.0
    best_epoch = 0
    best_balanced_score = -1.0
    best_balanced_epoch = 0
    best_clinical_valid_score = -1.0
    best_clinical_valid_epoch = 0
    history = []

    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    for epoch in range(start_epoch, args.epochs + 1):
        if epoch == args.freeze_ep + 1:
            log(f"[PHASE 2] unfreeze at epoch {epoch}")
            for p in model.backbone_params():
                p.requires_grad = True
            opt = AdamW([
                {"params": model.image_encoder.parameters(), "lr": args.lr_enc},
                {"params": model.text_encoder.parameters(), "lr": args.lr_text},
                {"params": model.head_params(), "lr": args.lr_head},
            ], weight_decay=WEIGHT_DECAY)
            planned_batches = len(train_loader)
            remain_steps = max(1, planned_batches * (args.epochs - args.freeze_ep) // args.grad_accum)
            warmup_steps = max(1, int(remain_steps * 0.05))
            sched = SequentialLR(opt, [
                LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
                CosineAnnealingLR(opt, T_max=max(1, remain_steps - warmup_steps), eta_min=1e-8),
            ], milestones=[warmup_steps])

        model.train()
        active_clinical_weight = clinical_weight_for_epoch(epoch, args)
        criterion.set_clinical_weight(active_clinical_weight)
        opt.zero_grad()
        t0 = time.time()
        losses = {"total": [], "main": [], "aux": [], "clinical": []}
        n_batches = 0
        n_opt_steps = 0
        n_sched_steps = 0

        for step, batch in enumerate(train_loader):
            n_batches += 1
            images = batch["images"].to(device)
            view_mask = batch["view_mask"].to(device)
            view_type_ids = batch["view_type_ids"].to(device)
            labels = batch["labels"].to(device)
            disease_vecs = base.labels_to_disease_vecs(labels)
            pids = batch["pid"]

            tok = tokenizer(
                batch["caption"],
                padding="max_length",
                truncation=True,
                max_length=TEXT_MAX_LEN,
                return_tensors="pt",
            ).to(device)

            with torch.amp.autocast("cuda"):
                (
                    img_emb,
                    txt_emb,
                    scale,
                    img_pathology_logits,
                    img_normal_logits,
                    txt_pathology_logits,
                    txt_normal_logits,
                    _,
                    _,
                ) = model(
                    images,
                    view_mask,
                    view_type_ids,
                    tok["input_ids"],
                    tok["attention_mask"],
                )
                logits = scale * (img_emb @ txt_emb.T)
                loss, loss_main, loss_aux, loss_clinical = criterion(
                    logits=logits,
                    img_emb=img_emb,
                    txt_emb=txt_emb,
                    patient_ids=pids,
                    disease_vecs=disease_vecs,
                    img_pathology_logits=img_pathology_logits,
                    img_normal_logits=img_normal_logits,
                    txt_pathology_logits=txt_pathology_logits,
                    txt_normal_logits=txt_normal_logits,
                    epoch=epoch,
                )
                scaled_loss = loss / args.grad_accum

            scaler.scale(scaled_loss).backward()

            losses["total"].append(loss.item())
            losses["main"].append(loss_main.item())
            losses["aux"].append(loss_aux.item())
            losses["clinical"].append(loss_clinical.item())

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD)
                old_scale = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                new_scale = scaler.get_scale()
                optimizer_was_skipped = new_scale < old_scale
                if not optimizer_was_skipped:
                    n_opt_steps += 1
                if epoch > args.freeze_ep and sched is not None and not optimizer_was_skipped:
                    sched.step()
                    n_sched_steps += 1

        elapsed = time.time() - t0
        phase = "Phase1" if epoch <= args.freeze_ep else "Phase2"
        log(
            f"Ep[{epoch:03d}/{args.epochs}] {phase} "
            f"loss={np.mean(losses['total']):.4f} "
            f"(main={np.mean(losses['main']):.4f} "
            f"clinical={np.mean(losses['clinical']):.4f} "
            f"clinical_w={active_clinical_weight:.4f} "
            f"aux={np.mean(losses['aux']):.4f}) "
            f"T={model.logit_scale.exp().item():.1f} "
            f"batches={n_batches} opt_steps={n_opt_steps} t={elapsed:.0f}s"
        )

        do_eval = epoch <= 3 or epoch % args.eval_every == 0 or epoch == args.epochs
        if do_eval:
            log(f"--- Eval epoch {epoch} ---")
            (
                si,
                st,
                ci,
                ct,
                cci,
                cct,
                ccvi,
                ccvt,
                ccvi_n,
                ccvt_n,
                sr1,
                cr1,
                ccr1,
                cvcr1,
                mrr,
            ) = evaluate_study(model, val_loader, tokenizer, device)
            base.log_eval_results(si, st, ci, ct, sr1, cr1)
            log(
                f"  Clinical R@1(all)={ccr1:.2f}%  "
                f"Clinical R@1(valid)={cvcr1:.2f}% "
                f"(i2t_n={ccvi_n}, t2i_n={ccvt_n})  MRR={mrr:.2f}%"
            )
            row = {
                "epoch": epoch,
                "phase": phase,
                "elapsed_s": float(elapsed),
                "num_batches": n_batches,
                "optimizer_steps": n_opt_steps,
                "scheduler_steps": n_sched_steps,
                "loss": float(np.mean(losses["total"])),
                "loss_main": float(np.mean(losses["main"])),
                "loss_aux": float(np.mean(losses["aux"])),
                "loss_clinical": float(np.mean(losses["clinical"])),
                "clinical_weight": active_clinical_weight,
                "strict_r1": sr1,
                "mrr": mrr,
                "cluster_r1": cr1,
                "strict_i2t_r1": si["R@1"], "strict_i2t_r5": si["R@5"], "strict_i2t_r10": si["R@10"],
                "strict_t2i_r1": st["R@1"], "strict_t2i_r5": st["R@5"], "strict_t2i_r10": st["R@10"],
                "cluster_i2t_r1": ci["R@1"], "cluster_i2t_r5": ci["R@5"], "cluster_i2t_r10": ci["R@10"],
                "cluster_t2i_r1": ct["R@1"], "cluster_t2i_r5": ct["R@5"], "cluster_t2i_r10": ct["R@10"],
                "clinical_all_r1": ccr1,
                "clinical_valid_r1": cvcr1,
                "clinical_valid_i2t_r1": ccvi["R@1"],
                "clinical_valid_t2i_r1": ccvt["R@1"],
            }
            balanced_score = (
                sr1 + args.balanced_clinical_weight * cvcr1
                if sr1 >= args.min_strict_for_balanced
                else -1.0
            )
            row["balanced_score"] = balanced_score
            history.append(row)
            if sr1 > best_r1:
                best_r1 = sr1
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "best_r1": best_r1,
                        "strict": {"i2t": si, "t2i": st},
                        "cluster": {"i2t": ci, "t2i": ct},
                        "clinical_all": {"i2t": cci, "t2i": cct},
                        "clinical_valid": {"i2t": ccvi, "t2i": ccvt},
                        "config": config,
                    }, os.path.join(args.out_dir, "best.pt"))
                with open(os.path.join(args.out_dir, "best_summary.json"), "w", encoding="utf-8") as f:
                    json.dump({
                        "best_epoch": epoch,
                        "best_val_strict_r1": best_r1,
                        "best_val_cluster_r1": cr1,
                        "best_val_clinical_all_r1": ccr1,
                        "best_val_clinical_valid_r1": cvcr1,
                        "strict": {"i2t": si, "t2i": st},
                        "cluster": {"i2t": ci, "t2i": ct},
                        "clinical_all": {"i2t": cci, "t2i": cct},
                        "clinical_valid": {"i2t": ccvi, "t2i": ccvt},
                    }, f, indent=2)
                log(f"  * NEW BEST Strict R@1={best_r1:.2f}% @ ep{epoch}")
            else:
                log(f"  (best={best_r1:.2f}% @ ep{best_epoch})")

            if balanced_score > best_balanced_score:
                best_balanced_score = balanced_score
                best_balanced_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "best_balanced_score": best_balanced_score,
                    "strict_r1": sr1,
                    "clinical_valid_r1": cvcr1,
                    "strict": {"i2t": si, "t2i": st},
                    "cluster": {"i2t": ci, "t2i": ct},
                    "clinical_all": {"i2t": cci, "t2i": cct},
                    "clinical_valid": {"i2t": ccvi, "t2i": ccvt},
                    "config": config,
                }, os.path.join(args.out_dir, "best_balanced.pt"))
                with open(os.path.join(args.out_dir, "best_balanced_summary.json"), "w", encoding="utf-8") as f:
                    json.dump({
                        "best_balanced_epoch": epoch,
                        "best_balanced_score": best_balanced_score,
                        "val_strict_r1": sr1,
                        "val_cluster_r1": cr1,
                        "val_clinical_all_r1": ccr1,
                        "val_clinical_valid_r1": cvcr1,
                        "strict": {"i2t": si, "t2i": st},
                        "cluster": {"i2t": ci, "t2i": ct},
                        "clinical_all": {"i2t": cci, "t2i": cct},
                        "clinical_valid": {"i2t": ccvi, "t2i": ccvt},
                    }, f, indent=2)
                log(
                    f"  * NEW BEST Balanced={best_balanced_score:.2f} "
                    f"(strict={sr1:.2f}, clinical_valid={cvcr1:.2f}) @ ep{epoch}"
                )

            clinical_valid_score = cvcr1 if sr1 >= args.clinical_best_min_strict else -1.0
            if clinical_valid_score > best_clinical_valid_score:
                best_clinical_valid_score = clinical_valid_score
                best_clinical_valid_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "best_clinical_valid_r1": best_clinical_valid_score,
                    "strict_r1": sr1,
                    "cluster_r1": cr1,
                    "strict": {"i2t": si, "t2i": st},
                    "cluster": {"i2t": ci, "t2i": ct},
                    "clinical_all": {"i2t": cci, "t2i": cct},
                    "clinical_valid": {"i2t": ccvi, "t2i": ccvt},
                    "config": config,
                }, os.path.join(args.out_dir, "best_clinical_valid.pt"))
                with open(os.path.join(args.out_dir, "best_clinical_valid_summary.json"), "w", encoding="utf-8") as f:
                    json.dump({
                        "best_clinical_valid_epoch": epoch,
                        "best_clinical_valid_r1": best_clinical_valid_score,
                        "val_strict_r1": sr1,
                        "val_cluster_r1": cr1,
                        "val_clinical_all_r1": ccr1,
                        "strict_floor": args.clinical_best_min_strict,
                        "strict": {"i2t": si, "t2i": st},
                        "cluster": {"i2t": ci, "t2i": ct},
                        "clinical_all": {"i2t": cci, "t2i": cct},
                        "clinical_valid": {"i2t": ccvi, "t2i": ccvt},
                    }, f, indent=2)
                log(
                    f"  * NEW BEST ClinicalValid={best_clinical_valid_score:.2f} "
                    f"(strict={sr1:.2f}, floor={args.clinical_best_min_strict:.2f}) @ ep{epoch}"
                )

            save_history(args, history, best_epoch, best_r1, args.out_dir)
            with open(os.path.join(args.out_dir, "latest_metrics.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "epoch": epoch,
                    "phase": phase,
                    "best_epoch": best_epoch,
                    "best_val_strict_r1": best_r1,
                    "best_balanced_epoch": best_balanced_epoch,
                    "best_balanced_score": best_balanced_score,
                    "best_clinical_valid_epoch": best_clinical_valid_epoch,
                    "best_clinical_valid_score": best_clinical_valid_score,
                    "current_val_strict_r1": sr1,
                    "current_val_cluster_r1": cr1,
                    "strict_i2t": si,
                    "strict_t2i": st,
                    "cluster_i2t": ci,
                    "cluster_t2i": ct,
                    "clinical_all_i2t": cci,
                    "clinical_all_t2i": cct,
                    "clinical_valid_i2t": ccvi,
                    "clinical_valid_t2i": ccvt,
                    "clinical_all_r1": ccr1,
                    "clinical_valid_r1": cvcr1,
                    "num_batches": n_batches,
                    "optimizer_steps": n_opt_steps,
                    "balanced_score": balanced_score,
                }, f, indent=2)
            with open(os.path.join(args.out_dir, "progress.log"), "a", encoding="utf-8") as f:
                f.write(
                    " | ".join([
                        f"epoch={epoch}",
                        f"phase={phase}",
                        f"loss={np.mean(losses['total']):.4f}",
                        f"strict_r1={sr1:.2f}",
                        f"strict_r5={((si['R@5'] + st['R@5']) / 2):.2f}",
                        f"strict_r10={((si['R@10'] + st['R@10']) / 2):.2f}",
                        f"cluster_r1={cr1:.2f}",
                        f"cluster_r5={((ci['R@5'] + ct['R@5']) / 2):.2f}",
                        f"cluster_r10={((ci['R@10'] + ct['R@10']) / 2):.2f}",
                        f"clinical_all_r1={ccr1:.2f}",
                        f"clinical_valid_r1={cvcr1:.2f}",
                        f"mrr={mrr:.2f}",
                        f"batches={n_batches}",
                        f"opt_steps={n_opt_steps}",
                        f"clinical_loss={np.mean(losses['clinical']):.4f}",
                        f"clinical_weight={active_clinical_weight:.4f}",
                        f"best_strict_r1={best_r1:.2f}",
                        f"best_balanced={best_balanced_score:.2f}",
                        f"best_clinical_valid={best_clinical_valid_score:.2f}",
                    ]) + "\n"
                )

    log("Final test evaluation")
    best_ckpt_path = os.path.join(args.out_dir, "best.pt")
    if os.path.isfile(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        log(f"Loaded best checkpoint (ep{ckpt['epoch']}, val Strict R@1={ckpt['best_r1']:.2f}%)")

    (
        si,
        st,
        ci,
        ct,
        cci,
        cct,
        ccvi,
        ccvt,
        ccvi_n,
        ccvt_n,
        sr1,
        cr1,
        ccr1,
        cvcr1,
        mrr,
    ) = evaluate_study(model, test_loader, tokenizer, device)
    base.log_eval_results(si, st, ci, ct, sr1, cr1)
    log(
        f"  Clinical R@1(all)={ccr1:.2f}%  "
        f"Clinical R@1(valid)={cvcr1:.2f}% "
        f"(i2t_n={ccvi_n}, t2i_n={ccvt_n})  MRR={mrr:.2f}%"
    )
    with open(os.path.join(args.out_dir, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": "best.pt",
            "test_strict": {"i2t": si, "t2i": st},
            "test_cluster": {"i2t": ci, "t2i": ct},
            "test_clinical_all": {"i2t": cci, "t2i": cct},
            "test_clinical_valid": {"i2t": ccvi, "t2i": ccvt},
            "test_clinical_valid_counts": {"i2t": ccvi_n, "t2i": ccvt_n},
            "best_epoch": best_epoch,
            "best_val_strict_r1": best_r1,
            "mean_strict_r1_test": sr1,
            "mean_cluster_r1_test": cr1,
            "mean_clinical_all_r1_test": ccr1,
            "mean_clinical_valid_r1_test": cvcr1,
            "mrr_test": mrr,
        }, f, indent=2)

    balanced_ckpt_path = os.path.join(args.out_dir, "best_balanced.pt")
    if os.path.isfile(balanced_ckpt_path):
        log("Final test evaluation for balanced checkpoint")
        ckpt = torch.load(balanced_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        log(
            f"Loaded balanced checkpoint (ep{ckpt['epoch']}, "
            f"score={ckpt['best_balanced_score']:.2f}, "
            f"val strict={ckpt['strict_r1']:.2f}%, "
            f"val clinical_valid={ckpt['clinical_valid_r1']:.2f}%)"
        )
        (
            si_b,
            st_b,
            ci_b,
            ct_b,
            cci_b,
            cct_b,
            ccvi_b,
            ccvt_b,
            ccvi_n_b,
            ccvt_n_b,
            sr1_b,
            cr1_b,
            ccr1_b,
            cvcr1_b,
            mrr_b,
        ) = evaluate_study(model, test_loader, tokenizer, device)
        base.log_eval_results(si_b, st_b, ci_b, ct_b, sr1_b, cr1_b)
        log(
            f"  Clinical R@1(all)={ccr1_b:.2f}%  "
            f"Clinical R@1(valid)={cvcr1_b:.2f}% "
            f"(i2t_n={ccvi_n_b}, t2i_n={ccvt_n_b})  MRR={mrr_b:.2f}%"
        )
        with open(os.path.join(args.out_dir, "test_results_balanced.json"), "w", encoding="utf-8") as f:
            json.dump({
                "checkpoint": "best_balanced.pt",
                "test_strict": {"i2t": si_b, "t2i": st_b},
                "test_cluster": {"i2t": ci_b, "t2i": ct_b},
                "test_clinical_all": {"i2t": cci_b, "t2i": cct_b},
                "test_clinical_valid": {"i2t": ccvi_b, "t2i": ccvt_b},
                "test_clinical_valid_counts": {"i2t": ccvi_n_b, "t2i": ccvt_n_b},
                "best_balanced_epoch": best_balanced_epoch,
                "best_balanced_score": best_balanced_score,
                "mean_strict_r1_test": sr1_b,
                "mean_cluster_r1_test": cr1_b,
                "mean_clinical_all_r1_test": ccr1_b,
                "mean_clinical_valid_r1_test": cvcr1_b,
                "mrr_test": mrr_b,
            }, f, indent=2)

    clinical_ckpt_path = os.path.join(args.out_dir, "best_clinical_valid.pt")
    if os.path.isfile(clinical_ckpt_path):
        log("Final test evaluation for clinical-valid checkpoint")
        ckpt = torch.load(clinical_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        log(
            f"Loaded clinical-valid checkpoint (ep{ckpt['epoch']}, "
            f"val clinical_valid={ckpt['best_clinical_valid_r1']:.2f}%, "
            f"val strict={ckpt['strict_r1']:.2f}%)"
        )
        (
            si_c,
            st_c,
            ci_c,
            ct_c,
            cci_c,
            cct_c,
            ccvi_c,
            ccvt_c,
            ccvi_n_c,
            ccvt_n_c,
            sr1_c,
            cr1_c,
            ccr1_c,
            cvcr1_c,
            mrr_c,
        ) = evaluate_study(model, test_loader, tokenizer, device)
        base.log_eval_results(si_c, st_c, ci_c, ct_c, sr1_c, cr1_c)
        log(
            f"  Clinical R@1(all)={ccr1_c:.2f}%  "
            f"Clinical R@1(valid)={cvcr1_c:.2f}% "
            f"(i2t_n={ccvi_n_c}, t2i_n={ccvt_n_c})  MRR={mrr_c:.2f}%"
        )
        with open(os.path.join(args.out_dir, "test_results_clinical_valid.json"), "w", encoding="utf-8") as f:
            json.dump({
                "checkpoint": "best_clinical_valid.pt",
                "test_strict": {"i2t": si_c, "t2i": st_c},
                "test_cluster": {"i2t": ci_c, "t2i": ct_c},
                "test_clinical_all": {"i2t": cci_c, "t2i": cct_c},
                "test_clinical_valid": {"i2t": ccvi_c, "t2i": ccvt_c},
                "test_clinical_valid_counts": {"i2t": ccvi_n_c, "t2i": ccvt_n_c},
                "best_clinical_valid_epoch": best_clinical_valid_epoch,
                "best_clinical_valid_score": best_clinical_valid_score,
                "mean_strict_r1_test": sr1_c,
                "mean_cluster_r1_test": cr1_c,
                "mean_clinical_all_r1_test": ccr1_c,
                "mean_clinical_valid_r1_test": cvcr1_c,
                "mrr_test": mrr_c,
            }, f, indent=2)

    log(f"DONE. Outputs saved to: {args.out_dir}")
    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()
