"""
Task 3: Hierarchical Clustering-Guided Contrastive Learning
===========================================================
Paper contributions vs MedCLIP (Wang et al., NeurIPS 2022):

  [C1] IDF-Weighted Jaccard Similarity
       - Rare diseases (Pneumothorax 2%, Fracture 1.8%) contribute MORE to
         cluster similarity than common findings (Cardiomegaly 9%)
       - Clinically motivated: rare findings are more discriminative
       - True Jaccard formula, not cosine similarity

  [C2] Healthy Prototype Cluster
       - 57.9% No Finding patients form their own cluster (not excluded)
       - Prevents scatter of majority class in embedding space

  [C3] Dual Prototype Bank with EMA
       - K+1 disease cluster centroids updated via Exponential Moving Average
       - Multi-morbid patients receive mixture assignment (not hard cluster)
       - Prototype alignment loss pulls embeddings toward disease centroids
       - Activates after warm-up (configurable epoch threshold)

Three-level positive hierarchy:
  L1: Same patient (hard positive, weight = 1-alpha)
  L2: Same disease cluster (IDF-Jaccard soft positive, weight = alpha)
  L3: Prototype alignment (regularization, weight = proto_weight)

CrossViewLoss is computed separately in train.py.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── IU-Xray CheXpert label order (must match CHEXPERT_COLS in dataset.py) ──────
CHEXPERT_COLS = [
    'Cardiomegaly', 'Lung Lesion', 'Lung Opacity', 'Edema',
    'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
    'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices',
]

# Disease frequencies from IU-Xray training set (v8_clean.csv)
# Used to compute IDF weights: rare diseases count MORE toward similarity
_DISEASE_FREQ = [
    0.090,  # Cardiomegaly
    0.138,  # Lung Lesion
    0.129,  # Lung Opacity
    0.025,  # Edema
    0.055,  # Consolidation
    0.022,  # Pneumonia
    0.100,  # Atelectasis
    0.020,  # Pneumothorax    ← highest IDF weight (most discriminative)
    0.057,  # Pleural Effusion
    0.009,  # Pleural Other   ← highest IDF weight (rarest pathology)
    0.018,  # Fracture
    0.031,  # Support Devices ← low IDF (device, not disease)
]


def _build_idf_weights(device: torch.device) -> torch.Tensor:
    """
    IDF weight for each disease: w_k = log(1 / freq_k).
    Normalized to sum to 1 for numerical stability.
    """
    freq = torch.tensor(_DISEASE_FREQ, dtype=torch.float32, device=device)
    idf  = torch.log(1.0 / freq.clamp(min=1e-4))   # (K,)
    return idf / idf.sum()                           # normalized, sums to 1


def build_positive_mask(patient_ids: list, device: torch.device) -> torch.Tensor:
    """pos_mask[i,j] = True iff patient_ids[i] == patient_ids[j]."""
    B = len(patient_ids)
    mask = torch.zeros(B, B, dtype=torch.bool, device=device)
    for i in range(B):
        for j in range(B):
            if patient_ids[i] == patient_ids[j]:
                mask[i, j] = True
    return mask


def idf_weighted_jaccard(dv: torch.Tensor, idf_weights: torch.Tensor) -> torch.Tensor:
    """
    True IDF-weighted Jaccard similarity for binary disease vectors.

    J_w(A, B) = Σ_k w_k·min(A_k,B_k) / Σ_k w_k·max(A_k,B_k)
              = weighted_intersection / weighted_union

    For binary vectors: min(a,b) = a AND b = a*b
                        max(a,b) = a OR b  = a+b-a*b

    Args:
        dv          : (B, K) binary float tensor of CheXpert labels
        idf_weights : (K,)   IDF weights (pre-computed, normalized)

    Returns:
        (B, B) similarity matrix, range [0, 1], symmetric
    """
    w = idf_weights                                       # (K,)

    # Weighted intersection: sum_k w_k * a_k * b_k
    # Computed as (a*sqrt(w)) @ (b*sqrt(w)).T
    sqrt_w  = w.sqrt()
    inter   = (dv * sqrt_w) @ (dv * sqrt_w).T            # (B, B)

    # Weighted union: a_sum + b_sum - intersection
    a_sum = (dv * w).sum(-1)                              # (B,)
    union = a_sum.unsqueeze(1) + a_sum.unsqueeze(0) - inter   # (B, B)

    return inter / (union + 1e-8)                        # (B, B)


# ── Loss Components ────────────────────────────────────────────────────────────

class CrossViewLoss(nn.Module):
    """
    InfoNCE between frontal and lateral embeddings of the same patients.
    Returns 0 if fewer than 2 pairs (can't form meaningful negatives).
    """
    def forward(self, frontal: torch.Tensor, lateral: torch.Tensor,
                logit_scale: torch.Tensor) -> torch.Tensor:
        n = frontal.shape[0]
        if n < 2:
            return frontal.sum() * 0.0
        sim    = logit_scale * (frontal @ lateral.T)
        labels = torch.arange(n, device=frontal.device)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2


class MultiPositiveInfoNCELoss(nn.Module):
    """
    InfoNCE with multi-positive support: all same-patient pairs are positives.

    Used as the base loss throughout training.
    Correctly handles PatientPairBatchSampler where frontal+lateral of the
    same patient must BOTH be positives.
    """
    def forward(self, logits: torch.Tensor, patient_ids: list) -> torch.Tensor:
        device   = logits.device
        pos_mask = build_positive_mask(patient_ids, device).float()
        n_pos    = pos_mask.sum(-1).clamp(min=1)

        log_i2t  = F.log_softmax(logits,   dim=-1)
        loss_i2t = -(pos_mask * log_i2t).sum(-1) / n_pos

        log_t2i  = F.log_softmax(logits.T, dim=-1)
        n_pos_t  = pos_mask.T.sum(-1).clamp(min=1)
        loss_t2i = -(pos_mask.T * log_t2i).sum(-1) / n_pos_t

        return (loss_i2t.mean() + loss_t2i.mean()) / 2


class HierarchicalClusteringLoss(nn.Module):
    """
    [C1] + [C2]: IDF-Weighted Jaccard with Healthy Cluster.

    Replaces MedCLIP-style binary cosine similarity with:
      - True IDF-weighted Jaccard (rare diseases count more)
      - Healthy cluster: No Finding patients cluster together (not excluded)
      - Soft positive blend: (1-alpha)*hard + alpha*idf_jaccard
    """

    def __init__(self, alpha: float = 0.4, temperature: float = 1.0):
        """
        Args:
            alpha       : weight of IDF-Jaccard soft labels (0=hard only, 1=soft only)
            temperature : softmax temperature for soft label distribution
                          higher → smoother, lower → sharper cluster boundaries
        """
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, logits: torch.Tensor, patient_ids: list,
                disease_vecs: torch.Tensor) -> torch.Tensor:
        device = logits.device
        dv     = disease_vecs.float()

        # ── L1: Hard same-patient positives ───────────────────────────────────
        pos_mask = build_positive_mask(patient_ids, device).float()
        n_pos    = pos_mask.sum(-1, keepdim=True).clamp(min=1)
        hard     = pos_mask / n_pos                        # (B, B)

        # ── L2: IDF-Jaccard soft positives ────────────────────────────────────
        idf_w   = _build_idf_weights(device)               # (K,)
        sim_mat = idf_weighted_jaccard(dv, idf_w)          # (B, B), [0,1]

        # [C2] Healthy cluster: No Finding patients are similar to each other
        # (similarity = 0.4 — weaker than pathological matches to avoid over-smoothing)
        is_normal    = (dv.sum(-1) == 0)                   # (B,)
        both_normal  = (is_normal.unsqueeze(1) & is_normal.unsqueeze(0)).float()
        sim_mat      = sim_mat + both_normal * 0.4
        sim_mat      = sim_mat.clamp(max=1.0)

        # Soft label distribution via temperature scaling
        soft = F.softmax(sim_mat / self.temperature, dim=-1)   # (B, B)

        # Blended target: same-patient hard label + cluster soft label
        labels = (1.0 - self.alpha) * hard + self.alpha * soft  # (B, B)

        loss_i2t = -(labels   * F.log_softmax(logits,   dim=-1)).sum(-1).mean()
        loss_t2i = -(labels.T * F.log_softmax(logits.T, dim=-1)).sum(-1).mean()
        return (loss_i2t + loss_t2i) / 2


class DualPrototypeBank(nn.Module):
    """
    [C3]: Dual Prototype Bank with EMA updates.

    Maintains K+1 disease cluster centroids in embedding space:
      proto[0]   = "Healthy" centroid (No Finding patients)
      proto[1..K] = per-disease centroids (one per CheXpert label)

    Multi-morbid patients (multiple diseases) receive proportional
    mixture assignment — not hard-assigned to a single cluster.

    Activates only after 'start_epoch' to avoid bad early anchors.
    """

    def __init__(self, n_clusters: int = 12, embed_dim: int = 512,
                 momentum: float = 0.95, proto_weight: float = 0.05):
        """
        Args:
            n_clusters  : number of disease clusters (K), excluding healthy
            embed_dim   : embedding dimension
            momentum    : EMA decay for prototype update (higher = slower update)
            proto_weight: weight of prototype alignment loss
        """
        super().__init__()
        self.n_clusters   = n_clusters
        self.proto_weight  = proto_weight
        self.momentum     = momentum

        # K+1 prototypes: [0]=healthy, [1..K]=disease
        self.register_buffer(
            'prototypes',
            F.normalize(torch.randn(n_clusters + 1, embed_dim), dim=-1)
        )
        self.register_buffer('initialized', torch.tensor(False))

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor, disease_vecs: torch.Tensor):
        """EMA update of cluster prototypes from current batch embeddings."""
        dv        = disease_vecs.float()
        is_normal = (dv.sum(-1) == 0)                      # (B,)

        # Prototype 0: healthy cluster
        if is_normal.sum() > 0:
            mean_h = F.normalize(embeddings[is_normal].mean(0), dim=-1)
            self.prototypes[0] = F.normalize(
                self.momentum * self.prototypes[0] + (1 - self.momentum) * mean_h, dim=-1
            )

        # Prototypes 1..K: disease clusters
        for k in range(self.n_clusters):
            mask = dv[:, k] > 0
            if mask.sum() > 0:
                mean_k = F.normalize(embeddings[mask].mean(0), dim=-1)
                self.prototypes[k + 1] = F.normalize(
                    self.momentum * self.prototypes[k + 1] + (1 - self.momentum) * mean_k,
                    dim=-1
                )

        self.initialized.fill_(True)

    def _soft_assignment(self, disease_vecs: torch.Tensor) -> torch.Tensor:
        """
        Compute soft prototype assignment for each sample.

        Returns (B, K+1) assignment weights:
          - Normal samples: weight 1.0 on proto[0], 0 elsewhere
          - Pathological: proportional to number of active disease labels
          - Multi-morbid: mixture across multiple disease prototypes
        """
        dv        = disease_vecs.float()
        B         = dv.shape[0]
        is_normal = (dv.sum(-1) == 0)

        assign = torch.zeros(B, self.n_clusters + 1, device=dv.device)

        # Healthy samples → proto[0] fully
        assign[is_normal, 0] = 1.0

        # Pathological → proportional to active labels (prototypes 1..K)
        path_sum = dv.sum(-1, keepdim=True).clamp(min=1.0)
        assign[:, 1:] = dv / path_sum                      # (B, K)
        assign[:, 1:] *= (~is_normal).float().unsqueeze(-1)

        return assign  # (B, K+1)

    def alignment_loss(self, img_emb: torch.Tensor,
                       txt_emb: torch.Tensor,
                       disease_vecs: torch.Tensor) -> torch.Tensor:
        """
        Pull image and text embeddings toward their cluster prototype targets.

        target_i = Σ_k assign[i,k] * proto[k]  (weighted mixture)
        loss = 1 - cosine_similarity(embedding, target)
        """
        if not self.initialized:
            return img_emb.sum() * 0.0

        assign = self._soft_assignment(disease_vecs)           # (B, K+1)
        target = assign @ self.prototypes                      # (B, D)
        target = F.normalize(target, dim=-1)

        loss_img = (1.0 - F.cosine_similarity(img_emb, target.detach())).mean()
        loss_txt = (1.0 - F.cosine_similarity(txt_emb, target.detach())).mean()
        return (loss_img + loss_txt) / 2


# ── Combined Loss — Orchestrates All Components ────────────────────────────────

class CombinedLoss(nn.Module):
    """
    Training phases:
      Phase 1 (epoch 0 → cluster_start-1):
          MultiPositiveInfoNCE only (basic image-text alignment warm-up)
          Uses correct multi-positive labels from start (no warm-up bug)

      Phase 2 (epoch cluster_start → proto_start-1):
          HierarchicalClusteringLoss [C1+C2]
          IDF-Jaccard soft positives active

      Phase 3 (epoch proto_start → end):
          + DualPrototypeBank alignment [C3]
          Prototypes must be initialized from Phase 2 embeddings

    CrossViewLoss is added externally in train.py with CROSS_VIEW_WEIGHT.

    Args:
        n_clusters    : number of disease clusters (12 for IU-Xray)
        embed_dim     : projection dimension (512)
        alpha         : IDF-Jaccard blend weight (0.4 recommended)
        cluster_temp  : temperature for soft label distribution (1.0)
        proto_momentum: EMA decay for prototype bank (0.95)
        proto_weight  : weight of prototype alignment loss (0.05)
        cluster_start : epoch to activate clustering loss
        proto_start   : epoch to activate prototype alignment
    """

    def __init__(self, n_clusters: int = 12, embed_dim: int = 512,
                 alpha: float = 0.4, cluster_temp: float = 1.0,
                 proto_momentum: float = 0.95, proto_weight: float = 0.05,
                 cluster_start: int = 5, proto_start: int = 30):
        super().__init__()

        self.cluster_start = cluster_start
        self.proto_start   = proto_start

        self.mp_infonce   = MultiPositiveInfoNCELoss()
        self.cluster_loss = HierarchicalClusteringLoss(
            alpha=alpha, temperature=cluster_temp
        )
        self.proto_bank   = DualPrototypeBank(
            n_clusters=n_clusters, embed_dim=embed_dim,
            momentum=proto_momentum, proto_weight=proto_weight
        )

    def forward(self, logits: torch.Tensor, img_emb: torch.Tensor,
                txt_emb: torch.Tensor, patient_ids: list,
                disease_vecs: torch.Tensor, epoch: int = 0):
        """
        Returns:
            total_loss   : scalar loss for backward()
            loss_main    : clustering or mp_infonce component
            loss_proto   : prototype alignment component (0 if inactive)
        """
        if epoch < self.cluster_start:
            # Phase 1: basic multi-positive InfoNCE
            loss_main  = self.mp_infonce(logits, patient_ids)
            loss_proto = torch.tensor(0.0, device=logits.device)

        else:
            # Phase 2+: IDF-Weighted Jaccard clustering [C1+C2]
            loss_main = self.cluster_loss(logits, patient_ids, disease_vecs)

            # Phase 3: + prototype bank alignment [C3]
            if epoch >= self.proto_start:
                # Update prototypes with detached embeddings (no gradient through update)
                all_emb = torch.cat([img_emb.detach(), txt_emb.detach()], dim=0)
                all_dv  = torch.cat([disease_vecs, disease_vecs], dim=0)
                self.proto_bank.update(all_emb, all_dv)

                loss_proto = self.proto_bank.alignment_loss(img_emb, txt_emb, disease_vecs)
            else:
                loss_proto = torch.tensor(0.0, device=logits.device)

        total = loss_main + self.proto_bank.proto_weight * loss_proto
        return total, loss_main, loss_proto
