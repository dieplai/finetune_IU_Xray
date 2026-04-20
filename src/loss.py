"""
Loss functions for SwinV2 + Bio_ClinicalBERT IU-Xray retrieval.

Training phases:
  Phase 1 (epoch  0- 9): MultiPositiveInfoNCE
  Phase 2 (epoch 10-19): + LocalAlignmentLoss (weight ramps 0 -> LOCAL_WEIGHT)
  Phase 3 (epoch 20+) : + DiseaseAware soft labels (20% blend)

CrossViewLoss is computed separately in train.py using the frontal/lateral
pair indices from PatientPairBatchSampler, then added with CROSS_VIEW_WEIGHT.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config


def build_positive_mask(captions, device):
    """pos_mask[i,j] = True iff caption[i] == caption[j]."""
    B = len(captions)
    pos_mask = torch.zeros(B, B, dtype=torch.bool, device=device)
    for i in range(B):
        for j in range(B):
            if captions[i] == captions[j]:
                pos_mask[i, j] = True
    return pos_mask


class CrossViewLoss(nn.Module):
    """
    InfoNCE between frontal and lateral embeddings of the same patients.

    frontal_embeds: (n_pairs, D) L2-normalized image embeddings
    lateral_embeds: (n_pairs, D) L2-normalized image embeddings
    logit_scale   : scalar temperature (exp of learnable parameter)

    Returns 0 if n_pairs < 2 (can't form meaningful negatives).
    """

    def forward(self, frontal_embeds, lateral_embeds, logit_scale):
        n = frontal_embeds.shape[0]
        if n < 2:
            return frontal_embeds.sum() * 0.0   # differentiable zero

        sim      = logit_scale * (frontal_embeds @ lateral_embeds.T)   # (n, n)
        labels   = torch.arange(n, device=frontal_embeds.device)
        loss_f2l = F.cross_entropy(sim, labels)
        loss_l2f = F.cross_entropy(sim.T, labels)
        return (loss_f2l + loss_l2f) / 2


class MultiPositiveInfoNCELoss(nn.Module):
    """
    InfoNCE that treats all same-caption pairs as positives.
    loss_i = -log( sum_{j in pos} exp(s_ij) / sum_j exp(s_ij) )
    """

    def forward(self, logits, captions):
        device   = logits.device
        pos_mask = build_positive_mask(captions, device).float()
        n_pos    = pos_mask.sum(-1).clamp(min=1)

        log_sm_i2t = F.log_softmax(logits, dim=-1)
        loss_i2t   = -(pos_mask * log_sm_i2t).sum(-1) / n_pos

        log_sm_t2i = F.log_softmax(logits.T, dim=-1)
        n_pos_t    = pos_mask.T.sum(-1).clamp(min=1)
        loss_t2i   = -(pos_mask.T * log_sm_t2i).sum(-1) / n_pos_t

        return (loss_i2t.mean() + loss_t2i.mean()) / 2


class LocalAlignmentLoss(nn.Module):
    """
    Patch-level image <-> token-level text alignment.

    Projects SwinV2 spatial tokens and BERT token states to a shared
    PROJECTION_DIM space, then computes soft attention to build context
    vectors, then InfoNCE on the aggregated global context.

    img_tokens : (B, n_img, D_img)  -- SwinV2 spatial tokens (64 patches, 1024-dim)
    txt_tokens : (B, n_txt, D_txt)  -- BERT token hidden states (L, 768-dim)
    attn_mask  : (B, n_txt)         -- BERT attention mask (1=real, 0=pad)
    logit_scale: scalar temperature
    """

    def __init__(self, img_dim=None, txt_dim=None, proj_dim=None):
        super().__init__()
        img_dim  = img_dim  or config.VISION_EMBED_DIM
        txt_dim  = txt_dim  or config.TEXT_EMBED_DIM
        proj_dim = proj_dim or config.PROJECTION_DIM
        self.img_proj = nn.Linear(img_dim, proj_dim, bias=False)
        self.txt_proj = nn.Linear(txt_dim, proj_dim, bias=False)

    def forward(self, img_tokens, txt_tokens, attn_mask, logit_scale):
        img_p = F.normalize(self.img_proj(img_tokens), dim=-1)   # (B, n_img, D)
        txt_p = F.normalize(self.txt_proj(txt_tokens), dim=-1)   # (B, n_txt, D)

        # Zero out padding tokens before attention
        if attn_mask is not None:
            mask  = attn_mask.unsqueeze(-1).float()               # (B, n_txt, 1)
            txt_p = txt_p * mask

        # Soft attention: img patches attend over text tokens
        att = torch.bmm(img_p, txt_p.transpose(1, 2))            # (B, n_img, n_txt)
        if attn_mask is not None:
            att = att.masked_fill(attn_mask.unsqueeze(1) == 0, -1e9)
        att_w   = F.softmax(att, dim=-1)                          # (B, n_img, n_txt)
        img_ctx = att_w.bmm(txt_p)                                # (B, n_img, D)

        # Global aggregation
        img_g = F.normalize(img_ctx.mean(dim=1), dim=-1)          # (B, D)
        txt_g = F.normalize(txt_p.mean(dim=1),   dim=-1)          # (B, D)

        sim    = logit_scale * (img_g @ txt_g.T)                  # (B, B)
        labels = torch.arange(img_g.shape[0], device=sim.device)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2


class DiseaseAwareLoss(nn.Module):
    """
    Clustering-Guided soft labels: blend hard multi-positive labels with
    Jaccard disease similarity from CheXpert cluster vectors.

    Key design for paper correctness:
      - Samples with NO pathological findings (disease_vec.sum() == 0) are
        treated as hard-label only — they don't generate cluster similarity.
        This prevents 'No Finding' (57.9%) from over-smoothing the loss.
      - Samples WITH pathological findings use Jaccard similarity as soft
        positive weights — same-disease samples from different patients
        are NOT treated as hard negatives.
    """

    def __init__(self, alpha=0.2, disease_temperature=0.5):
        super().__init__()
        self.alpha = alpha
        self.dt    = disease_temperature

    def forward(self, logits, captions, disease_vecs):
        device   = logits.device
        B        = logits.shape[0]

        # Hard multi-positive mask (same caption = same patient-view pair)
        pos_mask = build_positive_mask(captions, device).float()
        n_pos    = pos_mask.sum(-1, keepdim=True).clamp(min=1)
        hard     = pos_mask / n_pos

        # Jaccard similarity via dot product of L2-normalized binary vectors
        dv = disease_vecs.float()

        # Mask: which samples have at least 1 pathological finding?
        # Shape: (B,) — True if sample has any finding
        has_finding = (dv.sum(dim=-1) > 0)           # (B,)
        pair_has_finding = (
            has_finding.unsqueeze(1) & has_finding.unsqueeze(0)
        ).float()                                     # (B, B): 1 only if BOTH have findings

        # Jaccard similarity (normalized dot product for binary vectors)
        dv_norm = F.normalize(dv + 1e-8, dim=-1)
        jaccard = torch.mm(dv_norm, dv_norm.T)        # (B, B), range [0, 1]

        # Soft labels: only for pathological pairs; normal pairs get 0
        soft_raw = jaccard * pair_has_finding         # zero out normal-normal pairs
        soft     = F.softmax(soft_raw / self.dt, dim=-1)

        # Blended target: alpha% clustering-guided + (1-alpha)% hard label
        labels = (1 - self.alpha) * hard + self.alpha * soft

        loss_i2t = -(labels   * F.log_softmax(logits,   dim=-1)).sum(-1).mean()
        loss_t2i = -(labels.T * F.log_softmax(logits.T, dim=-1)).sum(-1).mean()
        return (loss_i2t + loss_t2i) / 2


class CombinedLoss(nn.Module):
    """
    Training phases:
      Phase 1 (epoch 0-2)  : MultiPositiveInfoNCE only (warm-up alignment)
      Phase 2 (epoch 3+)   : + Clustering-Guided DiseaseAwareLoss (20% blend)
                              Activates early to reduce false negatives from start
      Phase 3 (epoch 10+)  : + LocalAlignment (ramp-up: 0 -> LOCAL_WEIGHT over 10 ep)

    CrossViewLoss is NOT computed here; train.py adds it externally.
    """

    def __init__(self):
        super().__init__()
        self.mp_infonce   = MultiPositiveInfoNCELoss()
        self.local_loss   = LocalAlignmentLoss()
        self.disease_loss = DiseaseAwareLoss(
            alpha=config.DISEASE_ALPHA,
            disease_temperature=config.DISEASE_TEMPERATURE,
        )

    def forward(self, logits, captions, disease_vecs, img_tokens, txt_tokens,
                attn_mask, logit_scale, epoch=0):

        # Base loss (multi-positive InfoNCE, optionally blended with disease-aware)
        # Epoch threshold reads from config so it's consistent with NUM_EPOCHS
        disease_epoch = getattr(__import__('config'), 'DISEASE_EPOCH', 15)
        if epoch >= disease_epoch:
            loss_mp      = self.mp_infonce(logits, captions)
            loss_disease = self.disease_loss(logits, captions, disease_vecs)
            loss_main    = 0.8 * loss_mp + 0.2 * loss_disease
        else:
            loss_main = self.mp_infonce(logits, captions)

        # Local alignment (ramp-up from epoch 10 to 20)
        if epoch >= 10 and img_tokens is not None and txt_tokens is not None:
            ramp       = min((epoch - 10) / 10.0, 1.0)   # linearly 0 -> 1
            loss_local = self.local_loss(img_tokens, txt_tokens, attn_mask, logit_scale)
            total      = loss_main + config.LOCAL_WEIGHT * ramp * loss_local
        else:
            loss_local = torch.tensor(0.0, device=logits.device)
            total      = loss_main

        return total, loss_main, loss_local
