"""
Contrastive loss functions for medical image-text retrieval.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class InfoNCELoss(nn.Module):
    """
    Standard symmetric InfoNCE (CLIP-style) loss.
    Stable and proven to work.
    """
    
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, logits):
        """
        Args:
            logits: (B, B) similarity matrix
        Returns:
            loss: scalar
        """
        B = logits.shape[0]
        labels = torch.arange(B, device=logits.device)
        
        # Image -> Text
        loss_i2t = self.cross_entropy(logits, labels)
        
        # Text -> Image
        loss_t2i = self.cross_entropy(logits.T, labels)
        
        return (loss_i2t + loss_t2i) / 2


class SoftContrastiveLoss(nn.Module):
    """
    Disease-aware soft contrastive loss.
    Uses disease similarity as soft labels.
    """
    
    def __init__(self, alpha=0.5, disease_temperature=0.5):
        super().__init__()
        self.alpha = alpha
        self.disease_temperature = disease_temperature
    
    def forward(self, logits, disease_vecs_batch, disease_vecs_all):
        B, N = logits.shape
        device = logits.device
        
        hard_labels = torch.eye(B, N, device=device)
        
        # Disease similarity
        dv_b = F.normalize(disease_vecs_batch + 1e-8, dim=-1)
        dv_a = F.normalize(disease_vecs_all + 1e-8, dim=-1)
        disease_sim = torch.mm(dv_b, dv_a.T)
        disease_sim = disease_sim / self.disease_temperature
        
        soft_labels = F.softmax(disease_sim, dim=-1)
        labels = self.alpha * soft_labels + (1 - self.alpha) * hard_labels
        
        # Image -> Text
        log_probs = F.log_softmax(logits.float(), dim=-1)
        loss_i2t = -(labels * log_probs).sum(dim=-1).mean()
        
        # Text -> Image
        logits_t2i = logits.T.float()
        labels_t2i = labels.T
        log_probs_t2i = F.log_softmax(logits_t2i, dim=-1)
        loss_t2i = -(labels_t2i * log_probs_t2i).sum(dim=-1).mean()
        
        return (loss_i2t + loss_t2i) / 2


class MultiGranularityLoss(nn.Module):
    """Local feature alignment."""
    
    def __init__(self, local_weight=0.3):
        super().__init__()
        self.local_weight = local_weight
    
    def forward(self, local_image, local_text, attention_mask):
        if local_image is None or local_text is None:
            return torch.tensor(0.0, device='cuda')
        
        local_image = F.normalize(local_image, dim=-1)
        local_text = F.normalize(local_text, dim=-1)
        
        sim = torch.bmm(local_image, local_text.transpose(1, 2))
        mask = attention_mask.unsqueeze(1).float()
        sim = sim * mask + (1 - mask) * (-1e9)
        
        patch_best = sim.max(dim=-1)[0]
        loss = -patch_best.mean()
        
        return loss * self.local_weight


class CombinedLoss(nn.Module):
    """
    Combined loss with curriculum learning:
    Phase 1 (epochs 0-9): Standard InfoNCE only
    Phase 2 (epochs 10+): Add soft contrastive + local alignment
    """
    
    def __init__(self, alpha=0.5, disease_temperature=0.5, local_weight=0.3):
        super().__init__()
        self.info_nce = InfoNCELoss()
        self.soft_contrastive = SoftContrastiveLoss(alpha, disease_temperature)
        self.local_loss = MultiGranularityLoss(local_weight)
        self.alpha = alpha
        self.local_weight = local_weight
    
    def forward(self, logits, disease_vecs_batch, disease_vecs_all,
                local_image=None, local_text=None, attention_mask=None,
                epoch=0, total_epochs=50):
        
        # Curriculum: start with pure InfoNCE, gradually add soft contrastive
        if epoch < 10:
            # Phase 1: Pure InfoNCE (stable)
            loss_global = self.info_nce(logits)
            loss_local = torch.tensor(0.0, device=logits.device)
        else:
            # Phase 2: Blend InfoNCE + Soft Contrastive
            progress = min((epoch - 10) / 20, 1.0)  # Ramp up over 20 epochs
            
            loss_info = self.info_nce(logits)
            loss_soft = self.soft_contrastive(logits, disease_vecs_batch, disease_vecs_all)
            loss_global = (1 - progress) * loss_info + progress * loss_soft
            
            # Add local alignment gradually
            if local_image is not None and local_text is not None:
                loss_local = self.local_loss(local_image, local_text, attention_mask) * progress
            else:
                loss_local = torch.tensor(0.0, device=logits.device)
        
        return loss_global + loss_local, loss_global, loss_local
