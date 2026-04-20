import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer
import timm

class ProjectionHead(nn.Module):
    """MLP Projection Head (Linear -> BN -> GELU -> Dropout -> Linear)"""
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, out_dim)
        )

    def forward(self, x, normalize=True):
        x = self.proj(x)
        if normalize:
            x = F.normalize(x, dim=-1)
        return x

class MedicalSwinBERT(nn.Module):
    """
    SwinV2-Base Image Encoder + Bio_ClinicalBERT Text Encoder
    with Clustering-Guided Projection Heads.
    """
    def __init__(self, img_size=512, embed_dim=512, hid_dim=1024, dropout=0.3):
        super().__init__()

        # --- 1. Image Encoder (SwinV2 Base 384x384) ---
        self.image_encoder = AutoModel.from_pretrained('microsoft/swinv2-base-patch4-window12to24-192to384-22kto1k-ft')
        
        # Enable gradient checkpointing for memory efficiency
        try:
            self.image_encoder.gradient_checkpointing_enable()
        except Exception:
            pass
        
        # Detect actual pooler_output dim via dummy forward
        # (config.hidden_size may not match actual output for SwinV2 variants)
        with torch.no_grad():
            _dummy = torch.zeros(1, 3, img_size, img_size)
            _o = self.image_encoder(pixel_values=_dummy)
            img_feature_dim = _o.pooler_output.shape[-1]
        print(f"  [model_swin] SwinV2 pooler_output dim: {img_feature_dim}")

        # --- 2. Text Encoder (Bio_ClinicalBERT) ---
        self.text_encoder = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        self.text_encoder.gradient_checkpointing_enable()
        txt_feature_dim = self.text_encoder.config.hidden_size  # 768

        # --- 3. Projection Heads (MLPs) ---
        self.img_proj = ProjectionHead(img_feature_dim, hid_dim, embed_dim, dropout)
        self.txt_proj = ProjectionHead(txt_feature_dim, hid_dim, embed_dim, dropout)

        # Learnable Temperature
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07)))

    def encode_image(self, images):
        """Returns L2-normalized image embeddings."""
        out = self.image_encoder(pixel_values=images)
        features = out.pooler_output  # (B, 1024)
        return self.img_proj(features, normalize=True)

    def encode_text(self, input_ids, attention_mask):
        """Returns L2-normalized text embeddings based on [CLS] token."""
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = out.last_hidden_state[:, 0, :]  # (B, 768)
        return self.txt_proj(cls_token, normalize=True)

    def forward(self, images, input_ids, attention_mask):
        img_emb = self.encode_image(images)
        txt_emb = self.encode_text(input_ids, attention_mask)
        return img_emb, txt_emb

    def get_logit_scale(self):
        import numpy as np
        with torch.no_grad():
            self.logit_scale.data.clamp_(min=np.log(1.0), max=np.log(100.0))
        return self.logit_scale.exp()

    # --- Parameter Groups for Optimizer ---
    def get_backbone_params(self):
        """Low LR params"""
        return list(self.image_encoder.parameters()) + list(self.text_encoder.parameters())

    def get_head_params(self):
        """Normal LR params"""
        return list(self.img_proj.parameters()) + list(self.txt_proj.parameters()) + [self.logit_scale]

def init_tokenizer():
    return AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
