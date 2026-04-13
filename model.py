"""
Medical Image-Text Retrieval Model
Vision: SwinV2-CR-Small (pretrained on ImageNet-21K)
Text: PubMedBERT
Multi-granularity (global + local) alignment.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoModel, AutoTokenizer
import config


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        return self.projection(x)


class ImageEncoder(nn.Module):
    """SwinV2 image encoder with global + local features."""
    
    def __init__(self, model_name=None, pretrained=True, embed_dim=None, image_size=None):
        super().__init__()
        model_name = model_name or config.VISION_MODEL
        embed_dim = embed_dim or config.VISION_EMBED_DIM
        image_size = image_size or config.VISION_IMAGE_SIZE
        
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='',
        )
        
        self.feature_dim = self.model.num_features
        self.local_proj = nn.Conv2d(self.feature_dim, embed_dim, kernel_size=1)
        self.global_proj = ProjectionHead(self.feature_dim, config.PROJECTION_HIDDEN, embed_dim)
    
    def forward(self, x):
        features = self.model.forward_features(x)
        global_feat = features.mean(dim=[2, 3])
        global_embed = self.global_proj(global_feat)
        global_embed = F.normalize(global_embed, dim=-1, eps=1e-5)
        
        local_embed = self.local_proj(features)
        B, C, H, W = local_embed.shape
        local_embed = local_embed.permute(0, 2, 3, 1).reshape(B, H * W, C)
        local_embed = F.normalize(local_embed, dim=-1, eps=1e-5)
        
        return global_embed, local_embed


class TextEncoder(nn.Module):
    """PubMedBERT text encoder."""
    
    def __init__(self, model_name=None, embed_dim=None):
        super().__init__()
        model_name = model_name or config.TEXT_MODEL
        embed_dim = embed_dim or config.TEXT_EMBED_DIM
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        self.projection = ProjectionHead(
            self.model.config.hidden_size,
            config.PROJECTION_HIDDEN,
            embed_dim
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        cls_embed = outputs.last_hidden_state[:, 0, :]
        text_embed = self.projection(cls_embed)
        text_embed = F.normalize(text_embed, dim=-1, eps=1e-5)
        
        token_features = outputs.last_hidden_state
        return text_embed, token_features
    
    def tokenize(self, captions, max_length=None):
        max_length = max_length or config.TEXT_MAX_LENGTH
        return self.tokenizer(
            captions,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )


class MedicalCLIP(nn.Module):
    """Medical Image-Text Retrieval Model."""
    
    def __init__(self, pretrained_vision=True, image_size=None):
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained=pretrained_vision, image_size=image_size)
        self.text_encoder = TextEncoder()
        
        # Initialize temperature to CLIP's default (0.07) - much more stable
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.embed_dim = config.PROJECTION_DIM
    
    def forward(self, images, input_ids, attention_mask):
        image_embed, local_image = self.image_encoder(images)
        text_embed, local_text = self.text_encoder(input_ids, attention_mask)
        return image_embed, text_embed, local_image, local_text
    
    def get_logits(self, image_embed, text_embed):
        # Clamp the logit_scale parameter to prevent exp() overflow
        # np.log(100.0) ≈ 4.605
        self.logit_scale.data.clamp_(max=np.log(100.0))
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_embed @ text_embed.T
        return logits
