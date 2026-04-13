"""
Inference script for trained model.
Load checkpoint and evaluate on test set or run custom retrieval.
"""
import os
import json
import argparse
import torch
from PIL import Image
import numpy as np

import config
from model import MedicalCLIP
from dataset import get_dataloaders, get_val_transform
from eval import compute_retrieval_metrics, evaluate_model


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint."""
    model = MedicalCLIP(pretrained_vision=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Val metrics: {checkpoint.get('metrics', {})}")
    return model


def evaluate_test_set(model, test_loader, device='cuda'):
    """Evaluate on test set."""
    image_embeds, text_embeds = evaluate_model(model, test_loader, device)
    metrics = compute_retrieval_metrics(image_embeds, text_embeds)
    
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    print(f"  i2t R@1:  {metrics['i2t_R@1']:.2f}%")
    print(f"  i2t R@5:  {metrics['i2t_R@5']:.2f}%")
    print(f"  i2t R@10: {metrics['i2t_R@10']:.2f}%")
    print(f"  t2i R@1:  {metrics['t2i_R@1']:.2f}%")
    print(f"  t2i R@5:  {metrics['t2i_R@5']:.2f}%")
    print(f"  t2i R@10: {metrics['t2i_R@10']:.2f}%")
    print(f"  Mean R:   {metrics['mean_R']:.2f}%")
    
    return metrics


def retrieve_text_for_image(model, image_path, dataloader, device='cuda', top_k=5):
    """Given a query image, retrieve most relevant captions."""
    model.eval()
    
    # Encode query image
    transform = get_val_transform(config.VISION_IMAGE_SIZE)
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        query_embed, _, _, _ = model(image_tensor, None, None)
    
    # Encode all texts in dataloader
    all_text_embeds = []
    all_captions = []
    all_image_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            captions = batch['caption']
            tokenized = model.text_encoder.tokenize(captions)
            input_ids = tokenized['input_ids'].to(device)
            attention_mask = tokenized['attention_mask'].to(device)
            
            _, text_embed, _, _ = model(torch.zeros(1, 3, 512, 512, device=device),
                                         input_ids, attention_mask)
            all_text_embeds.append(text_embed.cpu())
            all_captions.extend(captions)
            all_image_ids.extend(batch['image_id'])
    
    all_text_embeds = torch.cat(all_text_embeds, dim=0)
    all_text_embeds = all_text_embeds / all_text_embeds.norm(dim=-1, keepdim=True).clamp(min=1e-5)
    query_embed = query_embed / query_embed.norm(dim=-1, keepdim=True).clamp(min=1e-5)
    
    # Compute similarity
    sims = (query_embed @ all_text_embeds.T).squeeze(0)
    top_indices = torch.argsort(sims, descending=True)[:top_k]
    
    print(f"\nQuery image: {image_path}")
    print(f"\nTop {top_k} retrieved captions:")
    for rank, idx in enumerate(top_indices):
        print(f"  {rank+1}. [{sims[idx]:.4f}] {all_captions[idx]}")
        print(f"      Image: {all_image_ids[idx]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--mode', type=str, default='evaluate',
                       choices=['evaluate', 'retrieve'])
    parser.add_argument('--image_path', type=str, default=None)
    parser.add_argument('--top_k', type=int, default=5)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_model(args.checkpoint, device)
    
    _, _, test_loader, _, _, _ = get_dataloaders(
        batch_size=32, num_workers=2, image_size=config.VISION_IMAGE_SIZE
    )
    
    if args.mode == 'evaluate':
        evaluate_test_set(model, test_loader, device)
    elif args.mode == 'retrieve':
        if args.image_path is None:
            print("Error: --image_path required for retrieve mode")
            return
        retrieve_text_for_image(model, args.image_path, test_loader, device, args.top_k)


if __name__ == '__main__':
    main()
