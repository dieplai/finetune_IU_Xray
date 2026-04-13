"""
Evaluation metrics for image-text retrieval.
Computes R@1, R@5, R@10 for both image->text and text->image.
"""
import torch
import numpy as np


def compute_retrieval_metrics(image_embeds, text_embeds, k_values=[1, 5, 10]):
    """
    Compute retrieval metrics.
    
    Args:
        image_embeds: (N, embed_dim) - image embeddings
        text_embeds: (N, embed_dim) - text embeddings
        k_values: list of K for R@K
    
    Returns:
        dict with R@K for image->text and text->image
    """
    N = image_embeds.shape[0]
    
    # Compute similarity matrix
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    sim_matrix = image_embeds @ text_embeds.T  # (N, N)
    
    results = {}
    
    # Image -> Text retrieval
    for k in k_values:
        ranks = []
        for i in range(N):
            # Sort by similarity (descending)
            sims = sim_matrix[i]
            ranked_indices = torch.argsort(sims, descending=True)
            # Find rank of correct match (index i)
            rank = (ranked_indices == i).nonzero(as_tuple=True)[0][0].item() + 1
            ranks.append(rank)
        
        ranks = np.array(ranks)
        results[f'i2t_R@{k}'] = (ranks <= k).mean() * 100
    
    # Text -> Image retrieval
    sim_matrix_t2i = sim_matrix.T  # (N, N)
    for k in k_values:
        ranks = []
        for i in range(N):
            sims = sim_matrix_t2i[i]
            ranked_indices = torch.argsort(sims, descending=True)
            rank = (ranked_indices == i).nonzero(as_tuple=True)[0][0].item() + 1
            ranks.append(rank)
        
        ranks = np.array(ranks)
        results[f't2i_R@{k}'] = (ranks <= k).mean() * 100
    
    # Mean recall
    results['mean_R'] = np.mean([v for k, v in results.items() if 'R@' in k])
    
    return results


def evaluate_model(model, dataloader, device='cuda'):
    """
    Evaluate model on a dataloader.
    Returns all image and text embeddings for metric computation.
    """
    model.eval()
    
    all_image_embeds = []
    all_text_embeds = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            captions = batch['caption']
            
            # Tokenize
            tokenized = model.text_encoder.tokenize(captions)
            input_ids = tokenized['input_ids'].to(device)
            attention_mask = tokenized['attention_mask'].to(device)
            
            # Forward
            image_embed, text_embed, _, _ = model(images, input_ids, attention_mask)
            
            all_image_embeds.append(image_embed.cpu())
            all_text_embeds.append(text_embed.cpu())
    
    all_image_embeds = torch.cat(all_image_embeds, dim=0)
    all_text_embeds = torch.cat(all_text_embeds, dim=0)
    
    return all_image_embeds, all_text_embeds


def full_evaluation(model, dataloader, device='cuda'):
    """Full evaluation pipeline."""
    image_embeds, text_embeds = evaluate_model(model, dataloader, device)
    metrics = compute_retrieval_metrics(image_embeds, text_embeds)
    return metrics
