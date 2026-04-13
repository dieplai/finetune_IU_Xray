"""
Main training script for Medical Image-Text Retrieval on IU-Xray.
Optimized for RTX 3090 (24GB) -> later A100 (80GB).
Curriculum learning: InfoNCE first, then soft contrastive.
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast, GradScaler

import config
from model import MedicalCLIP
from dataset import get_dataloaders
from loss import CombinedLoss
from eval import compute_retrieval_metrics


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, scaler, device, epoch, use_amp=True, grad_clip=1.0, total_epochs=50):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_loss_global = 0.0
    total_loss_local = 0.0
    n_batches = 0
    
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        
        images = batch['image'].to(device, non_blocking=True)
        captions = batch['caption']
        disease_vecs = batch['disease_vec'].to(device, non_blocking=True)
        
        tokenized = model.text_encoder.tokenize(captions)
        input_ids = tokenized['input_ids'].to(device, non_blocking=True)
        attention_mask = tokenized['attention_mask'].to(device, non_blocking=True)
        
        if use_amp:
            with autocast('cuda', enabled=True, dtype=torch.bfloat16):
                image_embed, text_embed, local_image, local_text = model(
                    images, input_ids, attention_mask
                )
                logits = model.get_logits(image_embed, text_embed)
                loss, loss_global, loss_local = loss_fn(
                    logits, disease_vecs, disease_vecs,
                    local_image, local_text, attention_mask,
                    epoch=epoch, total_epochs=total_epochs
                )
        else:
            image_embed, text_embed, local_image, local_text = model(
                images, input_ids, attention_mask
            )
            logits = model.get_logits(image_embed, text_embed)
            loss, loss_global, loss_local = loss_fn(
                logits, disease_vecs, disease_vecs,
                local_image, local_text, attention_mask,
                epoch=epoch, total_epochs=total_epochs
            )
        
        loss_val = loss.item()
        loss = loss / config.GRADIENT_ACCUMULATION
        
        if torch.isnan(loss):
            print(f"Warning: NaN loss at epoch {epoch} step {step+1}. Skipping.")
            optimizer.zero_grad()
            continue
        if use_amp:
            loss.backward()

            if (step + 1) % config.GRADIENT_ACCUMULATION == 0 or (step + 1) == len(train_loader):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"Warning: Invalid gradients at epoch {epoch} step {step+1}. Skipping optimizer step.")
                    optimizer.zero_grad()
                else:
                    optimizer.step()
                    optimizer.zero_grad()
        else:
            loss.backward()

            if (step + 1) % config.GRADIENT_ACCUMULATION == 0 or (step + 1) == len(train_loader):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"Warning: Invalid gradients at epoch {epoch} step {step+1}. Skipping optimizer step.")
                    optimizer.zero_grad()
                else:
                    optimizer.step()
                    optimizer.zero_grad()

        if scheduler is not None and ((step + 1) % config.GRADIENT_ACCUMULATION == 0 or (step + 1) == len(train_loader)):
            scheduler.step()
        
        total_loss += loss.item()
        total_loss_global += loss_global.item()
        total_loss_local += loss_local.item()
        n_batches += 1
        
        lr = optimizer.param_groups[0]['lr']
        temp = model.logit_scale.exp().item()
        print(f"  Epoch {epoch} [{step+1}/{len(train_loader)}] "
              f"Loss: {loss.item():.4f} "
              f"Global: {loss_global.item():.4f} "
              f"Local: {loss_local.item():.4f} "
              f"Temp: {temp:.3f} LR: {lr:.6f}")
    
    avg_loss = total_loss / n_batches
    avg_global = total_loss_global / n_batches
    avg_local = total_loss_local / n_batches
    
    return avg_loss, avg_global, avg_local


@torch.no_grad()
def validate(model, val_loader, device):
    """Validate and compute retrieval metrics."""
    model.eval()
    all_image_embeds = []
    all_text_embeds = []
    
    for batch in val_loader:
        images = batch['image'].to(device, non_blocking=True)
        captions = batch['caption']
        tokenized = model.text_encoder.tokenize(captions)
        input_ids = tokenized['input_ids'].to(device, non_blocking=True)
        attention_mask = tokenized['attention_mask'].to(device, non_blocking=True)
        image_embed, text_embed, _, _ = model(images, input_ids, attention_mask)
        all_image_embeds.append(image_embed.cpu())
        all_text_embeds.append(text_embed.cpu())
    
    all_image_embeds = torch.cat(all_image_embeds, dim=0)
    all_text_embeds = torch.cat(all_text_embeds, dim=0)
    return compute_retrieval_metrics(all_image_embeds, all_text_embeds)


def save_checkpoint(model, optimizer, epoch, metrics, save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'logit_scale': model.logit_scale.item(),
    }
    torch.save(checkpoint, save_path)
    print(f"  Saved checkpoint: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=config.LR)
    parser.add_argument('--image_size', type=int, default=config.VISION_IMAGE_SIZE)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--grad_clip', type=float, default=1.0)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    set_seed(config.SEED)
    
    print("\n" + "="*60)
    print("Loading data...")
    print("="*60)
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=config.NUM_WORKERS,
        image_size=args.image_size,
        seed=config.SEED,
    )
    
    print("\n" + "="*60)
    print("Building model...")
    print("="*60)
    model = MedicalCLIP(pretrained_vision=True, image_size=args.image_size)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {n_params/1e6:.1f}M")
    print(f"Trainable parameters: {n_trainable/1e6:.1f}M")
    model = model.to(device)
    
    loss_fn = CombinedLoss(
        alpha=config.DISEASE_ALPHA,
        disease_temperature=config.DISEASE_TEMPERATURE,
        local_weight=config.LOCAL_WEIGHT,
    )
    print(f"Using Curriculum Learning:")
    print(f"  Phase 1 (epochs 0-9): Pure InfoNCE")
    print(f"  Phase 2 (epochs 10+): Soft Contrastive + Local alignment")
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    
    use_amp = not args.no_amp
    scaler = GradScaler('cuda', enabled=use_amp)
    
    start_epoch = 0
    best_r1 = 0.0
    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_r1 = checkpoint['metrics'].get('mean_R', 0)
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    history = []
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        avg_loss, avg_global, avg_local = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, scaler, device, epoch,
            use_amp=use_amp, grad_clip=args.grad_clip, total_epochs=args.epochs
        )
        
        if (epoch + 1) % config.EVAL_EVERY == 0:
            metrics = validate(model, val_loader, device)
            elapsed = time.time() - epoch_start
            r1_mean = metrics['mean_R']
            
            print(f"\n  Epoch {epoch} | Time: {elapsed:.1f}s")
            print(f"  Loss: {avg_loss:.4f} (Global: {avg_global:.4f}, Local: {avg_local:.4f})")
            print(f"  i2t R@1: {metrics['i2t_R@1']:.1f}% | R@5: {metrics['i2t_R@5']:.1f}% | R@10: {metrics['i2t_R@10']:.1f}%")
            print(f"  t2i R@1: {metrics['t2i_R@1']:.1f}% | R@5: {metrics['t2i_R@5']:.1f}% | R@10: {metrics['t2i_R@10']:.1f}%")
            print(f"  Mean R: {r1_mean:.1f}%")
            
            if r1_mean > best_r1:
                best_r1 = r1_mean
                save_checkpoint(model, optimizer, epoch, metrics,
                               os.path.join(config.OUTPUT_DIR, 'best_model.pth'))
            
            if (epoch + 1) % config.SAVE_EVERY == 0:
                save_checkpoint(model, optimizer, epoch, metrics,
                               os.path.join(config.OUTPUT_DIR, f'checkpoint_epoch_{epoch}.pth'))
            
            history.append({'epoch': epoch, 'loss': avg_loss, 'metrics': metrics})
        else:
            elapsed = time.time() - epoch_start
            print(f"  Epoch {epoch} | Time: {elapsed:.1f}s | Loss: {avg_loss:.4f}")
    
    save_checkpoint(model, optimizer, args.epochs - 1, metrics,
                   os.path.join(config.OUTPUT_DIR, 'final_model.pth'))
    
    with open(os.path.join(config.OUTPUT_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*60)
    print("Final Test Evaluation")
    print("="*60)
    best_ckpt = torch.load(os.path.join(config.OUTPUT_DIR, 'best_model.pth'), map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])
    test_metrics = validate(model, test_loader, device)
    
    print(f"\n  Test Results:")
    print(f"  i2t R@1: {test_metrics['i2t_R@1']:.1f}% | R@5: {test_metrics['i2t_R@5']:.1f}% | R@10: {test_metrics['i2t_R@10']:.1f}%")
    print(f"  t2i R@1: {test_metrics['t2i_R@1']:.1f}% | R@5: {test_metrics['t2i_R@5']:.1f}% | R@10: {test_metrics['t2i_R@10']:.1f}%")
    print(f"  Mean R: {test_metrics['mean_R']:.1f}%")
    
    with open(os.path.join(config.OUTPUT_DIR, 'test_results.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    print(f"\n  Best Val Mean R: {best_r1:.1f}%")
    print(f"  Checkpoint saved to: {config.OUTPUT_DIR}")


if __name__ == '__main__':
    main()
