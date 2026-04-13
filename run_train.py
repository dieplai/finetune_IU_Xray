"""Training script - Both encoders frozen, deep projection head, batch=32, 100 epochs."""
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

sys.path.insert(0, '/root/IU_xray')
os.chdir('/root/IU_xray')

from dataset import get_dataloaders
from model import MedicalCLIP
from loss import CombinedLoss
from eval import compute_retrieval_metrics

device = torch.device('cuda')
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

print("Loading data...", flush=True)
train_loader, val_loader, test_loader, _, _, _ = get_dataloaders(
    batch_size=32, num_workers=0, image_size=384
)

print("Building model...", flush=True)
model = MedicalCLIP(pretrained_vision=True, image_size=384).to(device)

# Freeze BOTH encoders
for param in model.image_encoder.parameters():
    param.requires_grad = False
for param in model.text_encoder.model.parameters():
    param.requires_grad = False

# Replace projection heads with deeper ones - output 768 to match both encoders
class DeepProjectionHead(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=2048, output_dim=768):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        return self.projection(x)

# Replace both projection heads
model.text_encoder.projection = DeepProjectionHead().to(device)
model.image_encoder.global_proj = DeepProjectionHead().to(device)

trainable = [p for p in model.parameters() if p.requires_grad]
n_trainable = sum(p.numel() for p in trainable) / 1e6
print(f"Trainable params: {n_trainable:.2f}M (both encoders frozen, deep projection heads)", flush=True)

loss_fn = CombinedLoss()
optimizer = AdamW(trainable, lr=3e-4, weight_decay=0.0)

total_steps = len(train_loader) * 100
warmup_steps = int(total_steps * 0.05)
warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
cosine_sched = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])

best_r1 = 0.0
history = []

for epoch in range(100):
    model.train()
    epoch_loss = 0.0
    n_batches = 0
    t0 = time.time()
    
    for step, batch in enumerate(train_loader):
        images = batch['image'].to(device, non_blocking=True)
        captions = batch['caption']
        disease_vecs = batch['disease_vec'].to(device, non_blocking=True)
        tokenized = model.text_encoder.tokenize(captions)
        input_ids = tokenized['input_ids'].to(device, non_blocking=True)
        attention_mask = tokenized['attention_mask'].to(device, non_blocking=True)
        
        with torch.no_grad():
            ie_raw, li_raw = model.image_encoder.model.forward_features(images)
            ie_raw = ie_raw.mean(dim=[2, 3])  # Global pool
            li = model.image_encoder.local_proj(model.image_encoder.model.forward_features(images))
            B, C, H, W = li.shape
            li = li.permute(0, 2, 3, 1).reshape(B, H * W, C)
            li = F.normalize(li, dim=-1)
            
            te_out = model.text_encoder.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            cls_embed = te_out.last_hidden_state[:, 0, :]
            token_features = te_out.last_hidden_state
        
        # Apply projection heads
        ie = model.image_encoder.global_proj(ie_raw)
        ie = F.normalize(ie, dim=-1)
        te = model.text_encoder.projection(cls_embed)
        te = F.normalize(te, dim=-1)
        
        logits = model.get_logits(ie, te)
        loss, lg, ll = loss_fn(logits, disease_vecs, disease_vecs, li, token_features, attention_mask,
                               epoch=epoch, total_epochs=100)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        model.logit_scale.data.clamp_(min=0.0, max=4.6052)
        optimizer.zero_grad()
        scheduler.step()
        
        epoch_loss += loss.item()
        n_batches += 1
        
        if (step + 1) % 20 == 0:
            lr = optimizer.param_groups[0]['lr']
            temp = model.logit_scale.exp().item()
            print(f"  Epoch {epoch} [{step+1}/{len(train_loader)}] Loss: {loss.item():.4f} "
                  f"Temp: {temp:.3f} LR: {lr:.6f}", flush=True)
    
    elapsed = time.time() - t0
    avg_loss = epoch_loss / n_batches
    print(f"  Epoch {epoch} done in {elapsed:.1f}s, avg loss: {avg_loss:.4f}", flush=True)
    
    # Validate
    model.eval()
    all_ie, all_te = [], []
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            captions = batch['caption']
            tokenized = model.text_encoder.tokenize(captions)
            input_ids = tokenized['input_ids'].to(device)
            attention_mask = tokenized['attention_mask'].to(device)
            
            ie_raw = model.image_encoder.model.forward_features(images)
            ie_raw = ie_raw.mean(dim=[2, 3])
            ie = model.image_encoder.global_proj(ie_raw)
            ie = F.normalize(ie, dim=-1)
            
            te_out = model.text_encoder.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            cls_embed = te_out.last_hidden_state[:, 0, :]
            te = model.text_encoder.projection(cls_embed)
            te = F.normalize(te, dim=-1)
            
            all_ie.append(ie.cpu())
            all_te.append(te.cpu())
    all_ie = torch.cat(all_ie, dim=0)
    all_te = torch.cat(all_te, dim=0)
    metrics = compute_retrieval_metrics(all_ie, all_te)
    
    r1_mean = metrics['mean_R']
    print(f"  Val: i2t R@1={metrics['i2t_R@1']:.1f}% R@5={metrics['i2t_R@5']:.1f}% R@10={metrics['i2t_R@10']:.1f}%", flush=True)
    print(f"       t2i R@1={metrics['t2i_R@1']:.1f}% R@5={metrics['t2i_R@5']:.1f}% R@10={metrics['t2i_R@10']:.1f}%", flush=True)
    print(f"       Mean R={r1_mean:.1f}%", flush=True)
    
    if r1_mean > best_r1:
        best_r1 = r1_mean
        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'metrics': metrics,
        }, '/root/IU_xray/outputs/best_model.pth')
        print(f"  ** New best! Mean R: {best_r1:.1f}% **", flush=True)
    
    history.append({'epoch': epoch, 'loss': avg_loss, 'metrics': {k: float(v) for k, v in metrics.items()}})
    
    if (epoch + 1) % 20 == 0:
        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'metrics': metrics,
        }, f'/root/IU_xray/outputs/checkpoint_epoch_{epoch}.pth')
        print(f"  Saved checkpoint_epoch_{epoch}.pth", flush=True)

# Save history
with open('/root/IU_xray/outputs/training_history.json', 'w') as f:
    json.dump(history, f, indent=2)

# Final test
print("\n=== FINAL TEST ===", flush=True)
best_ckpt = torch.load('/root/IU_xray/outputs/best_model.pth', map_location=device, weights_only=False)
model.load_state_dict(best_ckpt['model_state_dict'])
model.eval()

all_ie, all_te = [], []
with torch.no_grad():
    for batch in test_loader:
        images = batch['image'].to(device)
        captions = batch['caption']
        tokenized = model.text_encoder.tokenize(captions)
        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)
        
        ie_raw = model.image_encoder.model.forward_features(images)
        ie_raw = ie_raw.mean(dim=[2, 3])
        ie = model.image_encoder.global_proj(ie_raw)
        ie = F.normalize(ie, dim=-1)
        
        te_out = model.text_encoder.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        cls_embed = te_out.last_hidden_state[:, 0, :]
        te = model.text_encoder.projection(cls_embed)
        te = F.normalize(te, dim=-1)
        
        all_ie.append(ie.cpu())
        all_te.append(te.cpu())
all_ie = torch.cat(all_ie, dim=0)
all_te = torch.cat(all_te, dim=0)
test_metrics = compute_retrieval_metrics(all_ie, all_te)

print(f"Test: i2t R@1={test_metrics['i2t_R@1']:.1f}% R@5={test_metrics['i2t_R@5']:.1f}% R@10={test_metrics['i2t_R@10']:.1f}%", flush=True)
print(f"      t2i R@1={test_metrics['t2i_R@1']:.1f}% R@5={test_metrics['t2i_R@5']:.1f}% R@10={test_metrics['t2i_R@10']:.1f}%", flush=True)
print(f"      Mean R={test_metrics['mean_R']:.1f}%", flush=True)

with open('/root/IU_xray/outputs/test_results.json', 'w') as f:
    json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)

print("\n=== TRAINING COMPLETE ===", flush=True)
