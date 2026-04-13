import torch
import config
from dataset import get_dataloaders
from model import MedicalCLIP
from loss import CombinedLoss
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

def debug():
    train_loader, _, _, _, _, _ = get_dataloaders(batch_size=16, num_workers=0)
    model = MedicalCLIP(image_size=config.VISION_IMAGE_SIZE).cuda()
    loss_fn = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)
    
    total_steps = len(train_loader) * 50
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        images = batch['image'].cuda()
        captions = batch['caption']
        disease_vecs = batch['disease_vec'].cuda()
        
        tokenized = model.text_encoder.tokenize(captions)
        input_ids = tokenized['input_ids'].cuda()
        attention_mask = tokenized['attention_mask'].cuda()
        
        image_embed, text_embed, local_image, local_text = model(images, input_ids, attention_mask)
        logits = model.get_logits(image_embed, text_embed)
        loss, _, _ = loss_fn(logits, disease_vecs, disease_vecs, local_image, local_text, attention_mask)
        
        loss = loss / 2.0
        
        loss.backward()
        
        if (i + 1) % 2 == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            print(f"Step {i} | Loss: {loss.item():.4f} | Grad norm: {grad_norm.item():.4f} | LR: {optimizer.param_groups[0]['lr']}")
            optimizer.step()
            scheduler.step()
        else:
            print(f"Step {i} | Loss: {loss.item():.4f} | Accumulating...")

        if i >= 5:
            break

debug()
