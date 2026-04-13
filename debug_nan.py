import torch
import config
from dataset import get_dataloaders
from model import MedicalCLIP
from loss import CombinedLoss

def debug():
    train_loader, _, _, _, _, _ = get_dataloaders(batch_size=32, num_workers=0)
    model = MedicalCLIP(image_size=config.VISION_IMAGE_SIZE).cuda()
    loss_fn = CombinedLoss(
        alpha=config.DISEASE_ALPHA,
        disease_temperature=config.DISEASE_TEMPERATURE,
        local_weight=config.LOCAL_WEIGHT,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)
    
    torch.autograd.set_detect_anomaly(True)
    
    for step, batch in enumerate(train_loader):
        images = batch['image'].cuda()
        captions = batch['caption']
        disease_vecs = batch['disease_vec'].cuda()
        
        tokenized = model.text_encoder.tokenize(captions)
        input_ids = tokenized['input_ids'].cuda()
        attention_mask = tokenized['attention_mask'].cuda()
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            image_embed, text_embed, local_image, local_text = model(images, input_ids, attention_mask)
            logits = model.get_logits(image_embed, text_embed)
            
            loss, loss_global, loss_local = loss_fn(
                logits, disease_vecs, disease_vecs,
                local_image, local_text, attention_mask,
                epoch=0, total_epochs=50
            )
            
        print(f"Step {step} - Loss: {loss.item():.4f}")
        if torch.isnan(loss):
            print("NaN detected in loss!")
            print("image_embed NaN:", torch.isnan(image_embed).any().item())
            print("text_embed NaN:", torch.isnan(text_embed).any().item())
            print("logits NaN:", torch.isnan(logits).any().item())
            break
            
        loss.backward()
        
        # Check gradients
        has_nan_grad = False
        for name, p in model.named_parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                print(f"NaN grad in {name}")
                has_nan_grad = True
                break
        
        if has_nan_grad:
            break
            
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step > 10:
            break

debug()
