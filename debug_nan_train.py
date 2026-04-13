import torch
import config
from dataset import get_dataloaders
from model import MedicalCLIP
from loss import CombinedLoss

def debug():
    train_loader, _, _, _, _, _ = get_dataloaders(batch_size=32, num_workers=0)
    model = MedicalCLIP(image_size=config.VISION_IMAGE_SIZE).cuda()
    loss_fn = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    
    torch.autograd.set_detect_anomaly(True)
    
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        images = batch['image'].cuda()
        captions = batch['caption']
        disease_vecs = batch['disease_vec'].cuda()
        
        tokenized = model.text_encoder.tokenize(captions)
        input_ids = tokenized['input_ids'].cuda()
        attention_mask = tokenized['attention_mask'].cuda()
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            image_embed, text_embed, local_image, local_text = model(images, input_ids, attention_mask)
            logits = model.get_logits(image_embed, text_embed)
            loss, _, _ = loss_fn(logits, disease_vecs, disease_vecs, local_image, local_text, attention_mask)
        
        if torch.isnan(loss):
            print(f"Loss is NaN at step {i} BEFORE backward!")
            print("Captions:", captions)
            break
            
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        if torch.isnan(grad_norm):
            print(f"Grad norm is NaN at step {i}!")
            break
            
        optimizer.step()
        
        print(f"Step {i} | Loss: {loss.item():.4f} | Grad norm: {grad_norm.item():.4f}")

debug()
