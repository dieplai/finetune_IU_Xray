import torch
import config
from dataset import get_dataloaders
from model import MedicalCLIP
from loss import CombinedLoss

def debug():
    train_loader, _, _, _, _, _ = get_dataloaders(batch_size=16, num_workers=0)
    model = MedicalCLIP(image_size=config.VISION_IMAGE_SIZE).cuda()
    loss_fn = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)
    
    torch.autograd.set_detect_anomaly(True)
    
    for i, batch in enumerate(train_loader):
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
        
        print(f"Step {i} | Loss: {loss.item()*2:.4f}")
        
        if (i + 1) % 2 == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isnan(grad_norm):
                print(f"Grad norm is NaN!")
                break
            optimizer.step()
            optimizer.zero_grad()
        
        if i >= 5:
            break

debug()
