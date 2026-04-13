import torch
import timm
from model import ImageEncoder

def test_model(model_name):
    print(f"Testing {model_name}...")
    try:
        if 'swin' in model_name and 'cr' not in model_name:
            model = ImageEncoder(model_name=model_name, pretrained=True, image_size=224).cuda()
        else:
            model = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='').cuda()
        
        x = torch.randn(4, 3, 224, 224).cuda()
        f = model.forward_features(x) if hasattr(model, 'forward_features') else model(x)
        loss = f.sum()
        loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        print(f"  Success! grad_norm: {grad_norm.item():.4f}")
    except Exception as e:
        print(f"  Failed: {e}")

test_model("resnet50")
test_model("convnext_tiny")
test_model("swin_base_patch4_window7_224")
