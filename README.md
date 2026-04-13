# Medical Image-Text Retrieval on IU-Xray Dataset

## Mục tiêu
Xây dựng mô hình retrieval ảnh X-ray ↔ báo cáo y tế với **R@1 cao nhất có thể**.
- Nếu ảnh A và báo cáo B cùng cụm bệnh → không coi là negative của nhau
- Dataset: [IU Chest X-Rays Cleaned](https://www.kaggle.com/datasets/masrursabab/iu-chest-x-rays-cleaned/)

## Cấu trúc thư mục
```
IU_xray/
├── README.md              # File này
├── config.py              # Cấu hình hyperparameters
├── dataset.py             # Dataset class với disease cluster
├── disease_cluster.py     # Disease keyword extraction
├── model.py               # SwinV2 + PubMedBERT architecture
├── loss.py                # InfoNCE + Soft Contrastive + Local alignment
├── eval.py                # R@1, R@5, R@10 metrics
├── train.py               # Training script (dùng run_train.py thay thế)
├── inference.py           # Inference script
├── run_train.py           # Script training chính (mới nhất)
└── outputs/
    ├── best_model.pth     # Best checkpoint
    ├── checkpoint_epoch_*.pth
    ├── training_history.json
    └── test_results.json
```

## Cài đặt môi trường

### 1. Cài Python dependencies
```bash
apt-get update && apt-get install -y python3 python3-pip
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip3 install transformers timm pandas numpy matplotlib peft accelerate kagglehub
```

### 2. Tải dataset từ Kaggle
```python
import kagglehub
path = kagglehub.dataset_download("masrursabab/iu-chest-x-rays-cleaned")
print("Path:", path)
```

Dataset sẽ được tải về: `~/.cache/kagglehub/datasets/masrursabab/iu-chest-x-rays-cleaned/versions/1/`

### 3. Cấu trúc dataset
```
cleaned_dataset.csv          # 7265 rows: image_id, projection, org_caption
resized_images/
├── 256/   (7265 files)     # 256x256 RGB PNG
├── 299/   (7265 files)     # 299x299 RGB PNG
├── 320/   (7265 files)     # 320x320 RGB PNG  ← DÙNG SIZE NÀY
└── 512/   (7265 files)     # 512x512 RGB PNG
```

### 4. Thống kê dataset
- **7,265 images** từ **3,844 bệnh nhân** (~2 ảnh/bệnh nhân: Frontal + Lateral)
- **3,030 unique captions** → 2,683 captions được chia sẻ bởi nhiều ảnh
- Caption trung bình: 37 từ
- **12 disease categories**: cardiac, pleural_effusion, pneumothorax, infection, edema, atelectasis, copd, nodule_mass, calcification, bone, normal, other

## Chạy training

### RTX 3090 (24GB VRAM)
```bash
cd /root/IU_xray
PYTHONUNBUFFERED=1 python3 run_train.py 2>&1 | tee training.log
```

### A100 (80GB VRAM) - Nâng cấp
Trên A100, có thể:
1. Tăng batch size lên 64-128
2. Unfreeze text encoder để fine-tune
3. Dùng SwinV2-Base hoặc Large thay vì Tiny

```bash
# Trên A100: sửa run_train.py
# - batch_size=64
# - Unfreeze text encoder
# - Đổi VISION_MODEL = "swinv2_cr_base_384"
```

## Kiến trúc model

```
┌─────────────────────────────────────────────┐
│              IMAGE ENCODER (FROZEN)          │
│  SwinV2-CR-Small (pretrained ImageNet-21K)  │
│  Input: 384x384, Output: 768-dim            │
│  ~49.7M params (không train)                │
└─────────────────────────────────────────────┘
                    ↓
        [Deep Projection Head]
        768 → 2048 → 2048 → 2048 → 768
                    ↓
┌─────────────────────────────────────────────┐
│              TEXT ENCODER (FROZEN)           │
│  PubMedBERT (pretrained)                     │
│  Input: 128 tokens, Output: 768-dim [CLS]   │
│  ~110M params (không train)                  │
└─────────────────────────────────────────────┘
                    ↓
        [Deep Projection Head]
        768 → 2048 → 2048 → 2048 → 768
                    ↓
┌─────────────────────────────────────────────┐
│         CONTRASTIVE LOSS (TRAIN)             │
│  - InfoNCE (phase 1: epochs 0-14)           │
│  - Soft Contrastive (phase 2: epochs 15+)   │
│  - Local alignment (phase 2)                │
│  Total trainable: ~23M params               │
└─────────────────────────────────────────────┘
```

## Các vấn đề đã gặp và giải pháp

### Vấn đề 1: NaN loss khi train toàn bộ model
**Mô tả:** Khi unfreeze cả vision encoder + text encoder, loss bị NaN ngay sau 1-2 optimizer steps.

**Debug chi tiết:**
- Gradient norms trước clip: ~323 (rất lớn)
- NaN xuất hiện SAU optimizer.step(), KHÔNG phải trong backward
- AdamW với weight_decay > 0 gây NaN ngay lập tức
- AdamW với weight_decay=0 vẫn NaN sau ~180 steps
- SGD cũng NaN sau ~500 steps
- NaN xuất hiện ở cả SwinV2 layers và BERT layers

**Các cách đã thử (đều thất bại):**
| Approach | Kết quả |
|----------|---------|
| AdamW + weight_decay=0 | NaN sau ~180 steps |
| AdamW + weight_decay=1e-4 | NaN ngay step 1 |
| AdamW + lr=1e-6 | NaN sau ~200 steps |
| AdamW + eps=1e-3 | NaN sau ~200 steps |
| SGD + momentum=0.9 | NaN sau ~500 steps |
| Gradient clip=0.1 | Vẫn NaN |
| Gradient clip=0.01 | Vẫn NaN |
| Gradient accumulation | NaN nặng hơn |
| AMP (mixed precision) | NaN nặng hơn |
| Freeze vision, train text | NaN sau ~180 steps |
| Freeze first 8 layers of BERT | NaN sau ~199 steps |
| LoRA adapters (r=16) | NaN sau ~8 steps |
| Different LR groups | Vẫn NaN |

**Giả thuyết:** Có thể do incompatibility giữa PyTorch 2.11+cu126 và CUDA driver 12.9 trên machine này. Hoặc do numerical instability trong SwinV2 shifted window attention khi kết hợp với contrastive loss.

**Giải pháp hiện tại:** Freeze cả 2 encoders, chỉ train projection heads (~23M params). Stable 100%, không NaN.

### Vấn đề 2: SwinV2 model name
- `swinv2_tiny_window_8_256` → sai (underscore)
- Đúng: `swinv2_tiny_window8_256`
- Model có pretrained: `swinv2_cr_small_224` (override img_size=384)

### Vấn đề 3: CUDA illegal instruction
- PyTorch 2.11+cu128 crash với driver 12.9
- Fix: downgrade xuống cu126

### Vấn đề 4: Temperature explosion
- logit_scale tăng không kiểm soát → exp(14.27) = 1.5M → NaN
- Fix: `model.logit_scale.data.clamp_(min=0.0, max=4.6052)` sau mỗi step

### Vấn đề 5: Working directory
- Background commands chạy từ /root thay vì /root/IU_xray
- Fix: dùng `os.chdir('/root/IU_xray')` trong script hoặc `cd /root/IU_xray && python3 ...`

## Hướng cải tiến trên A100 (80GB)

1. **Unfreeze text encoder** - Fine-tune PubMedBERT với LR nhỏ (1e-5)
2. **Tăng batch size** - 64-128 để contrastive learning hiệu quả hơn
3. **Dùng SwinV2-Base/Large** - `swinv2_cr_base_384` (87.9M) hoặc `swinv2_cr_large_384` (196.7M)
4. **Hard negative mining** - Chọn negatives khó hơn
5. **Multi-crop evaluation** - Test-time augmentation
6. **Ensemble** - Average multiple checkpoints
7. **Thử PyTorch nightly** - Có thể fix NaN issue

## Kết quả hiện tại
- Baseline (pretrained features, no training): R@1 ≈ 0.2%
- Với deep projection heads (23M params, đang train): chờ kết quả

## File log
- `training.log` - Full training output
- `outputs/training_history.json` - Metrics per epoch
- `outputs/test_results.json` - Final test results
