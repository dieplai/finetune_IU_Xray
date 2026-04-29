"""
BiomedCLIP (ViT-B/16 + PubMedBERT) fine-tuning for IU-Xray retrieval.
GPU   : RTX 3090 (24GB)
Target: R@1 > 40%

v4 strategy — asymmetric learning rate based on zero-shot diagnostic:
  Root cause of v3 failure: zero-shot BiomedCLIP on IU-Xray has signal≈0.003
  because BiomedCLIP text encoder was trained on PubMed figure captions
  ("Figure 3. Chest X-ray showing...") while IU-Xray uses clinical notes
  ("The heart is normal in size. The lungs are clear.") — completely different
  text distribution.

  Diagnosis findings:
  - Zero-shot R@1 = 0.78% (near random 0.28%)
  - image-text signal (pos-neg cosine) = 0.0034 ≈ zero
  - text-text discriminability = 0.19 (texts ARE distinguishable, alignment broken)
  - After 40ep @ LR=1e-5: R@1 = 5.33% (only 7× improvement, LR was too small)

  Fix — asymmetric LR:
  - LR_VISION = 5e-6 : very small, keep BiomedCLIP ViT visual features stable
  - LR_TEXT   = 5e-4 : 100× higher, aggressively adapt text encoder to IU-Xray
  - LR_PROJ   = 5e-4 : logit_scale can adjust quickly

  Why this works:
  - BiomedCLIP ViT already encodes medical image features well → preserve
  - Text encoder must learn IU-Xray clinical report style → force rapid adaptation
  - With LR=5e-4 and early stopping, expected R@1 = 20-35%
"""
import os

# ── Data paths (Local execution) ───────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CSV_PATH   = os.path.join(DATA_DIR, "v8_clean.csv")
IMG_DIR    = os.path.join(DATA_DIR, "images_384")   # pre-resized 384x384
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Images are already resized to 384x384 -> skip T.Resize in DataLoader
# This saves ~25-40% training time (no LANCZOS resize per batch)
IMAGES_PRE_RESIZED = True

# Vision: SwinV2-Base at 384x384 (native resolution)
VISION_MODEL      = "microsoft/swinv2-base-patch4-window12to16-192to256-22kto1k-ft"
VISION_EMBED_DIM  = 1024  # SwinV2-Base output dimension
VISION_IMAGE_SIZE = 384   # SwinV2-Base-384 native input resolution

# Text: Bio_ClinicalBERT
TEXT_MODEL      = "emilyalsentzer/Bio_ClinicalBERT"
TEXT_MAX_LENGTH = 128
TEXT_EMBED_DIM  = 768

# Shared projection dimension
PROJECTION_DIM = 512

# Caption prefix: False -> frontal+lateral share same caption (correct for cross-view loss)
USE_CAPTION_PREFIX = False

# ImageNet normalization (SwinV2 pretrained on ImageNet-22K)
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD  = (0.229, 0.224, 0.225)

# ── Fine-tuning mode ───────────────────────────────────────────────────────────
FREEZE_ENCODERS = False

# Training hyperparameters
BATCH_SIZE            = 32   # SwinV2-Base 384 needs more VRAM than ViT-B/16
GRADIENT_ACCUMULATION = 4    # Effective batch = 128 for strong contrastive signal
NUM_EPOCHS            = 30
WARMUP_RATIO          = 0.05
MAX_GRAD_NORM         = 1.0

# Asymmetric LR: SwinV2 (pretrained ImageNet) stable, ClinicalBERT needs adaptation
LR_VISION    = 1e-5   # SwinV2 fine-tuning rate
LR_TEXT      = 5e-5   # Bio_ClinicalBERT adaptation (medical domain already close)
LR_PROJ      = 1e-4   # Projection head trains fastest
WEIGHT_DECAY = 0.01

# Cross-View Contrastive Loss (frontal <-> lateral same-patient InfoNCE)
CROSS_VIEW_WEIGHT = 0.3

# Local alignment loss
LOCAL_WEIGHT = 0.0

# Clustering-Guided Loss — activate at epoch 3 (after basic alignment warm-up)
# Earlier than before (was 15) to reduce false negatives throughout training
DISEASE_EPOCH       = 3
DISEASE_ALPHA       = 0.2
DISEASE_TEMPERATURE = 0.5

DISEASE_KEYWORDS = {
    'cardiac':          ['cardiomegaly', 'enlarged heart', 'heart size', 'cardiac silhouette',
                         'heart failure', 'vascular congestion', 'aortic'],
    'pleural_effusion': ['pleural effusion', 'effusion', 'blunting', 'costophrenic'],
    'pneumothorax':     ['pneumothorax', 'pneumothoraces'],
    'infection':        ['consolidation', 'pneumonia', 'infiltrate', 'opacity',
                         'airspace disease', 'focal consolidation'],
    'edema':            ['edema', 'pulmonary edema', 'interstitial edema', 'fluid overload'],
    'atelectasis':      ['atelectasis', 'collapse', 'subsegmental'],
    'copd':             ['emphysema', 'hyperinflated', 'copd', 'flattened diaphragm'],
    'nodule_mass':      ['nodule', 'mass', 'lesion', 'tumor'],
    'calcification':    ['calcification', 'calcified', 'atherosclerotic', 'granuloma'],
    'bone':             ['fracture', 'scoliosis', 'degenerative', 'osteophyte', 'vertebral'],
    'normal':           ['clear', 'normal', 'no acute', 'unremarkable'],
}

EVAL_EVERY  = 1
SAVE_EVERY  = 10
SEED        = 42
NUM_WORKERS = 4
PIN_MEMORY  = True
