"""
Configuration for IU-Xray Medical Image-Text Retrieval
RTX 3090 (24GB) -> later A100 (80GB)
"""
import os

DATA_DIR = "/root/.cache/kagglehub/datasets/masrursabab/iu-chest-x-rays-cleaned/versions/1"
CSV_PATH = os.path.join(DATA_DIR, "cleaned_dataset.csv")
# Use 320 folder (closest to 384, will resize to 384 in transform)
IMG_DIR = os.path.join(DATA_DIR, "resized_images", "256")
OUTPUT_DIR = "/root/IU_xray/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Vision: ConvNeXt Small
VISION_MODEL = "convnext_small.fb_in22k_ft_in1k"
VISION_EMBED_DIM = 768
VISION_IMAGE_SIZE = 256

# Text: PubMedBERT (110M params)
TEXT_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
TEXT_MAX_LENGTH = 128
TEXT_EMBED_DIM = 768

PROJECTION_DIM = 512
PROJECTION_HIDDEN = 1024

# Training - RTX 3090 (24GB)
BATCH_SIZE = 16
GRADIENT_ACCUMULATION = 2  # Effective batch = 32
NUM_EPOCHS = 50
WARMUP_RATIO = 0.1
LR = 2e-5
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
USE_AMP = False

# Contrastive Loss
TEMPERATURE = 0.07
LEARNABLE_TEMPERATURE = True
USE_SOFT_CONTRASTIVE = True
DISEASE_ALPHA = 0.3
DISEASE_TEMPERATURE = 0.5
USE_LOCAL_ALIGNMENT = True
LOCAL_WEIGHT = 0.5

# Disease Clustering
DISEASE_KEYWORDS = {
    'cardiac': ['cardiomegaly', 'enlarged heart', 'heart size', 'cardiac silhouette',
                'heart failure', 'vascular congestion', 'aortic'],
    'pleural_effusion': ['pleural effusion', 'effusion', 'blunting', 'costophrenic'],
    'pneumothorax': ['pneumothorax', 'pneumothoraces'],
    'infection': ['consolidation', 'pneumonia', 'infiltrate', 'opacity',
                  'airspace disease', 'focal consolidation'],
    'edema': ['edema', 'pulmonary edema', 'interstitial edema', 'fluid overload'],
    'atelectasis': ['atelectasis', 'collapse', 'subsegmental'],
    'copd': ['emphysema', 'hyperinflated', 'copd', 'flattened diaphragm'],
    'nodule_mass': ['nodule', 'mass', 'lesion', 'tumor'],
    'calcification': ['calcification', 'calcified', 'atherosclerotic', 'granuloma'],
    'bone': ['fracture', 'scoliosis', 'degenerative', 'osteophyte', 'vertebral'],
    'normal': ['clear', 'normal', 'no acute', 'unremarkable'],
}

EVAL_EVERY = 1
SAVE_EVERY = 5
SEED = 42
NUM_WORKERS = 4
PIN_MEMORY = True
