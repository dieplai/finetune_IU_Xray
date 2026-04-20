"""
IU-Xray Dataset with patient-stratified split and cross-view support.

Key design:
  - PatientPairBatchSampler: ensures each batch has both frontal+lateral
    of the same patients -> B//2 guaranteed cross-view pairs per batch
  - USE_CAPTION_PREFIX = False: frontal+lateral share the same caption
    -> cross-view loss and text-image loss both pull same-patient pairs together
  - Clustering-Guided: disease_vec built from CheXpert hard labels (v8_clean.csv)
    instead of regex keyword matching -> more reliable cluster assignments
  - 'No Finding' excluded from cluster similarity to avoid over-smoothing
  - 'Enlarged Cardiomediastinum' merged into 'Cardiomegaly' (only 17 samples)
  - Images are already 384x384 (pre-resized) -> T.Resize skipped in transforms
"""
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
import torchvision.transforms as T
import config

# CheXpert pathology columns used as cluster labels (14 → 12 after merge/drop)
# Excludes 'No Finding' (too common, causes over-smoothing)
# Merges 'Enlarged Cardiomediastinum' into 'Cardiomegaly' (only 17 samples)
CHEXPERT_COLS = [
    'Cardiomegaly',       # merged: Enlarged Cardiomediastinum → here
    'Lung Lesion',
    'Lung Opacity',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices',
]
N_CLUSTERS = len(CHEXPERT_COLS)   # 12 meaningful pathology clusters


class IUXrayDataset(Dataset):
    def __init__(self, csv_path=None, img_dir=None, transform=None, split='train',
                 val_split=0.1, test_split=0.1, seed=42):
        self.csv_path = csv_path or config.CSV_PATH
        self.img_dir  = img_dir  or config.IMG_DIR

        df = pd.read_csv(self.csv_path)

        # patient_id: use existing column if present, else parse from image_id
        if 'patient_id' not in df.columns:
            df['patient_id'] = df['image_id'].apply(lambda x: x.split('_')[0])
        df['patient_id'] = df['patient_id'].astype(str)

        # ---- Projection type -------------------------------------------------
        if 'projection' in df.columns:
            df['proj_type'] = df['projection'].apply(
                lambda x: 'FRONTAL' if str(x).strip().lower().startswith('f') else 'LATERAL'
            )
        else:
            df['proj_type'] = df['image_id'].apply(
                lambda x: 'FRONTAL' if 'frontal' in x.lower() else 'LATERAL'
            )

        # ---- Caption ---------------------------------------------------------
        if config.USE_CAPTION_PREFIX:
            df['caption'] = df.apply(
                lambda r: f"{r['proj_type']}: {r['org_caption']}", axis=1
            )
        else:
            df['caption'] = df['org_caption']

        # ---- Clustering-Guided: CheXpert hard labels as cluster assignments ---
        # Merge 'Enlarged Cardiomediastinum' (17 samples) into 'Cardiomegaly'
        if 'Enlarged Cardiomediastinum' in df.columns and 'Cardiomegaly' in df.columns:
            df['Cardiomegaly'] = df[['Cardiomegaly', 'Enlarged Cardiomediastinum']].max(axis=1)

        # Build disease_vec_chexpert: 12-dim binary from CheXpert labels
        # Only includes meaningful pathology clusters (excludes 'No Finding')
        available_cols = [c for c in CHEXPERT_COLS if c in df.columns]
        df['disease_vec_np'] = df[available_cols].values.tolist()

        # ---- Patient-stratified split ----------------------------------------
        np.random.seed(seed)
        patient_ids = df['patient_id'].unique()
        np.random.shuffle(patient_ids)
        n      = len(patient_ids)
        n_test = int(n * test_split)
        n_val  = int(n * val_split)

        test_pats  = set(patient_ids[:n_test])
        val_pats   = set(patient_ids[n_test : n_test + n_val])
        train_pats = set(patient_ids[n_test + n_val :])

        split_map = {'train': train_pats, 'val': val_pats, 'test': test_pats}
        self.df = df[df['patient_id'].isin(split_map[split])].reset_index(drop=True)

        # ---- Store cluster label arrays --------------------------------------
        self.disease_vecs_np = np.array(
            self.df['disease_vec_np'].tolist(), dtype=np.float32
        )  # shape: (N, N_CLUSTERS)
        self.n_clusters      = self.disease_vecs_np.shape[1]
        self.cluster_cols    = available_cols

        # Stats: how many samples have at least 1 pathological finding
        n_patho = int((self.disease_vecs_np.sum(axis=1) > 0).sum())

        self.transform = transform or get_train_transform()
        self.split     = split

        n_unique = self.df['caption'].nunique()
        print(f"[{split}] {len(self.df)} images | "
              f"{self.df['patient_id'].nunique()} patients | "
              f"{n_unique} unique captions ({n_unique/len(self.df)*100:.1f}%) | "
              f"{n_patho} pathological ({n_patho/len(self.df)*100:.1f}%) | "
              f"{self.n_clusters} cluster dims")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.img_dir, row['image_id'])
        image    = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        caption = row['caption']

        # Clustering-Guided: 12-dim CheXpert hard labels as cluster vector
        disease_vec = torch.from_numpy(self.disease_vecs_np[idx])  # (N_CLUSTERS,)

        return {
            'image':       image,
            'caption':     caption,
            'image_id':    row['image_id'],
            'patient_id':  row['patient_id'],
            'projection':  row['proj_type'],
            'disease_vec': disease_vec,   # binary CheXpert labels
        }


class PatientPairBatchSampler(Sampler):
    """
    Custom batch sampler for cross-view contrastive learning.

    Guarantees each batch contains BOTH frontal AND lateral images of the
    same B//2 patients -> B//2 guaranteed cross-view pairs per batch.

    Only uses patients that have BOTH views. Single-view patients are excluded
    from training (they can't contribute cross-view signal anyway).

    Args:
        dataset   : IUXrayDataset (train split)
        batch_size: must be even; B//2 patients per batch
        shuffle   : whether to shuffle patient order each epoch
    """

    def __init__(self, dataset, batch_size, shuffle=True):
        super().__init__()
        assert batch_size % 2 == 0, "batch_size must be even for PatientPairBatchSampler"
        self.batch_size = batch_size
        self.shuffle    = shuffle

        # Build patient_id -> {FRONTAL: dataset_idx, LATERAL: dataset_idx}
        patient_views = {}
        for idx in range(len(dataset)):
            row  = dataset.df.iloc[idx]
            pid  = row['patient_id']
            proj = row['proj_type']
            if pid not in patient_views:
                patient_views[pid] = {}
            patient_views[pid][proj] = idx

        # Only keep patients that have BOTH frontal AND lateral
        self.pair_patients = sorted([
            pid for pid, views in patient_views.items()
            if 'FRONTAL' in views and 'LATERAL' in views
        ])
        self.patient_views = {
            pid: patient_views[pid] for pid in self.pair_patients
        }

        n_paired    = len(self.pair_patients)
        n_per_batch = batch_size // 2
        self._num_batches = n_paired // n_per_batch
        print(f"  PatientPairBatchSampler: {n_paired} paired patients | "
              f"{n_per_batch} patients/batch | {self._num_batches} batches/epoch")

    def __len__(self):
        return self._num_batches

    def __iter__(self):
        patients = list(self.pair_patients)
        if self.shuffle:
            np.random.shuffle(patients)

        n_per_batch = self.batch_size // 2
        for start in range(0, len(patients) - n_per_batch + 1, n_per_batch):
            batch_patients = patients[start : start + n_per_batch]
            indices = []
            for pid in batch_patients:
                indices.append(self.patient_views[pid]['FRONTAL'])
                indices.append(self.patient_views[pid]['LATERAL'])
            # Shuffle within batch so frontal/lateral interleave randomly
            np.random.shuffle(indices)
            yield indices


# ---- Transforms --------------------------------------------------------------

def get_train_transform(image_size=None):
    """
    Training augmentation pipeline.
    If images are already pre-resized to target size, T.Resize is skipped
    (controlled by config.IMAGES_PRE_RESIZED) to save CPU time.
    """
    size = image_size or config.VISION_IMAGE_SIZE
    transforms = []
    if not getattr(config, 'IMAGES_PRE_RESIZED', False):
        transforms.append(T.Resize((size, size)))
    transforms += [
        T.RandomHorizontalFlip(p=0.3),
        T.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.95, 1.05)),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        T.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
    ]
    return T.Compose(transforms)


def get_val_transform(image_size=None):
    size = image_size or config.VISION_IMAGE_SIZE
    transforms = []
    if not getattr(config, 'IMAGES_PRE_RESIZED', False):
        transforms.append(T.Resize((size, size)))
    transforms += [
        T.ToTensor(),
        T.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
    ]
    return T.Compose(transforms)


def get_dataloaders(batch_size=None, num_workers=None, image_size=None, seed=42):
    batch_size  = batch_size  or config.BATCH_SIZE
    num_workers = num_workers if num_workers is not None else config.NUM_WORKERS
    image_size  = image_size  or config.VISION_IMAGE_SIZE

    train_ds = IUXrayDataset(split='train', seed=seed, transform=get_train_transform(image_size))
    val_ds   = IUXrayDataset(split='val',   seed=seed, transform=get_val_transform(image_size))
    test_ds  = IUXrayDataset(split='test',  seed=seed, transform=get_val_transform(image_size))

    pair_sampler = PatientPairBatchSampler(train_ds, batch_size=batch_size, shuffle=True)

    train_loader = DataLoader(train_ds, batch_sampler=pair_sampler,
                              num_workers=num_workers, pin_memory=config.PIN_MEMORY)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=config.PIN_MEMORY)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=config.PIN_MEMORY)

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds
