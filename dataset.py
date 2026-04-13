"""
Dataset class for IU-Xray with cluster-aware sampling.
"""
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from disease_cluster import extract_diseases, build_disease_matrix
import config


class IUXrayDataset(Dataset):
    """IU-Xray dataset with disease cluster labels."""
    
    def __init__(self, csv_path=None, img_dir=None, transform=None, split='train',
                 val_split=0.1, test_split=0.1, seed=42):
        self.csv_path = csv_path or config.CSV_PATH
        self.img_dir = img_dir or config.IMG_DIR
        
        df = pd.read_csv(self.csv_path)
        
        # Extract patient ID and disease clusters
        df['patient_id'] = df['image_id'].apply(lambda x: x.split('_')[0])
        df['diseases'] = df['org_caption'].apply(extract_diseases)
        # Store as sorted list (not set) for collation
        df['disease_list'] = df['diseases'].apply(lambda x: '|'.join(sorted(x)))
        
        # Stratified split by disease cluster
        np.random.seed(seed)
        unique_diseases = df['disease_list'].unique()
        
        train_idx, val_idx, test_idx = [], [], []
        for disease in unique_diseases:
            mask = df['disease_list'] == disease
            indices = df[mask].index.tolist()
            np.random.shuffle(indices)
            n = len(indices)
            n_test = max(1, int(n * test_split))
            n_val = max(1, int(n * val_split))
            test_idx.extend(indices[:n_test])
            val_idx.extend(indices[n_test:n_test + n_val])
            train_idx.extend(indices[n_test + n_val:])
        
        if split == 'train':
            self.df = df.loc[train_idx].reset_index(drop=True)
        elif split == 'val':
            self.df = df.loc[val_idx].reset_index(drop=True)
        else:
            self.df = df.loc[test_idx].reset_index(drop=True)
        
        # Disease sets for similarity computation
        self.disease_sets = self.df['diseases'].tolist()
        
        # Build disease category mapping
        self.all_disease_sets, self.category_to_idx = build_disease_matrix(
            self.df['org_caption'].tolist()
        )
        
        self.transform = transform or get_train_transform()
        self.split = split
        
        print(f"[{split}] {len(self.df)} samples, "
              f"{self.df['patient_id'].nunique()} patients, "
              f"{len(self.category_to_idx)} disease categories")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = os.path.join(self.img_dir, row['image_id'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Caption
        caption = row['org_caption']
        
        # Disease cluster (multi-hot encoding)
        n_cats = len(self.category_to_idx)
        disease_vec = torch.zeros(n_cats)
        for disease in self.disease_sets[idx]:
            if disease in self.category_to_idx:
                disease_vec[self.category_to_idx[disease]] = 1.0
        
        return {
            'image': image,
            'caption': caption,
            'image_id': row['image_id'],
            'patient_id': row['patient_id'],
            'disease_vec': disease_vec,
            'disease_list': row['disease_list'],  # string, not set
        }


def get_train_transform(image_size=512):
    """Training transforms with augmentation."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.95, 1.05)),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform(image_size=512):
    """Validation/test transforms (no augmentation)."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_dataloaders(batch_size=32, num_workers=4, image_size=512, seed=42):
    """Create train/val/test dataloaders."""
    train_ds = IUXrayDataset(
        split='train', seed=seed,
        transform=get_train_transform(image_size)
    )
    val_ds = IUXrayDataset(
        split='val', seed=seed,
        transform=get_val_transform(image_size)
    )
    test_ds = IUXrayDataset(
        split='test', seed=seed,
        transform=get_val_transform(image_size)
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds


if __name__ == "__main__":
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = get_dataloaders(
        batch_size=4, num_workers=2, image_size=512
    )
    batch = next(iter(train_loader))
    print(f"Image shape: {batch['image'].shape}")
    print(f"Captions: {batch['caption'][:2]}")
    print(f"Disease vec shape: {batch['disease_vec'].shape}")
    print(f"Disease vec sample: {batch['disease_vec'][0]}")
