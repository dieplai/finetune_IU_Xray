"""
train_v8.py — Self-contained Training Script for Paper
=======================================================
Architecture : SwinV2-Base (384px) + Bio_ClinicalBERT + MLP projection
Loss         : Clustering-Guided InfoNCE (Task 3 of paper)
               + Cross-View InfoNCE (frontal <-> lateral same patient)
Eval metrics :
  Strict  R@1/5/10 — ground truth = exact same patient_id
  Cluster R@1/5/10 — ground truth = any shared pathological finding
  (both i2t and t2i directions)
Data         : /root/v8_dataset/v8_clean.csv + images_384/
"""
import os, sys, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoTokenizer
import torchvision.transforms as T
from PIL import Image

# ── Paths & hyperparameters ────────────────────────────────────────────────────
CSV_PATH    = "/root/v8_dataset/v8_clean.csv"
IMG_DIR     = "/root/v8_dataset/images_384"
OUT_DIR     = "/root/v8_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# 14 CheXpert cols as stored in v8_clean.csv (order matters for labels tensor)
PATH_COLS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices'
]

IMG_SIZE    = 384     # SwinV2-Base native
BATCH_SIZE  = 8       # Drop to 8 to avoid Phase 2 memory fragmentation
GRAD_ACCUM  = 16      # Effective batch = 128
NUM_EPOCHS  = 30
EVAL_EVERY  = 3       # eval on val set every 3 epochs
FREEZE_EP   = 5       # freeze encoders for first 5 epochs, train heads only
DISEASE_EP  = 3       # Clustering-Guided loss activates at epoch 3

LR_HEAD     = 1e-4    # Projection heads
LR_ENC      = 1e-5    # Encoders (after unfreeze)
LR_TEXT_ENC = 5e-5    # Bio_ClinicalBERT (slightly higher than vision)
WEIGHT_DECAY= 0.01
MAX_GRAD    = 1.0
CROSS_VIEW_W= 0.3

IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD  = (0.229, 0.224, 0.225)
TEXT_MAX_LEN = 128
NUM_WORKERS  = 4
SEED         = 42

# ── Logging ───────────────────────────────────────────────────────────────────
log_file = open(os.path.join(OUT_DIR, "train.log"), "w", encoding="utf-8")

def log(msg, also_print=True):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    log_file.write(line + "\n")
    log_file.flush()
    if also_print:
        print(line)

# ── Transforms ────────────────────────────────────────────────────────────────
# Images already 384x384 → skip T.Resize, only do augmentation
def train_tf():
    return T.Compose([
        T.RandomHorizontalFlip(p=0.3),
        T.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.95, 1.05)),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        T.Normalize(IMG_MEAN, IMG_STD),
    ])

def eval_tf():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(IMG_MEAN, IMG_STD),
    ])

# ── Dataset ───────────────────────────────────────────────────────────────────
def patient_split(df, split, val=0.1, test=0.1, seed=SEED):
    pats = df['patient_id'].astype(str).unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(pats)
    n = len(pats)
    n_t = int(n * test); n_v = int(n * val)
    splits = {
        'test' : set(pats[:n_t]),
        'val'  : set(pats[n_t:n_t+n_v]),
        'train': set(pats[n_t+n_v:])
    }
    return df[df['patient_id'].astype(str).isin(splits[split])].reset_index(drop=True)

class IUXrayDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        img  = Image.open(os.path.join(self.img_dir, row['image_id'])).convert('RGB')
        labs = np.array([row.get(c, 0) for c in PATH_COLS], dtype=np.float32)
        return {
            'image'  : self.transform(img),
            'caption': row['org_caption'],
            'pid'    : str(row['patient_id']),
            'proj'   : str(row.get('projection', '')).strip().lower()[:1],  # 'f' or 'l'
            'labels' : torch.from_numpy(labs),  # (14,) CheXpert hard labels
        }


class PatientPairSampler(Sampler):
    """Guarantees each batch: B//2 patients × 2 views (frontal + lateral)."""
    def __init__(self, dataset, batch_size, shuffle=True):
        assert batch_size % 2 == 0
        self.B   = batch_size
        self.shuf= shuffle
        df       = dataset.df

        pv = {}
        for i, row in df.iterrows():
            pid  = str(row['patient_id'])
            proj = str(row.get('projection', '')).strip().lower()
            view = 'F' if proj.startswith('f') else 'L'
            pv.setdefault(pid, {})[view] = i

        self.pairs = {pid: v for pid, v in pv.items() if 'F' in v and 'L' in v}
        self.pids  = sorted(self.pairs)
        n_per      = batch_size // 2
        self._n    = len(self.pids) // n_per
        log(f"  PatientPairSampler: {len(self.pids)} paired patients | "
            f"{n_per} patients/batch | {self._n} batches/epoch")

    def __len__(self): return self._n

    def __iter__(self):
        pids = list(self.pids)
        if self.shuf: np.random.shuffle(pids)
        n_per = self.B // 2
        for start in range(0, len(pids) - n_per + 1, n_per):
            batch_pids = pids[start:start+n_per]
            idxs = []
            for pid in batch_pids:
                idxs.append(self.pairs[pid]['F'])
                idxs.append(self.pairs[pid]['L'])
            np.random.shuffle(idxs)
            yield idxs


def collate(batch):
    return {
        'image'  : torch.stack([b['image']    for b in batch]),
        'caption': [b['caption'] for b in batch],
        'pid'    : [b['pid']     for b in batch],
        'proj'   : [b['proj']    for b in batch],
        'labels' : torch.stack([b['labels']   for b in batch]),
    }


# ── Model ─────────────────────────────────────────────────────────────────────
from src.models import MedicalSwinBERT, init_tokenizer

# ── Loss: Clustering-Guided InfoNCE ───────────────────────────────────────────
def clustering_guided_infonce(logits, labels):
    """
    Task 3 of paper: if img A and report B share a disease cluster,
    they are NOT treated as negatives.

    labels[:, 0]  = No Finding
    labels[:, 1:] = 13 pathological findings

    Mask[i,j] = True if:
      - BOTH have at least one SAME pathological finding (overlap), OR
      - BOTH are "No Finding" (same normal cluster)
    """
    path_labs = labels[:, 1:]              # (B, 13) — exclude No Finding
    is_normal = (path_labs.sum(-1) == 0)  # (B,)

    # Shared pathology: any common positive finding
    overlap     = (path_labs.unsqueeze(1) * path_labs.unsqueeze(0)).sum(-1) > 0  # (B,B)
    both_normal = is_normal.unsqueeze(1) & is_normal.unsqueeze(0)                 # (B,B)

    mask = (overlap | both_normal).float()     # (B, B) — soft positive mask
    mask_sum = mask.sum(-1).clamp(min=1)

    log_i2t = F.log_softmax(logits,   dim=-1)
    log_t2i = F.log_softmax(logits.T, dim=-1)

    loss_i2t = -(mask   * log_i2t).sum(-1) / mask_sum
    loss_t2i = -(mask.T * log_t2i).sum(-1) / mask_sum

    return (loss_i2t.mean() + loss_t2i.mean()) / 2


def cross_view_loss(logits_fv_lv):
    """InfoNCE between frontal and lateral of same patients."""
    n = logits_fv_lv.shape[0]
    if n < 2:
        return logits_fv_lv.sum() * 0.0
    labels = torch.arange(n, device=logits_fv_lv.device)
    return (F.cross_entropy(logits_fv_lv, labels) +
            F.cross_entropy(logits_fv_lv.T, labels)) / 2


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def encode_all(model, loader, tokenizer, device):
    model.eval()
    all_ie, all_te, all_caps, all_pids, all_labs = [], [], [], [], []
    for batch in loader:
        imgs = batch['image'].to(device)
        tok  = tokenizer(batch['caption'], padding='max_length', truncation=True,
                         max_length=TEXT_MAX_LEN, return_tensors='pt').to(device)
        ie, te = model(imgs, tok['input_ids'], tok['attention_mask'])
        all_ie.append(ie.cpu()); all_te.append(te.cpu())
        all_caps.extend(batch['caption']); all_pids.extend(batch['pid'])
        all_labs.append(batch['labels'])
    return (torch.cat(all_ie), torch.cat(all_te),
            all_caps, all_pids, torch.cat(all_labs).numpy())


def recall_at_k(sim_matrix, ground_truth_mask, ks=(1, 5, 10)):
    """
    sim_matrix : (N_queries, N_gallery)
    ground_truth_mask : (N_queries, N_gallery) bool
    Returns dict of R@k values.
    """
    results = {}
    N = sim_matrix.shape[0]
    for k in ks:
        topk_idx = sim_matrix.topk(k, dim=-1).indices  # (N, k)
        hit = 0
        for i in range(N):
            if ground_truth_mask[i][topk_idx[i]].any():
                hit += 1
        results[f"R@{k}"] = hit / N * 100
    return results


def evaluate(model, loader, tokenizer, device):
    """
    Compute two types of evaluation metrics:

    STRICT metrics — ground truth = exact same patient_id
      Traditional retrieval: retrieve the exact same patient's image/text.
      Based on patient_id match.

    CLUSTER metrics — ground truth = any shared pathological finding
      Clinically meaningful: retrieve any sample from same disease cluster.
      Based on label overlap (any shared CheXpert pathology).
      This is what the paper's Clustering-Guided loss optimizes for.
    """
    ie, te, caps, pids, labs = encode_all(model, loader, tokenizer, device)
    N = ie.shape[0]

    sim_i2t = ie @ te.T   # (N_img, N_text)
    sim_t2i = te @ ie.T   # (N_text, N_img)

    pid_arr = np.array(pids)

    # ── STRICT ground truth: same patient_id ────────────────────────────────
    gt_strict = torch.tensor(
        pid_arr[:, None] == pid_arr[None, :],   # (N, N) bool
        dtype=torch.bool
    )

    # ── CLUSTER ground truth: shared pathological finding ───────────────────
    # labs[:, 0] = No Finding → exclude (too common, not a meaningful cluster)
    path_labs = torch.tensor(labs[:, 1:], dtype=torch.float32)  # (N, 13)
    # Any shared pathology: dot product > 0 means at least 1 shared label
    overlap = (path_labs @ path_labs.T) > 0                      # (N, N) bool
    # For pure "No Finding" pairs: both have no pathology → same cluster
    is_normal = (path_labs.sum(-1) == 0)                          # (N,)
    both_normal = is_normal.unsqueeze(1) & is_normal.unsqueeze(0) # (N, N)
    gt_cluster = overlap | both_normal                             # (N, N) bool

    KS = (1, 5, 10)
    strict_i2t  = recall_at_k(sim_i2t, gt_strict,  ks=KS)
    strict_t2i  = recall_at_k(sim_t2i, gt_strict.T, ks=KS)
    cluster_i2t = recall_at_k(sim_i2t, gt_cluster,  ks=KS)
    cluster_t2i = recall_at_k(sim_t2i, gt_cluster.T, ks=KS)

    return strict_i2t, strict_t2i, cluster_i2t, cluster_t2i


# ── Main training loop ─────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log("=" * 65)
    log("  train_v8.py | SwinV2-Base + Bio_ClinicalBERT")
    log("  Clustering-Guided InfoNCE + Cross-View InfoNCE")
    log("=" * 65)
    log(f"Device : {device}")
    if device.type == 'cuda':
        log(f"GPU    : {torch.cuda.get_device_name(0)}")
        log(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory//1024**2} MB")

    # Data
    log("\n[DATA]")
    df = pd.read_csv(CSV_PATH)
    df['patient_id'] = df['patient_id'].astype(str)
    df_train = patient_split(df, 'train')
    df_val   = patient_split(df, 'val')
    df_test  = patient_split(df, 'test')
    log(f"  Train: {len(df_train)} images | {df_train['patient_id'].nunique()} patients")
    log(f"  Val  : {len(df_val)}   images | {df_val['patient_id'].nunique()} patients")
    log(f"  Test : {len(df_test)}  images | {df_test['patient_id'].nunique()} patients")

    train_ds     = IUXrayDataset(df_train, IMG_DIR, train_tf())
    val_ds       = IUXrayDataset(df_val,   IMG_DIR, eval_tf())
    test_ds      = IUXrayDataset(df_test,  IMG_DIR, eval_tf())
    pair_sampler = PatientPairSampler(train_ds, BATCH_SIZE, shuffle=True)

    train_loader = DataLoader(train_ds, batch_sampler=pair_sampler,
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate)

    # Model
    log("\n[MODEL]")
    model     = MedicalSwinBERT(img_size=IMG_SIZE).to(device)
    tokenizer = init_tokenizer()
    n_total   = sum(p.numel() for p in model.parameters()) / 1e6
    n_train   = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    log(f"  Total params    : {n_total:.1f}M")
    log(f"  Trainable params: {n_train:.1f}M")

    # Phase 1: freeze encoders, train projection heads only
    log(f"\n[PHASE 1] Freeze encoders for {FREEZE_EP} epochs, train heads only")
    for p in model.get_backbone_params():
        p.requires_grad = False
    opt = AdamW(model.get_head_params(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)

    best_r1 = 0.0
    best_epoch = 0
    history = []
    
    # ── PyTorch AMP for Memory Efficiency ──
    scaler = torch.amp.GradScaler('cuda')

    log("\n" + "=" * 65)
    log(f"  TRAINING START: {NUM_EPOCHS} epochs")
    log(f"  Batch={BATCH_SIZE} | GradAccum={GRAD_ACCUM} | EffBatch={BATCH_SIZE*GRAD_ACCUM}")
    log(f"  ClusteringGuided from epoch {DISEASE_EP}")
    log(f"  CrossViewWeight = {CROSS_VIEW_W}")
    log("=" * 65)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        # Phase 2: unfreeze encoders
        if epoch == FREEZE_EP + 1:
            log(f"\n[PHASE 2] Unfreezing encoders at epoch {epoch}")
            
            # Clear reserved memory from Phase 1 before computing large graphs
            torch.cuda.empty_cache()

            for p in model.get_backbone_params():
                p.requires_grad = True
            opt = AdamW([
                {'params': model.image_encoder.parameters(), 'lr': LR_ENC},
                {'params': model.text_encoder.parameters(),  'lr': LR_TEXT_ENC},
                {'params': model.get_head_params(),          'lr': LR_HEAD},
            ], weight_decay=WEIGHT_DECAY)
            total_steps  = len(train_loader) * (NUM_EPOCHS - FREEZE_EP) // GRAD_ACCUM
            warmup_steps = max(1, int(total_steps * 0.05))
            sched = SequentialLR(opt, [
                LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
                CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-8),
            ], milestones=[warmup_steps])
            log(f"  LR: vision={LR_ENC:.0e}, text={LR_TEXT_ENC:.0e}, head={LR_HEAD:.0e}")
            log(f"  Scheduler: Warmup({warmup_steps} steps) → CosineAnnealing")

        # ── Training ──
        model.train()
        losses_total, losses_cluster, losses_cv = [], [], []
        opt.zero_grad()

        for step, batch in enumerate(train_loader):
            imgs   = batch['image'].to(device)
            labels = batch['labels'].to(device)
            pids   = batch['pid']
            projs  = batch['proj']

            tok = tokenizer(
                batch['caption'], padding='max_length', truncation=True,
                max_length=TEXT_MAX_LEN, return_tensors='pt'
            ).to(device)

            with torch.amp.autocast('cuda'):
                ie, te = model(imgs, tok['input_ids'], tok['attention_mask'])

                with torch.no_grad():
                    model.logit_scale.data.clamp_(np.log(5.0), np.log(100.0))
                scale  = model.logit_scale.exp()
                logits = scale * (ie @ te.T)

                # Primary: Clustering-Guided InfoNCE (activates at DISEASE_EP)
                if epoch >= DISEASE_EP:
                    loss_cluster = clustering_guided_infonce(logits, labels)
                else:
                    # Warm-up: standard InfoNCE (diagonal = positives)
                    lbls_std = torch.arange(logits.shape[0], device=device)
                    loss_cluster = (F.cross_entropy(logits, lbls_std) +
                                    F.cross_entropy(logits.T, lbls_std)) / 2

                # Secondary: Cross-View InfoNCE (frontal <-> lateral same patient)
                f_idx = [i for i, p in enumerate(projs) if p.startswith('f')]
                l_idx = [i for i, p in enumerate(projs) if p.startswith('l')]
                if len(f_idx) >= 2 and len(l_idx) >= 2:
                    n_pairs = min(len(f_idx), len(l_idx))
                    f_emb = ie[f_idx[:n_pairs]]
                    l_emb = ie[l_idx[:n_pairs]]
                    cv_logits = scale * (f_emb @ l_emb.T)
                    loss_cv   = cross_view_loss(cv_logits)
                else:
                    loss_cv = logits.sum() * 0.0

                loss = loss_cluster + CROSS_VIEW_W * loss_cv
                loss = loss / GRAD_ACCUM

            scaler.scale(loss).backward()

            losses_total.append((loss_cluster + CROSS_VIEW_W * loss_cv).item())
            losses_cluster.append(loss_cluster.item())
            losses_cv.append(loss_cv.item())

            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD)
                
                scaler.step(opt)
                scaler.update()
                
                opt.zero_grad()
                if epoch > FREEZE_EP:
                    sched.step()

        avg_total   = float(np.mean(losses_total))
        avg_cluster = float(np.mean(losses_cluster))
        avg_cv      = float(np.mean(losses_cv))
        temp        = model.logit_scale.exp().item()
        elapsed     = time.time() - t0

        # ── Evaluation ──
        do_eval = (epoch % EVAL_EVERY == 0) or (epoch == NUM_EPOCHS) or (epoch <= 3)
        if do_eval:
            s_i2t, s_t2i, c_i2t, c_t2i = evaluate(model, val_loader, tokenizer, device)

            # Best model: based on mean Strict R@1 (conservative metric)
            strict_r1_mean   = (s_i2t['R@1'] + s_t2i['R@1']) / 2
            cluster_r1_mean  = (c_i2t['R@1'] + c_t2i['R@1']) / 2

            row = dict(
                epoch=epoch, loss=avg_total, loss_cluster=avg_cluster,
                loss_cv=avg_cv, temp=temp,
                strict_i2t_R1=s_i2t['R@1'],  strict_i2t_R5=s_i2t['R@5'],  strict_i2t_R10=s_i2t['R@10'],
                strict_t2i_R1=s_t2i['R@1'],  strict_t2i_R5=s_t2i['R@5'],  strict_t2i_R10=s_t2i['R@10'],
                cluster_i2t_R1=c_i2t['R@1'], cluster_i2t_R5=c_i2t['R@5'], cluster_i2t_R10=c_i2t['R@10'],
                cluster_t2i_R1=c_t2i['R@1'], cluster_t2i_R5=c_t2i['R@5'], cluster_t2i_R10=c_t2i['R@10'],
            )
            history.append(row)

            is_best = strict_r1_mean > best_r1
            if is_best:
                best_r1 = strict_r1_mean; best_epoch = epoch
                torch.save({'epoch': epoch, 'model': model.state_dict(),
                            'best_r1': best_r1,
                            'strict_i2t': s_i2t, 'strict_t2i': s_t2i,
                            'cluster_i2t': c_i2t, 'cluster_t2i': c_t2i,
                            }, os.path.join(OUT_DIR, 'best.pt'))

            log(f"\n{'='*60}")
            log(f"Epoch [{epoch:03d}/{NUM_EPOCHS}]  "
                f"Loss={avg_total:.4f} (cluster={avg_cluster:.4f} cv={avg_cv:.4f})  "
                f"Temp={temp:.1f}  Time={elapsed:.0f}s")
            log(f"")
            log(f"  [STRICT  - same patient]")
            log(f"  i2t | R@1={s_i2t['R@1']:6.2f}%  R@5={s_i2t['R@5']:6.2f}%  R@10={s_i2t['R@10']:6.2f}%")
            log(f"  t2i | R@1={s_t2i['R@1']:6.2f}%  R@5={s_t2i['R@5']:6.2f}%  R@10={s_t2i['R@10']:6.2f}%")
            log(f"  Mean Strict  R@1={strict_r1_mean:.2f}%  R@5={(s_i2t['R@5']+s_t2i['R@5'])/2:.2f}%  R@10={(s_i2t['R@10']+s_t2i['R@10'])/2:.2f}%")
            log(f"")
            log(f"  [CLUSTER - shared pathology]")
            log(f"  i2t | R@1={c_i2t['R@1']:6.2f}%  R@5={c_i2t['R@5']:6.2f}%  R@10={c_i2t['R@10']:6.2f}%")
            log(f"  t2i | R@1={c_t2i['R@1']:6.2f}%  R@5={c_t2i['R@5']:6.2f}%  R@10={c_t2i['R@10']:6.2f}%")
            log(f"  Mean Cluster R@1={cluster_r1_mean:.2f}%  R@5={(c_i2t['R@5']+c_t2i['R@5'])/2:.2f}%  R@10={(c_i2t['R@10']+c_t2i['R@10'])/2:.2f}%")
            log(f"")
            bmark = f'★ BEST Strict R@1' if is_best else f'(best={best_r1:.2f}% @ ep{best_epoch})'
            log(f"  {bmark}")
        else:
            log(f"Epoch [{epoch:03d}/{NUM_EPOCHS}]  "
                f"Loss={avg_total:.4f} (cluster={avg_cluster:.4f} cv={avg_cv:.4f})  "
                f"Temp={temp:.1f}  Time={elapsed:.0f}s")

    # ── Final evaluation on test set ──
    log("\n" + "="*55)
    log("  FINAL TEST EVALUATION")
    log("="*55)
    ckpt = torch.load(os.path.join(OUT_DIR, 'best.pt'), map_location=device)
    model.load_state_dict(ckpt['model'])
    log(f"Loaded best checkpoint from epoch {ckpt['epoch']} (val R@1={ckpt['best_r1']:.2f}%)")

    s_i2t, s_t2i, c_i2t, c_t2i = evaluate(model, test_loader, tokenizer, device)
    log(f"\n  [STRICT - same patient]")
    log(f"  i2t | R@1={s_i2t['R@1']:6.2f}%  R@5={s_i2t['R@5']:6.2f}%  R@10={s_i2t['R@10']:6.2f}%")
    log(f"  t2i | R@1={s_t2i['R@1']:6.2f}%  R@5={s_t2i['R@5']:6.2f}%  R@10={s_t2i['R@10']:6.2f}%")
    log(f"  Mean Strict  R@1={(s_i2t['R@1']+s_t2i['R@1'])/2:.2f}%  "
        f"R@5={(s_i2t['R@5']+s_t2i['R@5'])/2:.2f}%  "
        f"R@10={(s_i2t['R@10']+s_t2i['R@10'])/2:.2f}%")
    log(f"\n  [CLUSTER - shared pathology]")
    log(f"  i2t | R@1={c_i2t['R@1']:6.2f}%  R@5={c_i2t['R@5']:6.2f}%  R@10={c_i2t['R@10']:6.2f}%")
    log(f"  t2i | R@1={c_t2i['R@1']:6.2f}%  R@5={c_t2i['R@5']:6.2f}%  R@10={c_t2i['R@10']:6.2f}%")
    log(f"  Mean Cluster R@1={(c_i2t['R@1']+c_t2i['R@1'])/2:.2f}%  "
        f"R@5={(c_i2t['R@5']+c_t2i['R@5'])/2:.2f}%  "
        f"R@10={(c_i2t['R@10']+c_t2i['R@10'])/2:.2f}%")

    # Save history
    pd.DataFrame(history).to_csv(os.path.join(OUT_DIR, 'history.csv'), index=False)
    with open(os.path.join(OUT_DIR, 'test_results.json'), 'w') as f:
        json.dump({
            'strict':  {'i2t': s_i2t, 't2i': s_t2i},
            'cluster': {'i2t': c_i2t, 't2i': c_t2i},
            'best_epoch': best_epoch, 'best_val_strict_r1': best_r1
        }, f, indent=2)

    log("\nDONE. Results saved to " + OUT_DIR)
    log_file.close()


if __name__ == '__main__':
    main()
