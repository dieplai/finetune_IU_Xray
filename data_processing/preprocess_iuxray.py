"""
V8 Fast Image Preprocessing — multiprocessing with 28 workers
Skips already-processed images, resumes from where old script left off.
"""
import os, re, json, multiprocessing
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

with open('/root/iu_data_exploration/dataset_paths.json') as f:
    paths = json.load(f)

RAW_BASE    = paths['raw_path']
RAW_IMG_DIR = os.path.join(RAW_BASE, 'images', 'images_normalized')
OUT_DIR     = '/root/v8_dataset'
OUT_IMG_DIR = os.path.join(OUT_DIR, 'images')
OUT_CSV     = os.path.join(OUT_DIR, 'v8_clean.csv')
os.makedirs(OUT_IMG_DIR, exist_ok=True)

PATH_COLS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices'
]

NEGATION = [r'no\s+', r'without\s+', r'absence\s+of\s+',
            r'there\s+is\s+no\s+', r'there\s+are\s+no\s+',
            r'free\s+of\s+', r'rule\s+out\s+', r'negative\s+for\s+']

PATHOLOGY_PATTERNS = {
    'No Finding':                [r'no\s+acute', r'\bnormal\b', r'unremarkable', r'no\s+evidence\s+of'],
    'Enlarged Cardiomediastinum':[r'enlarged\s+cardiomediastin', r'widened\s+mediastinum'],
    'Cardiomegaly':              [r'cardiomegal', r'enlarged\s+heart', r'cardiac\s+silhouette.*enlarg', r'heart.*enlarg'],
    'Lung Lesion':               [r'\blesion\b', r'\bnodule\b', r'mass(?!ive)', r'granuloma'],
    'Lung Opacity':              [r'opacit', r'opacification', r'\bhaz', r'infiltrat'],
    'Edema':                     [r'pulmonary\s+edema', r'\bedema\b', r'vascular\s+congestion'],
    'Consolidation':             [r'consolidation', r'airspace\s+disease'],
    'Pneumonia':                 [r'pneumonia', r'bronchopneumonia'],
    'Atelectasis':               [r'atelectas', r'\bcollapse\b', r'subsegmental', r'volume\s+loss'],
    'Pneumothorax':              [r'pneumothora'],
    'Pleural Effusion':          [r'pleural\s+effusion', r'\beffusion\b'],
    'Pleural Other':             [r'pleural\s+thicken', r'pleural\s+calcif'],
    'Fracture':                  [r'\bfracture\b'],
    'Support Devices':           [r'\btube\b', r'\bcatheter\b', r'pacemaker', r'\bpicc\b'],
}


def extract_labels(text):
    text = str(text).lower()
    sents = re.split(r'[.;]', text)
    labels = {p: 0 for p in PATH_COLS}
    for sent in sents:
        has_neg = any(re.search(neg, sent) for neg in NEGATION)
        for path, patterns in PATHOLOGY_PATTERNS.items():
            if any(re.search(p, sent) for p in patterns) and not has_neg:
                labels[path] = 1
    if any(labels[p] == 1 for p in PATH_COLS if p != 'No Finding'):
        labels['No Finding'] = 0
    return labels


def clean_text(text):
    if pd.isna(text): return ''
    text = str(text).lower().strip()
    text = re.sub(r'x{2,}', '', text, flags=re.I)
    text = re.sub(r'\b\d{1,3}[\s-]year[s]?[\s-]old\b', '', text, flags=re.I)
    text = re.sub(r'\b(male|female|man\b|woman\b|boy\b|girl\b)\b', '', text, flags=re.I)
    text = re.sub(r'\b(findings|impression|clinical\s+information|comparison)\s*:?\s*', ' ', text, flags=re.I)
    text = re.sub(r'^\s*\d+[.)]\s*', '', text, flags=re.M)
    text = re.sub(r'[;|\[\]()#@^~*]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\.{2,}', '.', text)
    return text


def process_one_image(args):
    """Worker function: pad-to-square, convert to RGB, save PNG."""
    img_name, src_dir, dst_dir = args
    dst = os.path.join(dst_dir, img_name)
    if os.path.exists(dst):
        return (img_name, 'skip', None)
    try:
        img_gray = Image.open(os.path.join(src_dir, img_name)).convert('L')
        w, h = img_gray.size
        img_rgb = img_gray.convert('RGB')
        max_dim = max(w, h)
        padded = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))
        padded.paste(img_rgb, ((max_dim - w) // 2, (max_dim - h) // 2))
        padded.save(dst, 'PNG', optimize=False)
        return (img_name, 'ok', (w, h))
    except Exception as e:
        return (img_name, 'fail', str(e))


if __name__ == '__main__':
    print("=" * 65)
    print("  V8 Fast Preprocessing (28-worker multiprocessing)")
    print("=" * 65)

    # ── Build DataFrame ───────────────────────────────────────────────
    print("\n[1] Building DataFrame from RAW CSVs...")
    df_proj = pd.read_csv(os.path.join(RAW_BASE, 'indiana_projections.csv'))
    df_rep  = pd.read_csv(os.path.join(RAW_BASE, 'indiana_reports.csv'))
    df = df_proj.merge(df_rep, on='uid', how='inner')
    df['findings']    = df['findings'].fillna('').str.strip()
    df['impression']  = df['impression'].fillna('').str.strip()
    df['raw_caption'] = (df['findings'] + '. ' + df['impression']).str.strip('. ')
    df['org_caption'] = df['raw_caption'].apply(clean_text)
    df['word_count']  = df['org_caption'].str.split().str.len()
    before = len(df)
    df = df[df['word_count'] >= 10].reset_index(drop=True)
    df['image_id']   = df['filename']
    df['patient_id'] = df['uid'].astype(str)
    available = set(os.listdir(RAW_IMG_DIR))
    df = df[df['image_id'].isin(available)].reset_index(drop=True)
    print(f"  DataFrame: {len(df)} rows | {df['patient_id'].nunique()} patients")

    # ── Fast parallel image processing ───────────────────────────────
    all_imgs = df['image_id'].unique().tolist()
    already_done = sum(1 for f in all_imgs if os.path.exists(os.path.join(OUT_IMG_DIR, f)))
    todo = [f for f in all_imgs if not os.path.exists(os.path.join(OUT_IMG_DIR, f))]
    print(f"\n[2] Image processing: {len(all_imgs)} total, {already_done} already done, {len(todo)} to process")

    if todo:
        args_list = [(img, RAW_IMG_DIR, OUT_IMG_DIR) for img in todo]
        NUM_WORKERS = 28
        print(f"  Using {NUM_WORKERS} parallel workers...")
        results = []
        with multiprocessing.Pool(NUM_WORKERS) as pool:
            for r in tqdm(pool.imap_unordered(process_one_image, args_list),
                          total=len(args_list), desc="Padding images"):
                results.append(r)

        ok   = [r for r in results if r[1] == 'ok']
        fail = [r for r in results if r[1] == 'fail']
        print(f"\n  Processed: {len(ok)} OK | {len(fail)} failed")
        if fail:
            for f in fail[:5]: print(f"    FAIL: {f}")
            bad_ids = {f[0] for f in fail}
            df = df[~df['image_id'].isin(bad_ids)].reset_index(drop=True)

        # Verify aspect ratios from results
        sizes = [(r[2][0], r[2][1]) for r in ok if r[2]]
        if sizes:
            aspects = [w/h for w, h in sizes]
            very_skewed = sum(1 for a in aspects if a < 0.7 or a > 1.4)
            print(f"  Aspect ratio range: {min(aspects):.3f} – {max(aspects):.3f}")
            print(f"  Very skewed (would distort): {very_skewed} → now all padded to square ✓")
    else:
        print("  All images already processed!")

    # ── Spot-check ───────────────────────────────────────────────────
    print("\n[3] Spot-checking padded images (10 samples)...")
    sample = df['image_id'].sample(min(10, len(df)), random_state=42).tolist()
    non_square, non_rgb = [], []
    for name in sample:
        img = Image.open(os.path.join(OUT_IMG_DIR, name))
        w, h = img.size
        if w != h: non_square.append((name, w, h))
        if img.mode != 'RGB': non_rgb.append(name)
    if non_square or non_rgb:
        print(f"  WARNING: {len(non_square)} non-square, {len(non_rgb)} non-RGB!")
    else:
        ex = Image.open(os.path.join(OUT_IMG_DIR, sample[0]))
        print(f"  All OK: RGB + Square ({ex.size[0]}×{ex.size[1]}) ✓")

    # ── Label extraction ─────────────────────────────────────────────
    print(f"\n[4] Extracting pathology labels ({len(df)} rows)...")
    for col in PATH_COLS:
        df[col] = 0
    for i, row in df.iterrows():
        lbs = extract_labels(row['org_caption'])
        for col in PATH_COLS:
            df.at[i, col] = lbs[col]

    # ── Validation ───────────────────────────────────────────────────
    print("\n[5] Validation Report:")
    print(f"  Total rows:      {len(df)}")
    print(f"  Unique patients: {df['patient_id'].nunique()}")
    print(f"  Unique captions: {df['org_caption'].nunique()}")
    print(f"  Frontal/Lateral: {(df['projection']=='Frontal').sum()} / {(df['projection']=='Lateral').sum()}")
    print(f"  Word count:      min={df['word_count'].min()}, max={df['word_count'].max()}, mean={df['word_count'].mean():.1f}")
    print(f"\n  Pathology distribution:")
    for col in PATH_COLS:
        n = int((df[col] == 1).sum())
        bar = '█' * max(1, int(n / len(df) * 35))
        print(f"    {col:30s}: {n:4d} ({n/len(df)*100:5.1f}%) {bar}")

    no_label = (df[PATH_COLS].sum(axis=1) == 0).sum()
    print(f"\n  Rows with NO label: {no_label} ({no_label/len(df)*100:.1f}%)")

    # ── Save ─────────────────────────────────────────────────────────
    keep = ['image_id', 'patient_id', 'projection', 'org_caption'] + PATH_COLS
    df[keep].to_csv(OUT_CSV, index=False, encoding='utf-8-sig')

    summary = {
        "total_rows": len(df), "unique_patients": int(df['patient_id'].nunique()),
        "image_source": "raddar RAW — padded to square RGB PNG",
        "text_source": "raddar RAW reports — deep cleaned",
        "img_dir_for_training": OUT_IMG_DIR,
        "csv_for_training": OUT_CSV,
        "resize_note": "Images are square PNG. T.Resize(384,384) in training = pure scale, zero distortion."
    }
    with open(os.path.join(OUT_DIR, 'dataset_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\n{'='*65}")
    print(f"  PREPROCESSING COMPLETE")
    print(f"{'='*65}")
    print(f"  CSV:    {OUT_CSV}")
    print(f"  Images: {OUT_IMG_DIR}")
    print(f"  Total:  {len(df)} rows | {df['patient_id'].nunique()} patients")
