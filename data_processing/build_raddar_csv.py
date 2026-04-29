"""Build unified CSV for raddar: image_id, org_caption, patient_id, projection + 14 labels."""
import os, sys, argparse
import pandas as pd
import numpy as np
sys.path.insert(0, '/root/IU_medclip')
from extract_labels import extract_labels, PATHOLOGIES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raddar_dir', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    reports = pd.read_csv(os.path.join(args.raddar_dir, 'indiana_reports.csv'))
    projs = pd.read_csv(os.path.join(args.raddar_dir, 'indiana_projections.csv'))
    df = projs.merge(reports, on='uid', how='inner')
    print(f"Merged: {len(df)} rows")

    df['findings'] = df['findings'].fillna('')
    df['impression'] = df['impression'].fillna('')
    # Combined caption (findings + impression)
    df['caption'] = (df['findings'].str.strip() + ' ' + df['impression'].str.strip()).str.strip()
    # Remove "XXXX" anonymization tokens
    df['caption'] = df['caption'].str.replace(r'X{2,}', '', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()
    df = df[df['caption'].str.len() > 10].reset_index(drop=True)

    df['image_id'] = df['filename']
    df['org_caption'] = df['caption'].str.lower()
    df['patient_id'] = df['uid'].astype(str)
    print(f"After filter + clean XXXX: {len(df)} rows")

    print("Extracting labels...")
    for col in PATHOLOGIES: df[col] = 0
    for i, row in df.iterrows():
        l = extract_labels(row['org_caption'])
        for col in PATHOLOGIES: df.at[i, col] = l[col]

    print(f"No Finding: {int((df['No Finding']==1).sum())} / {len(df)} ({(df['No Finding']==1).mean()*100:.1f}%)")
    for p in PATHOLOGIES:
        if p == 'No Finding': continue
        n = int((df[p] == 1).sum())
        print(f"  {p:30s}: {n:4d} ({n/len(df)*100:.1f}%)")

    keep_cols = ['uid', 'image_id', 'patient_id', 'projection', 'org_caption'] + list(PATHOLOGIES)
    df[keep_cols].to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}  ({len(df)} rows, {len(keep_cols)} cols)")


if __name__ == '__main__':
    main()
