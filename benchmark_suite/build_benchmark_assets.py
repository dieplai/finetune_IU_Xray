import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "v8_clean.csv"
OUT_DIR = ROOT / "benchmark_suite" / "data"
SEED = 42
MANIFEST_NAME = f"iu_xray_test_manifest_seed{SEED}.csv"
PATIENT_IDS_NAME = f"iu_xray_test_patient_ids_seed{SEED}.json"
FIXED_QUERIES_NAME = f"iu_xray_fixed_qualitative_queries_seed{SEED}.csv"


def patient_split(df: pd.DataFrame, split: str, val: float = 0.1, test: float = 0.1, seed: int = SEED) -> pd.DataFrame:
    pats = list(df["patient_id"].astype(str).unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(pats)
    n = len(pats)
    nt = int(n * test)
    nv = int(n * val)
    mapping = {
        "test": set(pats[:nt]),
        "val": set(pats[nt: nt + nv]),
        "train": set(pats[nt + nv:]),
    }
    return df[df["patient_id"].astype(str).isin(mapping[split])].reset_index(drop=True)


def build_assets() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    df["patient_id"] = df["patient_id"].astype(str)
    df_test = patient_split(df, "test", seed=SEED).reset_index(drop=True)

    manifest_cols = ["image_id", "patient_id", "projection", "org_caption"]
    label_cols = [c for c in df.columns if c not in manifest_cols]
    ordered_cols = manifest_cols + [c for c in label_cols if c != "patient_id"]
    manifest = df_test[ordered_cols].copy()
    manifest.to_csv(OUT_DIR / MANIFEST_NAME, index=False)

    with open(OUT_DIR / PATIENT_IDS_NAME, "w", encoding="utf-8") as f:
        json.dump(sorted(df_test["patient_id"].astype(str).unique().tolist()), f, indent=2)

    query_rows = []
    for pid, g in df_test.groupby("patient_id", sort=True):
        g = g.reset_index()
        row = g.iloc[0]
        query_rows.append({
            "query_index": int(row["index"]),
            "patient_id": str(row["patient_id"]),
            "image_id": str(row["image_id"]),
            "projection": str(row["projection"]),
            "report": str(row["org_caption"]),
        })
    fixed_queries = pd.DataFrame(query_rows).head(20)
    fixed_queries.to_csv(OUT_DIR / FIXED_QUERIES_NAME, index=False)

    summary = {
        "seed": SEED,
        "csv_path": str(CSV_PATH.relative_to(ROOT)),
        "manifest_csv": f"benchmark_suite/data/{MANIFEST_NAME}",
        "patient_ids_json": f"benchmark_suite/data/{PATIENT_IDS_NAME}",
        "fixed_queries_csv": f"benchmark_suite/data/{FIXED_QUERIES_NAME}",
        "test_rows": int(len(df_test)),
        "test_patients": int(df_test["patient_id"].nunique()),
        "fixed_queries": int(len(fixed_queries)),
    }
    with open(OUT_DIR / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    build_assets()
