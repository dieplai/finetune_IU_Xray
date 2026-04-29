# Benchmark Data

Thu muc nay chua benchmark assets co dinh duoc sinh tu `data/v8_clean.csv` voi:
- `seed = 42`
- split patient-wise

## File

- `iu_xray_test_manifest_seed42.csv`
  Danh sach full test set co dinh de cham model

- `iu_xray_test_patient_ids_seed42.json`
  Danh sach patient id nam trong test split

- `iu_xray_fixed_qualitative_queries_seed42.csv`
  Tap query qualitative co dinh de hai model so tren cung mot bo query

- `benchmark_summary.json`
  Tong ket nhanh ve benchmark data

## Nguyen tac

Neu so sanh 2 model:
- khong doi `seed`
- khong doi `manifest`
- khong doi `fixed queries`

Neu doi 1 trong 3 thu nay, ket qua so sanh khong con cong bang nua.
