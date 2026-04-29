# IU-Xray Benchmark Suite

Bo benchmark nay duoc dung de so sanh checkpoint mot cach cong bang va lap lai duoc.

Muc tieu:
- Dung **cung mot test split co dinh**
- Dung **cung mot script evaluate**
- Bao cao day du `strict` va `cluster`, ca hai chieu `image->text` va `text->image`
- Tao **fixed qualitative queries** de hai model so tren cung mot tap query

## Cau truc

- `benchmark_suite/build_benchmark_assets.py`
  Tao manifest test co dinh va qualitative queries co dinh tu `data/v8_clean.csv`
- `benchmark_suite/evaluate_shared.py`
  Cham 1 checkpoint tren benchmark chung
- `benchmark_suite/compare_models.py`
  Cham nhieu checkpoint va xuat bang so sanh
- `benchmark_suite/generate_fixed_report.py`
  Tao bao cao HTML/JSON/CSV tren tap query qualitative co dinh
- `benchmark_suite/run_on_colab.sh`
  Cach chay nhanh tren Colab
- `benchmark_suite/data/`
  Chua benchmark data co dinh da sinh ra

## Benchmark chinh

Metric chinh:
- `strict_mean_r1`

Metric phu:
- `strict_i2t_r5`, `strict_t2i_r5`
- `strict_i2t_r10`, `strict_t2i_r10`

Metric ho tro thesis:
- `cluster_mean_r1`
- `cluster_i2t_r5`, `cluster_t2i_r5`
- `cluster_i2t_r10`, `cluster_t2i_r10`

Quy tac ket luan:
- `strict_mean_r1` dung de kiem tra exact patient/study retrieval.
- `clinical_valid_mean_r1` dung de danh gia dong gop cluster-aware false-negative mitigation.
- `cluster_mean_r1` la metric ho tro, khong nen dung mot minh vi both-normal/cluster rong co the lam so cao.

## Adapter hien co

`evaluate_shared.py` ho tro 2 adapter:

- `single_swinbert`
  Dung cho checkpoint sinh ra tu `train_single_gpu.py`
- `study_swinbert`
  Dung cho checkpoint sinh ra tu `train_proposed.py`

Neu ban cua ban co model khac, chi can them adapter moi theo mau trong `evaluate_shared.py`.

## Chay local

### 1. Tao benchmark assets

```bash
python benchmark_suite/build_benchmark_assets.py
```

Se tao:
- `benchmark_suite/data/iu_xray_test_manifest_seed42.csv`
- `benchmark_suite/data/iu_xray_test_patient_ids_seed42.json`
- `benchmark_suite/data/iu_xray_fixed_qualitative_queries_seed42.csv`
- `benchmark_suite/data/benchmark_summary.json`

### 2. Cham 1 checkpoint

Checkpoint tu `train_single_gpu.py`:

```bash
python benchmark_suite/evaluate_shared.py ^
  --adapter single_swinbert ^
  --checkpoint runs/exp_main_bs26_384/best.pt ^
  --output-dir benchmark_outputs/single_run
```

Checkpoint tu `train_proposed.py`:

```bash
python benchmark_suite/evaluate_shared.py ^
  --adapter study_swinbert ^
  --checkpoint runs/study_model/best.pt ^
  --output-dir benchmark_outputs/study_run
```

### 3. So sanh nhieu model

```bash
python benchmark_suite/compare_models.py ^
  --model baseline=single_swinbert=runs/exp_main_bs26_384/best.pt ^
  --model proposed=study_swinbert=runs/study_model/best.pt ^
  --output-dir benchmark_outputs/compare
```

Ket qua:
- `benchmark_outputs/compare/comparison.json`
- `benchmark_outputs/compare/comparison.csv`
- `benchmark_outputs/compare/comparison.md`

### 4. Tao bao cao qualitative co dinh

Luu y: script nay hien dung cho `single_swinbert`.

```bash
python benchmark_suite/generate_fixed_report.py ^
  --checkpoint runs/exp_main_bs26_384/best.pt ^
  --output-dir benchmark_outputs/fixed_report
```

Ket qua:
- `benchmark_outputs/fixed_report/report.html`
- `benchmark_outputs/fixed_report/report.json`
- `benchmark_outputs/fixed_report/report.csv`

## Chay tren Colab

### Cell 1: mount drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2: vao repo

```python
%cd /content/drive/MyDrive/finetune_IU_Xray
```

### Cell 3: chay nhanh

Checkpoint `single_swinbert`:

```python
!bash benchmark_suite/run_on_colab.sh /content/drive/MyDrive/checkpoints/best.pt single_swinbert
```

Checkpoint `study_swinbert`:

```python
!bash benchmark_suite/run_on_colab.sh /content/drive/MyDrive/checkpoints/best.pt study_swinbert
```

## Data benchmark

Benchmark nay dung test split co dinh theo:
- seed = `42`
- split patient-wise
- manifest luu san trong `benchmark_suite/data/iu_xray_test_manifest_seed42.csv`

Nghia la:
- Hai model phai duoc cham tren cung mot danh sach sample test
- Khong duoc tu y doi split khi so sanh

## Goi y de so voi LVTN

Neu ban cua ban dung model khac:
1. Pull repo nay
2. Chay `python benchmark_suite/build_benchmark_assets.py`
3. Them adapter moi trong `benchmark_suite/evaluate_shared.py`
4. Chay `evaluate_shared.py` tren checkpoint cua ban ay
5. Chay `compare_models.py` de co bang so sanh chung

## Cach bao cao trong paper

Nen bao cao:
- `Strict i2t R@1/R@5/R@10`
- `Strict t2i R@1/R@5/R@10`
- `Cluster i2t R@1/R@5/R@10`
- `Cluster t2i R@1/R@5/R@10`
- `Clinical valid i2t/t2i R@1/R@5/R@10` neu checkpoint duoc train bang `train_proposed.py`

Nen ket luan model tot hon dua tren tung muc tieu:
- Strict retrieval: dung `Strict mean R@1`.
- Thesis contribution: dung `Clinical-valid mean R@1`, kem `Strict mean R@1` de dam bao model khong bo exact matching.

`Cluster` la metric phu de chung minh contribution cua huong cluster-aware, khong phai metric duy nhat de claim chat luong.
