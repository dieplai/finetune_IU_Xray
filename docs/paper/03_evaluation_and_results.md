# 03. Evaluation And Results

Tài liệu này mô tả cách đánh giá và kết quả thực nghiệm đang có. Các file tham chiếu:

- `train_proposed.py`
- `tools/generate_study_retrieval_report.py`
- `benchmark_suite/evaluate_shared.py`
- `v8_curriculum_a100_150ep_seed42_bs32_v1/*`
- `v7_balanced_a100_120ep_seed42-20260426T084152Z-3-001/*`

## 1. Evaluation Level

Đánh giá chính là study-level:

- Mỗi `patient_id`/study là một sample.
- Image side là embedding của frontal/lateral views sau attention pooling.
- Text side là embedding của report.
- Similarity là dot product giữa embedding đã normalize.

Test split cố định:

```text
seed = 42
test patients/studies = 377
test rows/images = 724
```

## 2. Strict Retrieval

Strict retrieval đo exact identity:

```text
gt_strict[i,j] = patient_id[i] == patient_id[j]
```

Vì study-level dataset có một sample cho mỗi `patient_id`, ground-truth strict gần như là đường chéo.

Metrics:

- `strict_i2t R@1/R@5/R@10`
- `strict_t2i R@1/R@5/R@10`
- `strict_mean_r1 = (i2t R@1 + t2i R@1) / 2`
- `MRR` strict

Ý nghĩa:

- Đây là metric khó nhất.
- Nếu strict R@1 thấp, model chưa giỏi exact image-report matching.
- Nhưng strict không đo được việc khác patient cùng bệnh có được retrieval hợp lý hay không.

## 3. Cluster Retrieval

Cluster mask trong v8:

```text
gt_cluster = disease_overlap OR both_normal
```

Trong đó:

- `disease_overlap`: hai samples chia sẻ ít nhất một disease dimension.
- `both_normal`: cả hai có disease vector all-zero.

Ý nghĩa:

- Đo retrieval theo cụm bệnh rộng.
- Phù hợp để quan sát mô hình có học semantic/clinical grouping không.

Rủi ro:

- `both_normal` có thể làm cluster score cao.
- Cluster metric không nên là claim chính nếu paper muốn chặt chẽ.

## 4. Clinical-valid Retrieval

Clinical mask trong v8:

```text
gt_clinical = disease_overlap AND both_non_normal
```

Điểm khác cluster:

- Loại bỏ both-normal.
- Chỉ tính các query non-normal có ít nhất một positive cùng bệnh.
- Tránh việc normal cases làm metric bị inflated.

Đây là metric quan trọng nhất cho hướng thầy giao vì nó đo:

> Nếu ảnh A và báo cáo B cùng nhóm bệnh, model có kéo chúng lại gần nhau không, dù khác patient?

Metrics:

- `clinical_valid_i2t R@1/R@5/R@10`
- `clinical_valid_t2i R@1/R@5/R@10`
- `clinical_valid_mean_r1`
- `clinical_valid_counts`

Trong test v8:

```text
clinical_valid_counts:
  i2t = 147
  t2i = 147
```

Nghĩa là chỉ có 147 queries mỗi chiều có positive clinical-valid phù hợp để tính metric.

## 5. Clinical-all Retrieval

Clinical-all dùng clinical mask nhưng tính trên tất cả queries, kể cả queries không có clinical positive.

Điểm yếu:

- Nếu query không có positive trong gt mask, recall sẽ khó diễn giải.
- Mean thấp hơn clinical-valid.

Vì vậy clinical-all nên là metric phụ. Metric chính nên là clinical-valid.

## 6. Qualitative Report

Script:

```text
tools/generate_study_retrieval_report.py
```

Output:

- `study_retrieval_report.html`
- `study_retrieval_report.csv`
- `study_retrieval_report.json`

Các bucket top-1:

| Bucket | Ý nghĩa |
|---|---|
| strict_hit | Top-1 đúng cùng patient/study |
| clinical_rescue | Top-1 khác patient nhưng đúng clinical non-normal overlap |
| cluster_rescue | Top-1 đúng cluster rộng, gồm cả normal/cluster overlap |
| miss | Top-1 không đúng strict/clinical/cluster |

Report lấy các case định tính theo cả hai chiều:

- Image to Text
- Text to Image

Mỗi chiều có 20 cases, gồm 5 cases cho mỗi bucket.

## 7. V8 Main Results

Run:

```text
v8_curriculum_a100_150ep_seed42_bs32_v1
```

Checkpoint chính:

```text
best.pt
epoch = 120
```

Validation summary:

| Metric | Best epoch | Value |
|---|---:|---:|
| Best strict R@1 | 120 | 4.9072 |
| Best cluster R@1 | 40 | 65.2520 |
| Best clinical-valid R@1 | 90 | 52.1127 |
| Final strict R@1 | 150 | 4.5093 |
| Final clinical-valid R@1 | 150 | 47.8873 |

Test results for `best.pt` / epoch 120:

| Metric | i2t | t2i | Mean |
|---|---:|---:|---:|
| Strict R@1 | 3.4483 | 4.2440 | 3.8462 |
| Strict R@5 | 10.0796 | 12.2016 | 11.1406 |
| Strict R@10 | 17.7719 | 18.8329 | 18.3024 |
| Cluster R@1 | 68.1698 | 70.5570 | 69.3634 |
| Cluster R@5 | 79.5756 | 97.0822 | 88.3289 |
| Cluster R@10 | 85.1459 | 99.2042 | 92.1751 |
| Clinical-all R@1 | 19.3634 | 27.3210 | 23.3422 |
| Clinical-valid R@1 | 49.6599 | 70.0680 | 59.8639 |
| Clinical-valid R@5 | 60.5442 | 95.2381 | 77.8912 |
| Clinical-valid R@10 | 69.3878 | 97.9592 | 83.6735 |

MRR strict test:

```text
9.0104
```

## 8. V8 Qualitative Bucket Counts

For `report_best` / `report_best_balanced`:

Image to Text, total 377 test studies:

| Bucket | Count |
|---|---:|
| strict_hit | 13 |
| clinical_rescue | 64 |
| cluster_rescue | 180 |
| miss | 120 |

Text to Image, total 377 test studies:

| Bucket | Count |
|---|---:|
| strict_hit | 16 |
| clinical_rescue | 93 |
| cluster_rescue | 157 |
| miss | 111 |

Interpretation:

- Strict exact top-1 hit còn ít.
- Clinical rescue nhiều, đặc biệt text-to-image.
- Điều này ủng hộ luận điểm clustering-guided giúp tìm clinically related samples thay vì chỉ exact patient.

## 9. Comparison Với V7 A100

V7 A100 report chính dùng `best_balanced.pt`.

| Model | Strict mean R@1 test | Cluster mean R@1 test | Clinical-valid mean R@1 test | Clinical-all mean R@1 test |
|---|---:|---:|---:|---:|
| V7 best-balanced | 3.3156 | 67.3740 | 59.8639 | 23.3422 |
| V8 epoch 120 | 3.8462 | 69.3634 | 59.8639 | 23.3422 |

V8 cải thiện:

- Strict mean R@1 test: `+0.5305` absolute.
- Cluster mean R@1 test: `+1.9894` absolute.
- Clinical-valid giữ mức cao ngang v7.

Điểm cần nói thật:

- V8 không cải thiện clinical-valid mean test so với v7 best-balanced; nó giữ được clinical-valid trong khi tăng strict/cluster.
- Strict vẫn thấp nếu so với mong muốn model exact retrieval mạnh.

## 10. Nên Chọn Checkpoint Nào Để Báo Cáo

Checkpoint chính cho paper:

```text
best.pt / epoch 120
```

Lý do:

- Test strict tốt hơn `best_clinical_valid.pt`.
- Clinical-valid test vẫn rất cao.
- Balanced checkpoint trùng kết quả với best strict trong run hiện tại.

Không nên dùng `best_clinical_valid.pt` làm chính:

| Checkpoint | Test strict mean R@1 | Test cluster mean R@1 | Test clinical-valid mean R@1 |
|---|---:|---:|---:|
| best.pt / ep120 | 3.8462 | 69.3634 | 59.8639 |
| best_clinical_valid.pt / ep90 | 2.3873 | 65.7825 | 59.5238 |

`best_clinical_valid.pt` giữ clinical-valid cao nhưng strict tụt rõ, không phù hợp nếu thầy quan tâm R@1 strict.

## 11. Đánh Giá Chân Thật

Kết quả hiện tại đủ để viết luận văn thạc sĩ nếu định vị đúng:

- Đóng góp là clinical-aware / clustering-guided retrieval.
- Không phải SOTA strict retrieval.
- Paper cần báo cáo cả strict và clinical-valid để tránh claim lệch.

Mức độ hiện tại:

- Strong for thesis: có pipeline, model, loss, eval, report, kết quả thực nghiệm thật.
- Moderate for publication: cần thêm baseline/ablation cùng split để claim thuyết phục hơn.

## 12. Thực Nghiệm Nên Chạy Thêm Nếu Còn Thời Gian

Ưu tiên cao nhất:

1. Baseline strict-only cùng split.
2. V8 no clinical schedule hoặc fixed clinical weight.
3. Ablation không dùng clinical supervised contrastive.
4. Plot learning curves từ `history.csv`.

Mục tiêu không phải train model lớn hơn, mà là chứng minh từng thành phần có đóng góp.
