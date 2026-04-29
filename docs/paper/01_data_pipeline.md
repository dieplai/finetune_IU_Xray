# 01. Data Pipeline

Tài liệu này mô tả pipeline dữ liệu đang dùng cho paper. Các dữ kiện được đối chiếu với `data_processing/preprocess_iuxray.py`, `data/v8_clean.csv`, `data/images_384`, `train_single_gpu.py`, `train_proposed.py`, và `benchmark_suite/build_benchmark_assets.py`.

## 1. Nguồn Dữ Liệu

Dự án dùng bộ dữ liệu IU-Xray/RadDAR-style gồm:

- `indiana_projections.csv`: thông tin ảnh và projection.
- `indiana_reports.csv`: findings, impression, uid.
- Ảnh X-quang đã chuẩn hóa ở dạng PNG/DICOM-derived image.

Trong pipeline hiện tại:

- `uid` được chuyển thành `patient_id`.
- `filename` được chuyển thành `image_id`.
- `findings` và `impression` được nối thành một báo cáo văn bản.
- Một `patient_id` tương ứng với một study/report identity trong bài toán strict retrieval.

## 2. Text Cleaning

Script chính: `data_processing/preprocess_iuxray.py`.

Các bước xử lý text:

1. Điền rỗng cho `findings` và `impression` nếu bị thiếu.
2. Ghép `findings + impression` thành `raw_caption`.
3. Chuẩn hóa chữ thường.
4. Loại bỏ chuỗi placeholder như nhiều ký tự `x`.
5. Loại bỏ thông tin tuổi/giới tính đơn giản.
6. Loại bỏ nhãn section như `findings`, `impression`, `clinical information`, `comparison`.
7. Loại bỏ numbering đầu dòng như `1.` hoặc `1)`.
8. Loại bỏ một số ký tự đặc biệt.
9. Gộp khoảng trắng.
10. Lọc report có `word_count < 10`.

Cột văn bản cuối dùng để train là:

```text
org_caption
```

Thống kê local từ `data/v8_clean.csv`:

- Tổng rows ảnh-báo cáo: `7,322`
- Unique `patient_id`: `3,772`
- Unique images: `7,322`
- Unique reports: `3,018`
- Word count: min `10`, mean `37.27`, max `223`

## 3. Image Processing

Script tham chiếu: `data_processing/preprocess_iuxray.py`.

Logic xử lý ảnh:

1. Mở ảnh dạng grayscale.
2. Convert sang RGB.
3. Pad ảnh thành hình vuông bằng nền đen để tránh méo hình.
4. Lưu thành PNG.
5. Final local training assets nằm trong:

```text
data/images_384
```

Kiểm tra local hiện tại:

- Số file ảnh PNG: `7,322`
- Tất cả ảnh trong `data/images_384` có size: `384 x 384`
- Training transform không resize lại ảnh; ảnh đã ở kích thước native của SwinV2.

Lý do dùng padding thay vì crop:

- X-quang ngực có thông tin lâm sàng nằm ở toàn bộ phổi, vùng rìa, xương sườn, tim, màng phổi.
- Crop có thể làm mất vùng bệnh nhỏ hoặc dấu hiệu ở biên ảnh.
- Padding giữ toàn bộ trường nhìn, sau đó scale về 384x384 giúp tránh distortion hình học.

## 4. Projection Distribution

Từ `data/v8_clean.csv`:

| Projection | Count |
|---|---:|
| Frontal | 3,738 |
| Lateral | 3,584 |

Trong training study-level, các ảnh cùng `patient_id` được gom lại. Mỗi sample có thể dùng tối đa:

- 1 frontal view
- 1 lateral view

Nếu thiếu một view thì view còn lại vẫn được dùng; slot còn lại được pad bằng tensor zero và `view_mask=False`.

## 5. Pathology Label Extraction

Script chính: `data_processing/preprocess_iuxray.py`.

Các cột nhãn theo thứ tự `PATH_COLS`:

```text
No Finding
Enlarged Cardiomediastinum
Cardiomegaly
Lung Lesion
Lung Opacity
Edema
Consolidation
Pneumonia
Atelectasis
Pneumothorax
Pleural Effusion
Pleural Other
Fracture
Support Devices
```

Nhãn được tạo bằng rule-based keyword matching có xét negation đơn giản. Ví dụ các cụm phủ định:

```text
no, without, absence of, there is no, there are no, free of, rule out, negative for
```

Nếu một câu có keyword bệnh nhưng nằm trong câu có negation thì không gán nhãn đó. Nếu có bất kỳ bệnh nào khác `No Finding`, nhãn `No Finding` được reset về 0.

Phân bố nhãn local:

| Label | Count | Percent |
|---|---:|---:|
| No Finding | 4,238 | 57.88% |
| Enlarged Cardiomediastinum | 17 | 0.23% |
| Cardiomegaly | 659 | 9.00% |
| Lung Lesion | 1,014 | 13.85% |
| Lung Opacity | 942 | 12.87% |
| Edema | 185 | 2.53% |
| Consolidation | 402 | 5.49% |
| Pneumonia | 164 | 2.24% |
| Atelectasis | 732 | 10.00% |
| Pneumothorax | 149 | 2.03% |
| Pleural Effusion | 415 | 5.67% |
| Pleural Other | 65 | 0.89% |
| Fracture | 133 | 1.82% |
| Support Devices | 224 | 3.06% |

Ghi chú quan trọng:

- Có `171` rows không có nhãn nào bằng 1.
- Khi train cluster, `No Finding` không nằm trong disease vector chính; normal được xác định bằng tổng disease vector bằng 0.

## 6. Disease Vector Cho Clustering

Code tham chiếu: `train_single_gpu.py::labels_to_disease_vecs`.

Từ 14 nhãn `PATH_COLS`, model tạo disease vector 12 chiều:

- Loại bỏ `No Finding`.
- Gộp `Enlarged Cardiomediastinum` vào `Cardiomegaly` bằng phép max.
- Giữ các bệnh còn lại.

Kết quả là `CHEXPERT_COLS`:

```text
Cardiomegaly
Lung Lesion
Lung Opacity
Edema
Consolidation
Pneumonia
Atelectasis
Pneumothorax
Pleural Effusion
Pleural Other
Fracture
Support Devices
```

Disease vector này dùng cho:

- Cluster mask.
- Clinical cluster mask.
- IDF-weighted Jaccard soft labels.
- Auxiliary pathology heads.

## 7. Train/Val/Test Split

Code tham chiếu:

- `train_single_gpu.py::patient_split`
- `benchmark_suite/build_benchmark_assets.py`

Split được thực hiện theo unique `patient_id`, không split theo row ảnh. Điều này tránh leakage khi cùng một study có frontal/lateral image.

Cấu hình:

```text
seed = 42
train = 80%
val = 10%
test = 10%
```

Thống kê local:

| Split | Rows | Patient/Study IDs | Images |
|---|---:|---:|---:|
| Train | 5,865 | 3,018 | 5,865 |
| Val | 733 | 377 | 733 |
| Test | 724 | 377 | 724 |

Với study-level dataset, số sample thực tế theo split là số `patient_id`:

| Split | Study-level samples |
|---|---:|
| Train | 3,018 |
| Val | 377 |
| Test | 377 |

## 8. Benchmark Assets

Thư mục:

```text
benchmark_suite/data
```

Các file chính:

- `iu_xray_test_manifest_seed42.csv`
- `iu_xray_test_patient_ids_seed42.json`
- `iu_xray_fixed_qualitative_queries_seed42.csv`
- `benchmark_summary.json`

`benchmark_summary.json` hiện ghi:

```text
test_rows = 724
test_patients = 377
fixed_queries = 20
```

Các assets này giúp đánh giá nhiều model trên cùng test split, tránh việc mỗi run dùng một split khác nhau.

## 9. Rủi Ro Cần Nói Trong Paper

Các nhãn bệnh hiện tại là rule-based, không phải label được bác sĩ xác nhận. Điều này phù hợp cho luận văn nếu trình bày là weak supervision/pathology proxy, nhưng không nên nói như ground-truth clinical diagnosis tuyệt đối.

Cluster metric có thể cao do nhóm normal hoặc nhãn bệnh rộng. Vì vậy paper nên dùng:

- `strict R@K` để đo exact pair retrieval.
- `clinical-valid R@K` để đo đúng mục tiêu false-negative mitigation.
- `cluster R@K` chỉ là metric hỗ trợ.
