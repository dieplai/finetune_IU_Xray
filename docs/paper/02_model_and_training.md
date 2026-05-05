# 02. Model And Training

Tài liệu này mô tả implementation v8 đang dùng cho kết quả chính. Các file tham chiếu:

- `train_proposed.py`
- `train_single_gpu.py`
- `v8_curriculum_a100_150ep_seed42_bs32_v1/config.json`

## 1. Mục Tiêu Kiến Trúc

Mỗi sample study-level gồm:

- Image side: tối đa 2 ảnh của cùng study/patient, thường là frontal và lateral.
- Text side: một báo cáo `org_caption`.
- Label side: vector bệnh 14 chiều từ CSV, sau đó chuyển thành disease vector 12 chiều.

Model học ánh xạ ảnh và text vào cùng embedding space để tính similarity.

## 2. Image Encoder

Model thị giác:

```text
microsoft/swinv2-base-patch4-window12to24-192to384-22kto1k-ft
```

Đặc điểm:

- Input ảnh `384 x 384`.
- Dùng `pooler_output`.
- Output dimension từ log runtime: `1024`.
- Bật gradient checkpointing để giảm VRAM.

## 3. Text Encoder

Model văn bản:

```text
emilyalsentzer/Bio_ClinicalBERT
```

Đặc điểm:

- Dùng `[CLS]` embedding từ `last_hidden_state[:, 0, :]`.
- Hidden dimension: `768`.
- Max token length: `128`.
- Bật gradient checkpointing.

Ghi chú: tên model là Bio_ClinicalBERT, phù hợp với hướng dùng pretrained medical language model. Khi load `BertModel`, các head pretraining như MLM/NSP có thể báo unexpected keys; điều này bình thường vì chỉ dùng encoder.

## 4. Projection Head

Cả image và text đều đi qua MLP projection:

```text
input_dim -> 1024 hidden -> 512 embedding
```

Thông số:

- `EMBED_DIM = 512`
- `PROJ_HID_DIM = 1024`
- Dropout theo base config: `0.2`
- Output được L2-normalize.

Similarity:

```text
logits = exp(logit_scale) * image_embedding @ text_embedding.T
```

`logit_scale` được clamp trong khoảng tương ứng `exp(logit_scale)` từ `1` tới `100`.

## 5. Multi-view Study Encoder

Code tham chiếu: `StudyIUXrayDataset` và `StudyMedicalSwinBERT.encode_study_images`.

Mỗi `patient_id` được gom thành một study sample:

1. Tách image IDs theo projection: frontal, lateral, other.
2. Khi train, nếu có nhiều ảnh cùng loại thì random chọn một ảnh.
3. Khi eval, chọn ảnh đầu tiên theo thứ tự sorted để deterministic.
4. Dùng tối đa `MAX_VIEWS = 2`.
5. Thiếu view thì pad bằng zero tensor và `view_mask=False`.

Model encode từng valid view bằng SwinV2, projection về 512 chiều, thêm `view_type_embed`, rồi dùng attention pooling:

```text
view_attn(embedding + view_type_embedding) -> attention weight
study_embedding = weighted sum of view embeddings
```

Lý do dùng attention pooling:

- Frontal và lateral không luôn có mức thông tin như nhau.
- Một số study chỉ có một view.
- Attention pooling cho phép model tự học view nào quan trọng hơn.

## 6. Auxiliary Heads

V8 có các head phụ:

- `img_pathology_head`: dự đoán 12 bệnh từ image embedding.
- `img_normal_head`: dự đoán normal từ image embedding.
- `txt_pathology_head`: dự đoán 12 bệnh từ text embedding.
- `txt_normal_head`: dự đoán normal từ text embedding.

Auxiliary loss gồm:

- BCE cho image pathology.
- BCE cho image normal.
- BCE cho text pathology.
- BCE cho text normal.
- Consistency MSE giữa sigmoid output của image/text pathology heads.
- Consistency MSE giữa sigmoid output của image/text normal heads.

Trọng số:

```text
aux_weight = 0.02
```

Vai trò: ép embedding giữ thông tin bệnh lý, nhưng không để auxiliary task lấn át contrastive retrieval.

## 7. Base Loss: MultiPositive And Cluster-Aware

V8 dùng `StudyLoss`, bên trong gọi `CombinedLoss` từ `train_single_gpu.py`.

### 7.1 Phase 1: MultiPositiveInfoNCE

Áp dụng trước `cluster_start`.

Trong v8:

```text
cluster_start = 12
```

Trước epoch 12, loss chính là MultiPositiveInfoNCE. Positive mask được định nghĩa bằng cùng `patient_id`.

Trong study-level run hiện tại, mỗi batch có một sample trên mỗi patient/study, nên positive gần như là đường chéo. Tuy nhiên code vẫn hỗ trợ multi-positive để tránh lỗi nếu có nhiều row cùng patient.

Mục tiêu phase này:

- Cho projection heads và encoders học exact image-report alignment trước.
- Tránh đưa cluster supervision quá sớm khi embedding còn nhiễu.

### 7.2 Phase 2: HierarchicalClusterLoss

Từ epoch 12 trở đi, loss chính chuyển sang cluster-aware soft target.

Soft target:

```text
target = (1 - alpha) * hard_same_patient + alpha * disease_similarity_soft_target
```

Trong v8:

```text
alpha = 0.35
temperature = 1.0
```

Disease similarity dùng IDF-weighted Jaccard:

```text
J_w(A,B) = sum_k w_k * (A_k AND B_k) / sum_k w_k * (A_k OR B_k)
```

Rare diseases có trọng số cao hơn common diseases nhờ IDF weights.

### 7.3 Healthy / Normal Cluster

Nếu hai samples đều không có disease dimension nào, code xem chúng là both-normal và thêm similarity `0.4` trong cluster loss.

Lý do:

- Disease vector của normal là all-zero nên Jaccard không có nghĩa.
- Nhưng hai normal reports vẫn có tương đồng lâm sàng.

Rủi ro:

- Cluster metric có thể bị tăng bởi normal cases.
- Vì vậy paper không nên dùng cluster metric làm claim duy nhất.

## 8. Clinical Supervised Contrastive Loss

Đây là phần sát nhất với hướng dẫn của thầy: nếu hai samples có cùng bệnh/cụm bệnh thì không nên coi nhau là âm tính sai.

Code tham chiếu:

- `build_clinical_cluster_mask`
- `ClinicalSupervisedContrastiveLoss`

Clinical mask:

```text
clinical_mask(i,j) = disease_overlap(i,j) AND non_normal(i) AND non_normal(j)
```

Điểm quan trọng:

- Cặp both-normal bị loại khỏi clinical mask.
- Diagonal exact match bị loại khỏi peer mask vì strict contrastive đã xử lý.
- Clinical positives là các bệnh nhân/studies khác nhưng có disease overlap.

Loss dùng supervised contrastive dạng cross-entropy mềm trên logits image-text theo cả hai chiều:

- Image to Text
- Text to Image

Vai trò:

- Kéo các cặp cùng bệnh lại gần nhau trong embedding space.
- Giảm áp lực coi cùng bệnh khác patient là negative sai.
- Phù hợp trực tiếp với đề tài clustering-guided false-negative mitigation.

## 9. Clinical Weight Curriculum

V8 dùng curriculum thay vì bật clinical loss mạnh ngay từ đầu.

Cấu hình v8:

```text
clinical_start = 15
clinical_ramp_end = 25
clinical_ramp_start_weight = 0.03
clinical_peak_weight = 0.12
clinical_hold_end = 60
clinical_mid_weight = 0.08
clinical_decay_end = 105
clinical_final_weight = 0.04
```

Lịch weight:

| Epoch range | Clinical weight |
|---|---:|
| `< 15` | 0.00 |
| `15 - 25` | ramp 0.03 -> 0.12 |
| `26 - 60` | 0.12 |
| `61 - 105` | 0.08 |
| `106 - 150` | 0.04 |

Lý do thiết kế:

- Early epochs ưu tiên strict image-report alignment.
- Middle epochs tăng clinical same-disease supervision.
- Late epochs giảm clinical pressure để strict retrieval hồi lại, tránh drift về cluster quá rộng.

Kết quả thực tế xác nhận ý tưởng này:

- Strict tăng mạnh lại ở epoch 70 và đạt best validation ở epoch 120.
- Clinical-valid peak validation ở epoch 90.

## 10. Code Final Sau Khi Clean

Code paper-facing hiện chỉ giữ các thành phần đã được dùng trong run chính:

| Component | Có trong code final | Dùng trong V8 final |
|---|---:|---:|
| Study-level multi-view attention fusion | Có | Có |
| Multi-positive InfoNCE warmup | Có | Có |
| IDF-weighted Jaccard soft target | Có | Có |
| Healthy Cluster | Có | Có |
| Clinical supervised contrastive curriculum | Có | Có |
| Auxiliary pathology/normal heads | Có | Có |

Các nhánh thử nghiệm không dùng trong kết quả chính đã được loại khỏi code clean để tránh nhầm lẫn khi viết paper.

## 11. Training Setup V8 Chính

Run chính:

```text
v8_curriculum_a100_150ep_seed42_bs32_v1
```

Config:

| Parameter | Value |
|---|---:|
| epochs | 150 |
| batch size | 32 |
| gradient accumulation | 4 |
| effective batch | 128 |
| freeze epoch | 5 |
| eval every | 5 |
| seed | 42 |
| num workers | 4 |
| image encoder LR | 1e-5 |
| text encoder LR | 1e-4 |
| head LR | 2e-4 |
| weight decay | 0.01 |
| max grad norm | 1.0 |

Training details:

- Epoch 1-5: freeze SwinV2 + ClinicalBERT, train heads only.
- Epoch 6 onward: unfreeze all backbones.
- Optimizer: AdamW.
- Scheduler after unfreeze: Linear warmup then cosine annealing.
- Mixed precision: `torch.amp.autocast("cuda")` + `GradScaler`.
- Gradient accumulation: loss divided by `grad_accum`.
- Gradient clipping: norm 1.0.

## 12. Checkpoint Strategy

V8 lưu ba loại checkpoint:

1. `best.pt`: best validation strict R@1.
2. `best_balanced.pt`: best `strict + 0.1 * clinical_valid`, chỉ khi strict >= 4.0.
3. `best_clinical_valid.pt`: best clinical-valid, chỉ khi strict >= 3.5.

Trong run v8 hiện tại:

- `best.pt` và `best_balanced` cùng chọn epoch 120.
- `best_clinical_valid.pt` chọn epoch 90.
- Local downloaded folder hiện thiếu file `best_balanced.pt`, nhưng report và JSON cho thấy nó trùng kết quả với `best.pt`. Nếu cần reproducibility sạch, có thể copy `best.pt` thành `best_balanced.pt`.
