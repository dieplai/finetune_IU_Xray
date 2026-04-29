# 04. Paper Writing Plan

Tài liệu này đề xuất cách viết paper/luận văn chuyên nghiệp dựa trên kết quả hiện tại.

## 1. Câu Chuyện Nên Kể

Không nên viết bài theo hướng:

> Model của tôi đạt SOTA strict retrieval.

Vì strict R@1 test hiện tại chỉ khoảng `3.8462%`.

Nên viết theo hướng:

> Medical image-text contrastive learning gặp vấn đề false negatives khi các samples khác patient nhưng cùng bệnh bị coi là negatives. Phương pháp đề xuất dùng clustering-guided supervision để giảm vấn đề này, giúp mô hình truy hồi tốt hơn theo tương đồng lâm sàng trong khi vẫn giữ strict retrieval ở mức cạnh tranh.

Đây là câu chuyện đúng với code, đúng hướng thầy giao, và không overclaim.

## 2. Suggested Title

Một số tiêu đề có thể dùng:

1. **Clustering-Guided Contrastive Learning for Clinically Aware Chest X-ray Image-Report Retrieval**
2. **Reducing False Negatives in Medical Image-Text Retrieval via Pathology Cluster Guidance**
3. **A Clinical Cluster-Aware Framework for Cross-Modal Chest X-ray Retrieval**

Nếu viết tiếng Việt:

1. **Học Đối Sánh Đa Phương Thức Có Hướng Dẫn Bởi Cụm Bệnh Cho Truy Hồi Ảnh-Báo Cáo X-quang Ngực**
2. **Giảm Mẫu Âm Tính Sai Trong Truy Hồi Ảnh-Văn Bản Y Khoa Bằng Hướng Dẫn Cụm Bệnh**

## 3. Abstract Nên Có Gì

Abstract nên gồm 5 ý:

1. Bối cảnh: image-report retrieval trong y khoa.
2. Vấn đề: contrastive learning coi khác patient là negative, tạo false negatives khi cùng bệnh.
3. Phương pháp: SwinV2 image encoder, ClinicalBERT text encoder, projection heads, clustering-guided contrastive/curriculum.
4. Đánh giá: strict, cluster, clinical-valid trên IU-Xray patient-level split.
5. Kết quả: v8 đạt clinical-valid mean R@1 `59.86%`, cluster mean R@1 `69.36%`, strict mean R@1 `3.85%`.

Không nên đưa câu "state-of-the-art" vào abstract nếu chưa có baseline từ paper khác chạy cùng split.

## 4. Introduction Outline

Introduction nên đi theo logic:

1. Medical image-text retrieval giúp liên kết ảnh X-quang với báo cáo tương ứng.
2. Contrastive learning là hướng phổ biến vì học embedding chung cho ảnh và text.
3. Tuy nhiên trong y khoa, bệnh nhân khác nhau có thể có cùng bệnh/cùng biểu hiện lâm sàng.
4. Nếu coi các cặp đó là negatives, model bị phạt khi đưa chúng lại gần, tạo false-negative pressure.
5. Luận văn đề xuất clustering-guided objective để các mẫu cùng cụm bệnh không bị xử lý như negatives cứng.
6. Đánh giá không chỉ strict patient matching mà thêm clinical-valid retrieval.

Đóng góp nên viết rõ:

- Xây dựng pipeline study-level image-report retrieval cho IU-Xray.
- Thiết kế cluster-aware training objective dựa trên pathology labels.
- Đề xuất clinical-valid evaluation để đo retrieval cùng bệnh khác patient.
- Cung cấp phân tích định lượng và định tính qua retrieval report.

## 5. Method Section

Method nên chia thành các mục:

### 5.1 Data Representation

Mô tả:

- Mỗi study gồm tối đa frontal + lateral views.
- Text là report sau cleaning.
- Labels là 14 pathology labels rule-based.
- Disease vector 12 chiều bỏ `No Finding`, merge `Enlarged Cardiomediastinum` vào `Cardiomegaly`.

### 5.2 Image And Text Encoders

Mô tả:

- SwinV2-base cho image.
- Bio_ClinicalBERT cho text.
- Projection MLP đưa hai modality về 512 chiều.
- L2 normalize embedding.
- Similarity bằng dot product có learnable temperature/logit scale.

### 5.3 Multi-view Fusion

Mô tả:

- Frontal/lateral được encode riêng.
- Thêm view-type embedding.
- Attention pooling để tạo study image embedding.

### 5.4 Clustering-Guided Objective

Mô tả chính:

- Phase 1 học strict alignment bằng MultiPositiveInfoNCE.
- Phase 2 dùng IDF-weighted Jaccard disease similarity làm soft target.
- Normal cluster được xử lý riêng qua both-normal similarity.
- Clinical supervised contrastive kéo non-normal disease-overlap peers lại gần.

Nên viết rõ:

```text
Pairs from different patients but sharing pathology labels are treated as clinically related positives/soft positives rather than ordinary hard negatives.
```

### 5.5 Curriculum Training

Mô tả clinical schedule:

- `0.00` trước epoch 15.
- Ramp lên `0.12` từ epoch 15 đến 25.
- Giữ `0.12` tới epoch 60.
- Giảm còn `0.08` tới epoch 105.
- Giảm còn `0.04` tới epoch 150.

Giải thích:

- Tránh model bị kéo quá sớm về broad cluster.
- Cho strict alignment ổn định trước.
- Late decay giúp strict recovery.

## 6. Experiments Section

Cần có các bảng:

### 6.1 Dataset Statistics

| Item | Value |
|---|---:|
| Rows/images | 7,322 |
| Patient/study IDs | 3,772 |
| Frontal | 3,738 |
| Lateral | 3,584 |
| Train studies | 3,018 |
| Val studies | 377 |
| Test studies | 377 |

### 6.2 Training Configuration

| Parameter | Value |
|---|---:|
| Image encoder | SwinV2-base |
| Text encoder | Bio_ClinicalBERT |
| Image size | 384 |
| Embedding dim | 512 |
| Batch size | 32 |
| Grad accumulation | 4 |
| Effective batch | 128 |
| Epochs | 150 |
| Seed | 42 |

### 6.3 Main Results

| Model | Strict R@1 | Cluster R@1 | Clinical-valid R@1 |
|---|---:|---:|---:|
| V7 best-balanced | 3.3156 | 67.3740 | 59.8639 |
| V8 proposed | 3.8462 | 69.3634 | 59.8639 |

### 6.4 Directional Results

| Metric | Image-to-Text | Text-to-Image |
|---|---:|---:|
| Strict R@1 | 3.4483 | 4.2440 |
| Cluster R@1 | 68.1698 | 70.5570 |
| Clinical-valid R@1 | 49.6599 | 70.0680 |

### 6.5 Qualitative Bucket Counts

| Direction | Strict hit | Clinical rescue | Cluster rescue | Miss |
|---|---:|---:|---:|---:|
| Image-to-Text | 13 | 64 | 180 | 120 |
| Text-to-Image | 16 | 93 | 157 | 111 |

## 7. Figures Nên Có

1. Architecture diagram.
2. False-negative motivation diagram.
3. Training curves:
   - Strict R@1
   - Cluster R@1
   - Clinical-valid R@1
   - Loss
4. Qualitative retrieval examples.
5. Optional: UMAP/t-SNE embedding colored by pathology group.

## 8. Discussion Nên Nói Gì

Các điểm nên nhấn mạnh:

- Strict retrieval khó vì IU-Xray nhỏ và nhiều reports có nội dung rất giống nhau.
- Clinical-valid cao cho thấy model học được disease-level semantic retrieval.
- V8 không tăng clinical-valid so với v7 nhưng giữ clinical-valid cao đồng thời cải thiện strict và cluster.
- Đây là kết quả phù hợp với thesis objective hơn là pure exact-pair retrieval.

Không nên né điểm yếu. Nên viết:

> Although strict R@1 remains limited, the proposed method substantially improves clinically meaningful retrieval behavior, as shown by high clinical-valid R@1 and qualitative clinical rescue cases.

## 9. Limitations

Nên có section limitation rõ ràng:

1. Dataset nhỏ, chỉ IU-Xray.
2. Labels được tạo bằng rule-based extraction, chưa phải annotation bác sĩ.
3. Cluster metric có thể bị ảnh hưởng bởi normal/broad labels.
4. Strict R@1 còn thấp.
5. Chưa đánh giá external dataset như MIMIC-CXR.
6. Chưa có đủ ablation cho prototype/HNM vì final v8 không bật hai thành phần đó.

Viết limitation tốt sẽ làm paper đáng tin hơn, không yếu đi.

## 10. Claims Nên Dùng

Nên claim:

- Proposed method improves clinically meaningful retrieval under a fixed patient-level split.
- Clinical-valid metric better reflects the false-negative mitigation objective.
- Curriculum cluster guidance preserves strict retrieval better than applying strong clinical supervision too early.

Không nên claim:

- SOTA trên IU-Xray nếu không có benchmark chuẩn cùng split.
- Hard negative mining/prototype bank là đóng góp chính của result v8, vì final config không bật.
- Cluster R@1 cao đồng nghĩa model strict retrieval tốt.

## 11. Việc Cần Làm Trước Khi Nộp

Để luận văn chắc hơn, nên làm thêm:

1. Train baseline strict-only cùng split.
2. Train ablation without clinical schedule hoặc fixed clinical weight.
3. Generate plots từ `history.csv`.
4. Đưa qualitative report vào phụ lục.
5. Ghi rõ mọi config trong appendix.
6. Nếu dùng `best_balanced` trong tài liệu, đảm bảo local archive có file checkpoint hoặc ghi rằng nó trùng `best.pt` ở epoch 120.

## 12. One-paragraph Summary Cho Thầy

Mô hình sử dụng SwinV2 để mã hóa ảnh X-quang và Bio_ClinicalBERT để mã hóa báo cáo, sau đó chiếu cả hai về embedding 512 chiều để truy hồi hai chiều image-to-text và text-to-image. Điểm mới là quá trình huấn luyện không chỉ dùng strict patient-pair contrastive loss mà thêm hướng dẫn từ cụm bệnh: các mẫu khác bệnh nhân nhưng có cùng bệnh lý được xem là liên quan lâm sàng, giúp giảm false negatives. Trên IU-Xray patient-level split, mô hình v8 đạt clinical-valid mean R@1 59.86%, cluster mean R@1 69.36%, và strict mean R@1 3.85%. Kết quả cho thấy mô hình học tốt tương đồng lâm sàng, dù strict exact matching vẫn còn là hạn chế cần thảo luận.
