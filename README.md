# PPE Detection & Edge Optimization Pipeline (YOLOv8)

Dự án phát hiện Trang thiết bị Bảo hộ Lao động (PPE - Personal Protective Equipment) và tối ưu hóa mô hình học sâu (Model Compression) chạy trên các thiết bị biên (Edge Devices). 

Quy trình tối ưu hóa bao gồm 4 giai đoạn chính được thực thi qua 4 notebook nối tiếp: **Sparsity Training (Huấn luyện thưa)** $\rightarrow$ **Structured Pruning (Cắt tỉa cấu trúc)** $\rightarrow$ **Finetuning (Tinh chỉnh phục hồi)** $\rightarrow$ **Quantization-Aware Training (INT8 QAT & ONNX/TensorRT Export)**.

---

## 📌 Tổng quan Quy trình Tối ưu hóa (4 Notebooks)

```
[Baseline Model / Pretrained]
       │
       ▼
[01_sparsity_training.ipynb] ──► Huấn luyện L1 Regularization lên tham số γ (BatchNorm)
       │
       ▼ (runs/train-sparsity/weights/last.pt)
[02_structured_pruning.ipynb] ──► Cắt tỉa kênh thưa (Channel Pruning 30%)
       │
       ▼ (weights/pruned_r0.30.pt)
[03_finetuning.ipynb]         ──► Huấn luyện phục hồi độ chính xác (mAP Recovery)
       │
       ▼ (runs/train-finetune/weights/best.pt)
[04_quantize.ipynb]           ──► INT8 QAT (NVIDIA PyTorch Quantization) & ONNX Export
       │
       ▼
[ONNX TensorRT Engine Deployment / Streamlit Demo]
```

---

## 🚀 Chi tiết các Notebooks trong Pipeline

### 1. `01_sparsity_training.ipynb` — Giai đoạn 1: Sparsity Training (Huấn luyện thưa)
- **Mục tiêu**: Áp dụng L1 Regularization lên hệ số tỷ lệ $\gamma$ của các lớp Batch Normalization (BN) trong mạng YOLOv8 để thúc đẩy các hệ số $\gamma$ tiệm cận về 0, chuẩn bị cho việc cắt tỉa kênh.
- **Các bước chính**:
  1. Cài đặt môi trường & liên kết repository Ultralytics tùy chỉnh.
  2. Thực thi lệnh huấn luyện thưa với tham số $s = 0.015$:
     ```bash
     python scripts/train.py \
         --mode sparsity \
         --weights checkpoint/best_baseline.pt \
         --data data.yaml \
         --batch 16 \
         --epochs 80 \
         --device 0
     ```
  3. Trực quan hóa phân bố trọng số $\gamma$ của BN trước và sau khi huấn luyện thưa qua script `scripts/vis_bn_weight.py` (xuất biểu đồ Histogram & Density Heatmap lưu tại `assets/`).
  4. Đánh giá độ chính xác (mAP) và độ trễ suy luận bằng `scripts/val.py`.
  5. Tự động ghi nhận parameters, weights và metrics lên DagsHub / MLflow (`YOLOv8_Sparsity_Pruning`).
- **Đầu ra**: Trọng số mô hình thưa (`runs/train-sparsity3/weights/last.pt`).

---

### 2. `02_structured_pruning.ipynb` — Giai đoạn 2: Structured Pruning (Cắt tỉa cấu trúc)
- **Mục tiêu**: Loại bỏ các kênh/filters có giá trị $\gamma$ nhỏ dựa trên ngưỡng đã học, giúp giảm trực tiếp số lượng tham số, dung lượng mô hình và lượng tính toán (FLOPs).
- **Các bước chính**:
  1. Cài đặt môi trường và load checkpoint thưa từ Giai đoạn 1.
  2. Thực thi script cắt tỉa kênh với tỷ lệ prune 30% (`--prune-ratio 0.30`):
     ```bash
     python scripts/prune.py \
         --weights runs/train-sparsity3/weights/last.pt \
         --prune-ratio 0.30 \
         --model-size m \
         --cfg ultralytics/cfg/models/v8/yolov8.yaml \
         --save-dir weights
     ```
  3. Kiểm tra cấu trúc mô hình sau cắt tỉa (`maskbndict`, các lớp custom `C2fPruned`, `SPPFPruned`, `DetectPruned`).
  4. Đánh giá mAP zero-shot ngay sau khi cắt tỉa để đo lường mức độ sụt giảm độ chính xác ban đầu.
  5. Log checkpoint đã cắt tỉa lên DagsHub / MLflow (`YOLOv8_Structured_Pruning`).
- **Đầu ra**: Checkpoint cắt tỉa kèm mask kênh (`weights/pruned_r0.30.pt`).

---

### 3. `03_finetuning.ipynb` — Giai đoạn 3: Finetuning (Tinh chỉnh phục hồi)
- **Mục tiêu**: Huấn luyện lại kiến trúc mô hình đã được cắt tỉa để phục hồi chỉ số độ chính xác (mAP50 và mAP50-95) bị sụt giảm ở bước trước.
- **Các bước chính**:
  1. Load trọng số từ `weights/pruned_r0.30.pt`.
  2. Chạy quá trình finetune 100 epochs với cấu hình tối ưu:
     ```bash
     python scripts/train.py \
         --mode finetune \
         --weights weights/pruned_r0.30.pt \
         --data data.yaml \
         --epochs 100 \
         --batch 32 \
         --device 0
     ```
  3. Đánh giá chất lượng mô hình sau finetune trên tập validation (`scripts/val.py`).
  4. Upload toàn bộ log huấn luyện, biểu đồ metrics, confusion matrix và weight lên DagsHub / MLflow (`YOLOv8_Finetuning`).
- **Đầu ra**: Mô hình cắt tỉa đạt độ chính xác cao (`runs/train-finetune/weights/best.pt`).

---

### 4. `04_quantize.ipynb` — Giai đoạn 4: Quantization-Aware Training (INT8 QAT) & ONNX Export
- **Mục tiêu**: Chuyển đổi độ chính xác của mô hình từ FP32 sang INT8 bằng phương pháp Quantization-Aware Training (QAT) và xuất ra định dạng ONNX tối ưu cho NVIDIA TensorRT trên thiết bị biên.
- **Các bước chính**:
  1. Cài đặt thư viện NVIDIA PyTorch Quantization (`pytorch-quantization`):
     ```bash
     pip install "setuptools<70.0.0" wheel
     pip install --no-build-isolation .
     ```
  2. Thực thi huấn luyện QAT (chèn các nút Giả lượng tử hóa - Fake Quantization Nodes):
     ```bash
     python scripts/qat.py \
         --pruned-checkpoint weights/pruned_r0.30.pt \
         --pretrained-weight runs/train-finetune/weights/best.pt \
         --data-config data.yaml \
         --epochs 10 \
         --batch 32 \
         --workers 4 \
         --device 0
     ```
  3. Xuất mô hình INT8 QAT sang định dạng ONNX TensorRT:
     ```bash
     python scripts/export_onnx.py \
         --pruned-checkpoint weights/pruned_r0.30.pt \
         --weight runs/qat-pruned/train/weights/best.pt
     ```
  4. Ghi nhận mô hình xuất ra (`best.tensorrt.onnx`) và thông số QAT lên DagsHub / MLflow (`YOLOv8_Quantization`).
- **Đầu ra**: Mô hình ONNX INT8 đã sẵn sàng build TensorRT Engine (`runs/qat-pruned/train/weights/best.tensorrt.onnx`).

---

## 🛠️ Cài đặt Môi trường & Hướng dẫn Sử dụng

### 1. Yêu cầu hệ thống
- Python $\ge$ 3.10
- PyTorch với hỗ trợ CUDA (`torch`, `torchvision`)
- MLflow & DagsHub account (để quản lý thí nghiệm)

### 2. Cài đặt thư viện
```bash
# Tạo môi trường ảo với uv
uv venv
.venv\Scripts\activate

# Cài đặt các gói phụ thuộc
uv pip install -r requirements.txt
```

### 3. Chạy ứng dụng DEMO (Streamlit Interface)
Tải checkpoint tương ứng đưa vào thư mục `checkpoint/` (Cập nhật đường dẫn trong `config.py`), sau đó khởi chạy ứng dụng web:
```bash
streamlit run demo.py
```

---

## 📋 Danh sách Lớp Vi phạm & Bảo hộ (Classes)

Tập dữ liệu bao gồm **9 lớp** được định nghĩa trong `data.yaml`:
- **Trang bị bảo hộ**: `boots`, `gloves`, `helmet`, `vest`, `person`
- **Cảnh báo vi phạm (Violation Alert)**: `no_boots`, `no_gloves`, `no_helmet`, `no_vest`

---

## 📁 Cấu trúc Thư mục Dự án

```
ppe_v2/
├── 01_sparsity_training.ipynb  # Notebook Giai đoạn 1: Sparsity Training
├── 02_structured_pruning.ipynb # Notebook Giai đoạn 2: Structured Pruning
├── 03_finetuning.ipynb        # Notebook Giai đoạn 3: Finetuning
├── 04_quantize.ipynb          # Notebook Giai đoạn 4: INT8 QAT & Export ONNX
├── scripts/
│   ├── train.py               # Script huấn luyện (Sparsity & Finetune)
│   ├── prune.py               # Script cắt tỉa kênh cấu trúc
│   ├── val.py                 # Script đánh giá mAP & Latency
│   ├── vis_bn_weight.py       # Script trực quan hóa phân bố γ BatchNorm
│   ├── qat.py                 # Script huấn luyện Quantization-Aware Training
│   └── export_onnx.py         # Script xuất mô hình QAT sang ONNX TensorRT
├── config.py                  # Cấu hình đường dẫn model, video & lớp vi phạm cho Demo
├── data.yaml                  # Cấu hình tập dữ liệu & nhãn PPE
├── demo.py                    # Giao diện ứng dụng Streamlit phát hiện vi phạm PPE
├── requirements.txt           # Danh sách phụ thuộc thư viện Python
└── README.md                  # Hướng dẫn dự án
```
