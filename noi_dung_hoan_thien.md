# BÁO CÁO KẾT QUẢ NGHIÊN CỨU
## PHẦN: TỐI ƯU HÓA MÔ HÌNH VỚI LƯỢNG TỬ HÓA (QUANTIZATION)

### 1. Thiết lập cấu hình và quá trình chuyển đổi mô hình sang định dạng INT8 (Tuần 8 - Ngày 3)

Quá trình lượng tử hóa mô hình sang định dạng số nguyên 8-bit (INT8) được thực hiện thông qua kỹ thuật **Lượng tử hóa trong quá trình huấn luyện (Quantization-Aware Training - QAT)** kết hợp với công cụ biên dịch **TensorRT**. Phương pháp này giúp mô hình thích nghi dần với định dạng số học với độ chính xác thấp mà không làm giảm đáng kể hiệu năng tổng thể.

Các bước thực hiện chi tiết bao gồm:

*   **1.1. Huấn luyện nhận thức lượng tử hóa (QAT):**
    *   **Mã nguồn thực thi:** `scripts/qat.py`
    *   **Quy trình:** Mô hình sau khi trải qua bước cắt tỉa (pruned) được đưa vào bộ huấn luyện `QuantizationPrunedTrainer`. Thay vì lượng tử hóa trực tiếp (Post-Training Quantization - PTQ) vốn dễ gây hao hụt độ chính xác, QAT giúp mô hình làm quen và điều chỉnh trọng số với việc biểu diễn ở định dạng INT8 ngay trong lúc huấn luyện.

*   **1.2. Xuất mô hình trung gian sang ONNX:**
    *   **Mã nguồn thực thi:** `scripts/export_onnx.py`
    *   **Quy trình:** Mô hình tốt nhất (best.pt) có chứa cấu trúc lượng tử hóa được xuất ra định dạng trung gian ONNX thông qua hàm `export_onnx_pruned`. Các lớp lượng tử hóa giả lập (fake-quant nodes) được cấu hình giữ nguyên để hỗ trợ bộ biên dịch đọc hiểu và ánh xạ cấu trúc mạng.

*   **1.3. Biên dịch mô hình sang TensorRT Engine:**
    *   **Mã nguồn thực thi:** `scripts/build_engine.py`
    *   **Quy trình:** Thông qua API của TensorRT, file ONNX được dịch thành engine (`.engine`). Quá trình này thiết lập các cờ quan trọng như `trt.BuilderFlag.INT8`, cấp phát bộ nhớ Workspace. TensorRT sẽ tự động tái cấu trúc đồ thị tính toán (graph optimization) để tối ưu hóa hoàn toàn cho các thiết bị phần cứng của NVIDIA, tạo ra một tệp tin gọn nhẹ và siêu tốc.

---

### 2. Chạy kiểm thử mô hình sau lượng tử hóa để đo lường lại độ chính xác và tốc độ (FPS) (Tuần 9 - Ngày 2)

Sau khi có được mô hình TensorRT (.engine) hoặc mô hình QAT (.pt), công tác kiểm thử (validation) được tiến hành một cách cẩn trọng thông qua kịch bản tự động để bảo đảm tính khách quan.

*   **Mã nguồn thực thi:** `scripts/val.py`

**Tiêu chí và quy trình đánh giá:**

*   **2.1. Đánh giá tốc độ và Độ trễ (Latency & FPS):**
    *   **Warm-up:** Sử dụng một ma trận đầu vào giả (dummy input) có kích thước tương đương ảnh thật (ví dụ: `1x3x640x640`) chạy qua mô hình một lần nhằm đánh thức GPU và các luồng cấp phát bộ nhớ.
    *   **Benchmark:** Thực hiện lặp lại vòng suy luận liên tục (mặc định 50 lần vòng lặp `benchmark-runs`) và sử dụng hàm thời gian chuẩn xác (`time.perf_counter()`) để đo đạc.
    *   **Kết quả đo:** Thông số thời gian được tự động tính trung bình để xuất ra **Độ trễ trung bình (Average Latency - ms/ảnh)** và từ đó quy đổi trực tiếp ra số **Khung hình/giây (Approx FPS)**.

*   **2.2. Đánh giá bảo toàn Độ chính xác (Accuracy Metrics):**
    *   Khởi chạy tiến trình `model.val()` trên tập dữ liệu kiểm thử (định cấu hình tại thư mục chuẩn `data.yaml`).
    *   Chương trình tự động kết xuất các thông số tổng hợp quan trọng gồm:
        *   **Precision (P):** Độ chính xác của các dự đoán.
        *   **Recall (R):** Tỉ lệ phát hiện hoặc độ bao phủ.
        *   **mAP@50:** Mean Average Precision ở ngưỡng IoU 0.5.
        *   **mAP@50-95:** Mean Average Precision trung bình trên toàn dải ngưỡng IoU từ 0.5 tới 0.95 (Thước đo chặt chẽ nhất).
    *   **Lưu ý kỹ thuật:** Trong tình huống xác thực trực tiếp trên checkpoint chưa compile `.pt`, cờ `--skip-fuse` được sử dụng nhằm tắt tính năng gộp Batch Normalization (BN folding), giữ cho cấu trúc mạng không bị sai lệch biểu diễn so với quá trình QAT trước đó.

---

### 3. Tổng hợp dữ liệu, báo cáo kết quả và thảo luận nhóm (Tuần 9 - Ngày 5)

Bước cuối cùng là phân tích thực chứng và trình bày các dữ liệu thu thập được một cách khoa học:

*   **3.1. Đối sánh (Benchmarking) và Phân tích:**
    *   Thiết lập bảng so sánh ba trục mô hình: **Baseline (YOLOv8)** vs **Pruned (Cắt tỉa)** vs **QAT INT8 (Lượng tử hóa)**.
    *   Làm nổi bật mức độ thuyên giảm về dung lượng ổ đĩa (MB), phần trăm cải thiện tốc độ suy luận (FPS) so với phần suy giảm vô cùng nhỏ của độ chính xác (mAP). Kết quả sẽ cho thấy lợi ích to lớn của phương pháp tối ưu hóa.

*   **3.2. Trao đổi và Nghiệm thu với nhóm:**
    *   Trình bày số liệu FPS hiện tại để nhóm ra quyết định đánh giá tính khả thi ứng dụng: Liệu tốc độ này đã đủ yêu cầu thời gian thực (real-time) cho các hệ thống phần cứng biên (Edge AI / IoT Devices) hay chưa.
    *   Phân tích các nút thắt hiệu năng hoặc các lỗi bất ngờ (ví dụ: Out of Memory khi build engine) xảy ra trong giai đoạn 1 để rút kinh nghiệm.

*   **3.3. Hoàn thiện tài liệu:**
    *   Tất cả các số liệu dạng bảng, biểu đồ Precision-Recall, Confusion Matrix sinh ra từ `val.py` cùng với phần biện luận đánh giá được chèn vào văn bản báo cáo cuối cùng để nghiệm thu nghiên cứu.
