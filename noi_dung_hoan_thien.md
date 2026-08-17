# Nội dung hoàn thiện cho file "Mô hình triển khai + thực nghiệm.docx"

> [!IMPORTANT]
> Dưới đây là nội dung hoàn chỉnh cho **2 mục** cần bổ sung vào file Word. Bạn có thể copy trực tiếp sang Word và chỉnh sửa định dạng (bảng, công thức) theo template hiện có.

---

## MỤC 1: Mô hình triển khai

### Kiến trúc tổng quan hệ thống tối ưu hóa mô hình

Nghiên cứu này đề xuất một quy trình tối ưu hóa mô hình học sâu đa giai đoạn (Multi-stage Model Optimization Pipeline) nhằm nén và tăng tốc mô hình phát hiện đối tượng YOLOv8 Medium cho việc triển khai trên các thiết bị biên có tài nguyên tính toán hạn chế. Toàn bộ đường ống xử lý được thiết kế theo kiến trúc tuần tự bốn bước nối tiếp, trong đó đầu ra của mỗi giai đoạn trở thành đầu vào bắt buộc cho giai đoạn kế tiếp. Kiến trúc tổng thể bao gồm: **(1) Huấn luyện thưa (Sparsity Training)** → **(2) Cắt tỉa cấu trúc (Structured Pruning)** → **(3) Tinh chỉnh phục hồi (Finetuning)** → **(4) Lượng tử hóa nhận thức huấn luyện (Quantization-Aware Training)**.

### Giai đoạn 1: Huấn luyện thưa (Sparsity Training)

Mục tiêu của giai đoạn đầu tiên là chuẩn bị nền tảng trọng số cho việc cắt tỉa bằng cách áp dụng kỹ thuật chuẩn hóa L1 (L1 Regularization) lên các hệ số tỷ lệ γ (gamma) của các lớp chuẩn hóa theo lô (Batch Normalization — BN). Trong kiến trúc mạng nơ-ron tích chập, mỗi lớp BN chứa một vector γ có vai trò điều chỉnh biên độ đầu ra của kênh đặc trưng tương ứng. Khi một hệ số γ tiệm cận giá trị 0, điều này cho thấy kênh đặc trưng tương ứng đóng góp không đáng kể vào khả năng biểu diễn tổng thể của mạng và có thể được loại bỏ an toàn.

Cụ thể, hàm mất mát tổng hợp được bổ sung thêm một thành phần phạt thưa (sparsity penalty) theo công thức:

$$\mathcal{L}_{total} = \mathcal{L}_{detection} + s \cdot \sum_{\gamma \in \Gamma_{BN}} |\gamma|$$

trong đó $\mathcal{L}_{detection}$ là hàm mất mát phát hiện đối tượng gốc của YOLOv8, $\Gamma_{BN}$ là tập hợp toàn bộ các hệ số γ trong mạng, và $s$ là hệ số sparsity rate kiểm soát cường độ chuẩn hóa. Trong nghiên cứu này, $s$ được thiết lập ở mức 0.015 nhằm cân bằng giữa mức độ thưa hóa và việc duy trì độ chính xác phát hiện.

Trọng số của mô hình cơ sở (baseline) được huấn luyện lại trong 80 chu kỳ (epochs) với kích thước lô (batch size) là 16 mẫu. Cơ chế tính toán độ chính xác hỗn hợp tự động (Automatic Mixed Precision — AMP) được chủ động vô hiệu hóa trong pha lan truyền ngược (backward pass) để đảm bảo gradient của thành phần phạt L1 được tính toán chính xác trên miền dấu phẩy động đầy đủ (FP32). Kết thúc giai đoạn này, khoảng 6.99% tổng số hệ số γ đã hội tụ về giá trị nhỏ hơn ngưỡng 0.001, đồng thời độ chính xác mAP@50 chỉ suy giảm nhẹ từ 0.714 xuống 0.705 (chênh lệch -1.3%) so với mô hình cơ sở. Bản đồ trọng số thưa thu được từ giai đoạn này cung cấp cơ sở định lượng cho thuật toán cắt tỉa ở bước tiếp theo.

### Giai đoạn 2: Cắt tỉa cấu trúc (Structured Pruning)

Dựa trên phân bố trọng số đã được làm thưa, thuật toán cắt tỉa có cấu trúc được áp dụng để loại bỏ vật lý các kênh đặc trưng kém quan trọng khỏi đồ thị tính toán của mạng. Khác biệt với phương pháp cắt tỉa phi cấu trúc (unstructured pruning) vốn chỉ gán giá trị 0 cho các trọng số riêng lẻ mà không thay đổi kiến trúc mạng, cắt tỉa cấu trúc loại bỏ hoàn toàn các kênh tích chập, các hàng/cột tương ứng trong ma trận trọng số, và các tham số BN liên quan. Nhờ vậy, mô hình sau cắt tỉa có kiến trúc vật lý nhỏ hơn và có thể tận dụng trực tiếp sự tăng tốc phần cứng mà không cần thư viện hỗ trợ tính toán thưa chuyên biệt.

Quy trình cắt tỉa được thực thi theo các bước sau: **(i)** thu thập toàn bộ các giá trị |γ| từ các lớp BN đủ điều kiện trong mạng; **(ii)** sắp xếp theo thứ tự tăng dần và xác định ngưỡng cắt tỉa (pruning threshold) tại phân vị ứng với tỷ lệ cắt tỉa mong muốn; **(iii)** loại bỏ các kênh có |γ| nằm dưới ngưỡng và tái cấu trúc các lớp tích chập liền kề để đảm bảo tính tương thích chiều (dimension compatibility). Tỷ lệ cắt tỉa được thiết lập ở mức 30%, với ngưỡng BN γ xác định là 0.891113. Kết quả sau cắt tỉa cho thấy: từ tổng cộng 15.024 kênh ban đầu, 4.652 kênh đã bị loại bỏ, chỉ giữ lại 10.372 kênh (tỷ lệ cắt tỉa thực tế đạt 30.96%). Số lượng tham số giảm từ 25.844.971 xuống còn 14.044.710 (sau khi hợp nhất — fused) và lượng phép tính giảm từ 78.7 GFLOPs xuống 57.8 GFLOPs.

Đánh giá zero-shot ngay sau cắt tỉa ghi nhận sự sụt giảm nghiêm trọng về độ chính xác: mAP@50 tụt từ 0.705 xuống 0.112, khẳng định rằng mô hình sau cắt tỉa cần trải qua quá trình tinh chỉnh phục hồi trước khi có thể đưa vào ứng dụng thực tiễn.

### Giai đoạn 3: Tinh chỉnh phục hồi (Finetuning)

Mục tiêu cốt lõi của giai đoạn tinh chỉnh là khôi phục năng lực biểu diễn đặc trưng và khả năng nhận diện đối tượng của mạng nơ-ron sau khi bị can thiệp kiến trúc bởi quá trình cắt tỉa. Mô hình thu gọn được huấn luyện tiếp trong 100 chu kỳ với kích thước lô mở rộng lên 32 mẫu nhằm tận dụng hiệu quả hơn bộ nhớ GPU — do mô hình sau cắt tỉa có kích thước nhỏ hơn đáng kể, cho phép tăng batch size mà không gây tràn bộ nhớ. Bộ tối ưu hóa AdamW được sử dụng với tốc độ học khởi tạo 0.001 và hệ số suy giảm trọng số (weight decay) 0.0005.

Kết quả sau tinh chỉnh cho thấy mô hình YOLOv8m Pruned-FT đã phục hồi gần như hoàn toàn hiệu suất nhận diện. Cụ thể, chỉ số mAP@50 đạt 0.699, chỉ thấp hơn 2.1% so với mô hình cơ sở (0.714). Đáng chú ý, chỉ số mAP@50-95 đạt 0.358, thậm chí nhỉnh hơn 0.001 điểm so với mô hình gốc (0.357), chứng tỏ mạng nơ-ron thu gọn vẫn duy trì được khả năng định vị và phân loại đối tượng chính xác ở các ngưỡng IoU nghiêm ngặt. Tốc độ suy luận trên GPU T4 cũng được cải thiện từ 11.8 ms xuống 10.7 ms cho mỗi lượt suy luận (giảm 9.3%).

### Giai đoạn 4: Lượng tử hóa nhận thức huấn luyện (Quantization-Aware Training — QAT)

Giai đoạn cuối cùng trong đường ống tối ưu hóa nhằm chuyển đổi độ chính xác số học của mô hình từ dấu phẩy động 32-bit (FP32) sang số nguyên 8-bit (INT8) thông qua phương pháp lượng tử hóa nhận thức huấn luyện. Khác biệt với lượng tử hóa hậu kỳ (Post-Training Quantization — PTQ) vốn áp dụng lượng tử hóa trực tiếp lên mô hình đã huấn luyện xong, QAT tích hợp các nút giả lượng tử hóa (Fake Quantization Nodes) vào đồ thị tính toán trong quá trình huấn luyện, cho phép mạng nơ-ron học cách thích ứng với sai số lượng tử hóa.

Quá trình QAT được thực thi trên nền tảng thư viện NVIDIA PyTorch Quantization, huấn luyện mô hình Pruned-FT trong 10 chu kỳ với kích thước lô 32. Tổng số tham số có thể huấn luyện trong pha QAT là 127 (chỉ bao gồm các tham số scale và zero-point của các nút lượng tử hóa), trong khi toàn bộ trọng số chính của mạng được đóng băng. Mô hình QAT sau đó được xuất sang định dạng ONNX tương thích với công cụ biên dịch NVIDIA TensorRT, sẵn sàng cho việc xây dựng engine suy luận INT8 trên thiết bị biên.

Kết quả đánh giá cho thấy mô hình QAT INT8 duy trì độ chính xác gần như nguyên vẹn so với phiên bản Pruned-FT, với mAP@50 đạt 0.697 (so với 0.699 của Pruned-FT), trong khi dung lượng trọng số giảm xuống chỉ còn 28.6 MB.

### Hệ thống quản lý thực nghiệm

Xuyên suốt toàn bộ đường ống tối ưu hóa, hệ thống theo dõi thực nghiệm MLflow được tích hợp trực tiếp nhằm tự động ghi nhận các siêu tham số (hyperparameters), chỉ số đánh giá (metrics), biểu đồ ma trận nhầm lẫn (confusion matrix), và quản lý hệ thống tệp trọng số (model artifacts) một cách minh bạch tại mỗi giai đoạn. Việc này đảm bảo tính tái tạo (reproducibility) của các kết quả thực nghiệm và cho phép so sánh định lượng giữa các phiên bản mô hình khác nhau.

### Sơ đồ quy trình tối ưu hóa

```
Mô hình cơ sở YOLOv8m (25.8M params, 78.7 GFLOPs)
        │
        ▼
[Giai đoạn 1] Huấn luyện thưa (Sparsity Training)
  • L1 Regularization (sr = 0.015) lên γ_BN
  • 80 epochs, batch 16
  • Kết quả: 6.99% γ < 0.001, mAP50 = 0.705
        │
        ▼
[Giai đoạn 2] Cắt tỉa cấu trúc (Structured Pruning 30%)
  • Ngưỡng BN γ: 0.891113
  • 4.652/15.024 kênh bị loại bỏ (30.96%)
  • Kết quả: 14.04M params, 57.8 GFLOPs, mAP50 = 0.112 (zero-shot)
        │
        ▼
[Giai đoạn 3] Tinh chỉnh phục hồi (Finetuning)
  • AdamW (lr=0.001), 100 epochs, batch 32
  • Kết quả: mAP50 = 0.699, mAP50-95 = 0.358
        │
        ▼
[Giai đoạn 4] Lượng tử hóa nhận thức huấn luyện (QAT INT8)
  • 10 epochs, 127 tham số huấn luyện
  • Kết quả: mAP50 = 0.697, weight = 28.6 MB
        │
        ▼
Xuất ONNX → Biên dịch TensorRT Engine (FP16/INT8)
        │
        ▼
Triển khai trên thiết bị biên NVIDIA Jetson Nano
```

---

## MỤC 2: Kết quả thực nghiệm mô hình

### Tổng quan kết quả

Phần này trình bày kết quả đánh giá toàn diện quy trình tối ưu hóa mô hình YOLOv8m thông qua bốn nhóm chỉ số: **(i)** hiệu quả thu gọn kiến trúc mạng, **(ii)** khả năng phục hồi độ chính xác sau tinh chỉnh, **(iii)** hiệu năng suy luận thời gian thực trên thiết bị biên NVIDIA Jetson Nano, và **(iv)** mức tiêu thụ tài nguyên phần cứng. Kết quả được so sánh trên ba trạng thái mô hình: YOLOv8m Baseline (mô hình gốc), YOLOv8m Pruned (mô hình sau cắt tỉa, chưa tinh chỉnh) và YOLOv8m Pruned-FT (mô hình sau cắt tỉa và tinh chỉnh phục hồi).

### Hiệu quả thu gọn mô hình và sự suy giảm độ chính xác nguyên thủy

Việc áp dụng kỹ thuật cắt tỉa có cấu trúc với tỷ lệ 30% đã mang lại hiệu quả nổi bật trong việc thu gọn quy mô kiến trúc mạng nơ-ron. So với mô hình nguyên bản YOLOv8m, phương pháp này cắt giảm thành công 45.66% tổng số tham số, tương ứng với việc giảm từ 25.845 triệu xuống chỉ còn 14.045 triệu tham số. Đồng thời, độ phức tạp tính toán lý thuyết (theoretical computational complexity) được tối ưu đáng kể khi lượng phép toán dấu phẩy động giảm từ 78.7 GFLOPs xuống 57.8 GFLOPs, tương ứng mức giảm 26.56%. Kết quả này cho thấy việc loại bỏ 30% kênh đặc trưng không chỉ tác động đến chiều rộng (width) của mạng mà còn kéo giảm đồng bộ cả ba chiều đo: số lượng tham số, chi phí tính toán và dung lượng lưu trữ.

**Bảng 1.** So sánh tổng thể các mô hình YOLOv8m trước và sau cắt tỉa

| Mô hình | Tham số (M) | GFLOPs | mAP@50 | mAP@50-95 | Engine (MB) | Latency (ms) | P95 (ms) | FPS |
|---|---|---|---|---|---|---|---|---|
| YOLOv8m Baseline | 25.845 | 78.7 | 0.714 | 0.357 | 101.20 | 232.476 | 233.533 | 4.301 ± 0.003 |
| YOLOv8m Pruned | 14.045 | 57.8 | 0.112 | 0.0608 | 58.58 | 195.112 | 200.109 | 5.125 ± 0.017 |
| YOLOv8m Pruned-FT | 14.045 | 57.8 | 0.699 | 0.358 | 58.57 | 194.875 | 198.510 | 5.131 ± 0.017 |

*Ghi chú: Giá trị FPS được biểu diễn dưới dạng trung bình ± độ lệch chuẩn của ba lần đo độc lập. Các chỉ số mAP được đánh giá trên tập kiểm thử 229 ảnh có nhãn. Latency và FPS được đo trên Jetson Nano sử dụng TensorRT FP16.*

Đặc biệt, khi các mô hình được biên dịch sang định dạng suy luận TensorRT FP16 trên thiết bị biên, dung lượng tệp engine giảm mạnh 42.12%, từ mức 101.20 MB xuống chỉ còn 58.57 MB. Mức giảm này mang ý nghĩa thực tiễn quan trọng đối với các thiết bị biên có dung lượng lưu trữ hạn chế.

Tuy nhiên, cần nhấn mạnh rằng cắt tỉa cấu trúc loại bỏ trực tiếp các kênh đặc trưng nội tại của mạng, phá vỡ phân bố trọng số đã được tối ưu hóa qua quá trình huấn luyện. Hệ quả là mô hình YOLOv8m Pruned (chưa tinh chỉnh) ghi nhận sự sụt giảm nghiêm trọng về độ chính xác: chỉ số mAP@50 tụt dốc từ 0.714 xuống 0.112 (giảm tương đối 84.31%), và chỉ số mAP@50-95 giảm từ 0.357 xuống 0.0608 (giảm tương đối 82.97%). Kết quả này khẳng định rằng mô hình sau cắt tỉa không thể được sử dụng trực tiếp trong ứng dụng mà bắt buộc phải trải qua giai đoạn tinh chỉnh phục hồi.

### Năng lực khôi phục hiệu suất thông qua tinh chỉnh (Fine-tuning)

Giai đoạn tinh chỉnh phục hồi đóng vai trò then chốt trong việc khắc phục hiện tượng suy thoái đặc trưng do quá trình cắt tỉa gây ra. Mô hình YOLOv8m Pruned-FT sau khi hoàn tất 100 chu kỳ huấn luyện tinh chỉnh đã chứng minh khả năng phục hồi hiệu suất nhận diện gần như hoàn toàn. Cụ thể, chỉ số mAP@50 khôi phục lên mức 0.699, chỉ chênh lệch 0.015 điểm tuyệt đối (giảm tương đối khoảng 2.10%) so với phiên bản Baseline. Đáng chú ý hơn, ở dải ngưỡng IoU đánh giá nghiêm ngặt hơn, chỉ số mAP@50-95 đạt mức 0.358, thậm chí nhỉnh hơn 0.001 điểm so với mô hình gốc ban đầu (0.357).

**Bảng 2.** Kết quả đánh giá chi tiết theo từng lớp đối tượng của mô hình Pruned-FT

| Lớp đối tượng | Precision (P) | Recall (R) | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| Tổng thể (all) | 0.755 | 0.647 | 0.699 | 0.358 |
| boots | 0.601 | 0.542 | 0.556 | 0.236 |
| gloves | 0.690 | 0.412 | 0.483 | 0.163 |
| helmet | 0.828 | 0.807 | 0.874 | 0.527 |
| no_boots | 0.674 | 0.446 | 0.479 | 0.189 |
| no_gloves | 0.664 | 0.510 | 0.534 | 0.190 |
| no_helmet | 0.789 | 0.670 | 0.764 | 0.369 |
| no_vest | 0.801 | 0.745 | 0.822 | 0.451 |
| person | 0.907 | 0.866 | 0.929 | 0.592 |
| vest | 0.841 | 0.822 | 0.851 | 0.503 |

Phân tích chi tiết theo từng lớp đối tượng (Bảng 2) cho thấy mô hình Pruned-FT duy trì hiệu suất nhận diện tốt nhất trên các lớp có số lượng mẫu huấn luyện dồi dào: lớp *person* đạt mAP@50 cao nhất là 0.929, tiếp theo là lớp *helmet* (0.874) và *vest* (0.851). Ngược lại, các lớp thiểu số như *gloves* (0.483) và *no_boots* (0.479) ghi nhận chỉ số thấp hơn do sự mất cân bằng trong phân bố nhãn của bộ dữ liệu huấn luyện. Kết quả này gợi ý rằng việc áp dụng các kỹ thuật cân bằng mẫu (class balancing) hoặc tăng cường dữ liệu có chọn lọc (targeted augmentation) có thể cải thiện thêm hiệu suất trên các lớp thiểu số.

Kết quả tổng thể minh chứng rằng mạng nơ-ron có khả năng tái cấu trúc lại các biểu diễn đặc trưng nội tại để thích ứng với kiến trúc thu gọn, duy trì độ chính xác định vị và phân loại đối tượng ở mức tương đương mô hình gốc bất chấp việc phải vận hành trên một kiến trúc tham số hạn hẹp hơn 45.66%.

### Tối ưu hóa tốc độ suy luận và phân tích độ trễ trên thiết bị biên

Trên môi trường thực thi của thiết bị biên NVIDIA Jetson Nano, phiên bản YOLOv8m Pruned-FT ghi nhận sự cải thiện đáng kể về tốc độ xử lý thời gian thực. Tốc độ thông lượng đạt 5.131 ± 0.017 FPS, tăng 19.30% so với mức 4.301 ± 0.003 FPS của mô hình Baseline. Đồng thời, độ trễ trung bình mỗi lượt suy luận được rút ngắn 16.17%, từ 232.476 ms xuống còn 194.875 ms. Độ trễ phân vị 95 (P95 latency) — chỉ số đo lường tính ổn định trong các trường hợp độ trễ cao nhất — cũng được cải thiện 15.00%, thu hẹp từ 233.533 ms xuống 198.510 ms.

**Bảng 3.** Mức thay đổi của mô hình Pruned-FT so với Baseline

| Chỉ số | Baseline | Pruned-FT | Mức thay đổi |
|---|---|---|---|
| Số tham số | 25.845 M | 14.045 M | ↓ Giảm 45.66% |
| GFLOPs | 78.7 | 57.8 | ↓ Giảm 26.56% |
| Dung lượng TensorRT engine | 101.20 MB | 58.57 MB | ↓ Giảm 42.12% |
| mAP@50 | 0.714 | 0.699 | ↓ Giảm 0.015 |
| mAP@50-95 | 0.357 | 0.358 | ↑ Tăng 0.001 |
| Latency trung bình | 232.476 ms | 194.875 ms | ↓ Giảm 16.17% |
| P95 latency | 233.533 ms | 198.510 ms | ↓ Giảm 15.00% |
| FPS | 4.301 | 5.131 | ↑ Tăng 19.30% |

Để làm rõ nguồn gốc của mức tăng tốc, phân tích chi tiết từng thành phần cấu thành độ trễ suy luận được trình bày trong Bảng 4.

**Bảng 4.** Phân tích chi tiết thời gian suy luận TensorRT FP16

| Mô hình | H2D (ms) | GPU Compute (ms) | D2H (ms) | Enqueue (ms) | Tổng latency (ms) |
|---|---|---|---|---|---|
| YOLOv8m Baseline | 0.485 | 231.937 | 0.053 | 7.469 | 232.476 |
| YOLOv8m Pruned | 0.486 | 194.573 | 0.053 | 8.313 | 195.112 |
| YOLOv8m Pruned-FT | 0.485 | 194.337 | 0.052 | 8.195 | 194.875 |

*Ghi chú: H2D — thời gian truyền dữ liệu từ bộ nhớ hệ thống (Host) sang GPU (Device); GPU Compute — thời gian GPU thực hiện các phép tính của mô hình; D2H — thời gian truyền kết quả từ GPU về bộ nhớ hệ thống; Enqueue — thời gian CPU gửi tác vụ thực thi tới TensorRT runtime.*

Kết quả phân tích cho thấy thời gian thực thi cốt lõi trên GPU (GPU Compute Time) là thành phần ghi nhận mức giảm lớn nhất: từ 231.937 ms xuống 194.337 ms (giảm 16.21%). Trong khi đó, độ trễ truyền tải bộ nhớ giữa Host và Device (H2D: ~0.485 ms; D2H: ~0.053 ms) hầu như không biến động giữa các phiên bản mô hình, điều này là hợp lý vì kích thước tensor đầu vào (1×3×640×640) và tensor đầu ra không thay đổi. Thời gian Enqueue tăng nhẹ từ 7.469 ms lên khoảng 8.2 ms ở các mô hình đã cắt tỉa, có thể do sự khác biệt trong cấu trúc đồ thị thực thi mà TensorRT tạo ra cho các kiến trúc mạng khác nhau. Tổng hợp lại, gia tốc suy luận đạt được hoàn toàn bắt nguồn từ việc tinh giảm số lượng phép toán trong đồ thị tính toán của mô hình, chứ không phải từ quá trình truyền dữ liệu.

### Đánh giá mức độ tiêu thụ tài nguyên phần cứng

Bảng 5 trình bày mức tiêu thụ tài nguyên phần cứng trên thiết bị biên NVIDIA Jetson Nano trong suốt quá trình benchmark.

**Bảng 5.** Mức sử dụng tài nguyên trên Jetson Nano

| Mô hình | RAM TB (MB) | RAM đỉnh (MB) | GPU TB (%) | GPU đỉnh (%) | Nhiệt độ TB (°C) | Nhiệt độ đỉnh (°C) |
|---|---|---|---|---|---|---|
| YOLOv8m Baseline | 2363.523 | 2476 | 88.567 | 99 | 40.214 | 43.5 |
| YOLOv8m Pruned | 2362.100 | 2423 | 89.284 | 99 | 42.142 | 43.5 |
| YOLOv8m Pruned-FT | 2363.961 | 2454 | 89.020 | 99 | 42.439 | 43.5 |

*Ghi chú: "TB" biểu thị giá trị trung bình. Dữ liệu được thu thập qua công cụ tegrastats trong suốt quá trình benchmark.*

Bất chấp các can thiệp sâu về mặt kiến trúc, tổng mức tiêu thụ RAM hệ thống trung bình giữa cả ba trạng thái mô hình không ghi nhận sự chênh lệch có ý nghĩa thống kê, duy trì ổn định quanh mức 2.36 GB. Kết quả này phản ánh thực tế rằng dung lượng RAM cấp phát trên thiết bị Jetson Nano chủ yếu bị chi phối bởi nền tảng hệ điều hành Linux, trình điều khiển CUDA, bộ thực thi TensorRT, và các vùng đệm hệ thống (system buffers), chứ không thuần túy phụ thuộc vào kích thước ma trận trọng số của mô hình. Mặc dù vậy, mức giảm đáng kể về dung lượng tệp engine (từ 101.20 MB xuống 58.57 MB) vẫn mang ý nghĩa thực tiễn quan trọng khi xét đến giới hạn dung lượng lưu trữ trên các thiết bị biên nhúng.

Tỷ lệ khai thác phần cứng đồ họa (GPU Utilization) luôn duy trì ở cường độ cao, đạt trung bình từ 88.57% đến 89.28% và thường xuyên chạm đỉnh 99% trong các chu kỳ kiểm thử. Điều này cho thấy GPU Tegra X1 trên Jetson Nano được khai thác gần tối đa năng lực xử lý trong quá trình suy luận TensorRT FP16, xác nhận rằng GPU là thành phần nút thắt cổ chai (bottleneck) chính trong pipeline suy luận trên thiết bị biên.

Về mặt tản nhiệt, hệ thống ghi nhận mức nhiệt độ đỉnh điểm đạt 43.5°C đồng nhất trên tất cả các kịch bản kiểm thử, nằm trong ngưỡng vận hành an toàn của thiết bị (dưới ngưỡng giảm tốc do nhiệt — thermal throttling threshold). Không xuất hiện dấu hiệu quá nhiệt bất thường (thermal anomaly) hoặc chênh lệch nhiệt độ có ý nghĩa giữa các phiên bản mô hình, khẳng định Jetson Nano vận hành ổn định ở trạng thái nhiệt học lý tưởng trong suốt quá trình đánh giá.

### So sánh với mô hình YOLOv8n

Để đánh giá tính cạnh tranh của mô hình đã tối ưu hóa, Bảng 6 trình bày phép so sánh giữa mô hình YOLOv8m Pruned-FT với hai phiên bản tham chiếu: mô hình gốc YOLOv8m Baseline và mô hình nhẹ YOLOv8n (Nano) thuộc cùng họ kiến trúc.

**Bảng 6.** So sánh mô hình Pruned YOLOv8m với các phiên bản tham chiếu

| Chỉ số | YOLOv8n | YOLOv8m Baseline | Pruned YOLOv8m (Pruned-FT) |
|---|---|---|---|
| Số tham số | 3.007.403 | 25.844.971 | 14.044.710 |
| GFLOPs | 8.1 | 78.7 | 57.8 |
| mAP@50 | 0.663 | 0.714 | 0.699 |
| mAP@50-95 | 0.325 | 0.357 | 0.358 |
| Inference (GPU T4) | 0.2 ms | 11.8 ms | 10.7 ms |

Kết quả cho thấy mô hình Pruned YOLOv8m đạt được sự vượt trội rõ ràng so với YOLOv8n ở cả hai chỉ số độ chính xác: mAP@50 cao hơn 3.6 điểm phần trăm (0.699 so với 0.663) và mAP@50-95 cao hơn 3.3 điểm phần trăm (0.358 so với 0.325). Mức chênh lệch này đặc biệt có ý nghĩa trong bài toán phát hiện phương tiện bảo hộ cá nhân, nơi yêu cầu định vị chính xác các đối tượng nhỏ (găng tay, giày bảo hộ) đòi hỏi khả năng trích xuất đặc trưng đa tỷ lệ (multi-scale feature extraction) mà kiến trúc Medium vốn có ưu thế hơn so với phiên bản Nano.

Tuy nhiên, cần ghi nhận rằng mô hình Pruned YOLOv8m vẫn có quy mô tham số lớn hơn đáng kể so với YOLOv8n (14.04M so với 3.00M), dẫn đến chi phí tính toán và tốc độ suy luận chậm hơn. Đây là sự đánh đổi cần được cân nhắc dựa trên yêu cầu cụ thể của ứng dụng triển khai: nếu ưu tiên tốc độ tối đa, YOLOv8n là lựa chọn phù hợp; nếu ưu tiên độ chính xác trong điều kiện tài nguyên trung bình, mô hình Pruned YOLOv8m mang lại sự cân bằng tối ưu hơn.

### Tổng hợp kết quả quy trình tối ưu hóa đa giai đoạn

Bảng 7 tổng hợp các kết quả chính đạt được qua toàn bộ quy trình tối ưu hóa bốn giai đoạn, từ mô hình cơ sở ban đầu đến phiên bản cuối cùng đã cắt tỉa và tinh chỉnh.

**Bảng 7.** Tổng hợp kết quả quy trình tối ưu hóa Baseline → Pruned-FT

| Chỉ số | Baseline YOLOv8m | Pruned-FT YOLOv8m | Mức thay đổi |
|---|---|---|---|
| Số tham số | 25.8 M | 14.0 M | ↓ 45.7% |
| GFLOPs | 78.7 | 57.8 | ↓ 26.6% |
| Dung lượng mô hình (PyTorch) | 52.0 MB | 28.4 MB | ↓ 45.4% |
| Dung lượng TensorRT engine (FP16) | 101.20 MB | 58.57 MB | ↓ 42.1% |
| mAP@50 | 0.714 | 0.699 | ↓ 2.1% |
| mAP@50-95 | 0.357 | 0.358 | ↑ 0.3% |
| Inference (GPU T4) | 11.8 ms | 10.7 ms | ↓ 9.3% |
| Latency trên Jetson Nano | 232.476 ms | 194.875 ms | ↓ 16.2% |
| FPS trên Jetson Nano | 4.301 | 5.131 | ↑ 19.3% |

Kết quả thực nghiệm tổng thể khẳng định rằng quy trình tối ưu hóa đa giai đoạn *Sparsity Training → Structured Pruning → Finetuning → QAT* là giải pháp hiệu quả và có tính hệ thống để nén mô hình phát hiện đối tượng YOLOv8 cho triển khai trên thiết bị biên. Mô hình Pruned-FT đạt được sự cân bằng tối ưu giữa ba yếu tố then chốt: **(1)** duy trì độ chính xác phát hiện tương đương mô hình gốc (mAP@50-95 thậm chí tăng nhẹ 0.3%); **(2)** giảm đáng kể quy mô kiến trúc (45.7% tham số, 26.6% GFLOPs, 42.1% dung lượng engine); và **(3)** cải thiện hiệu năng suy luận thời gian thực trên thiết bị biên (tăng 19.3% FPS, giảm 16.2% latency). Mặc dù mô hình ngay sau cắt tỉa bị suy giảm nghiêm trọng về độ chính xác (mAP@50 giảm 84.31%), quá trình tinh chỉnh phục hồi đã khôi phục gần như hoàn toàn hiệu năng phát hiện, chứng minh tính khả thi của phương pháp trong thực tiễn.

---

> [!TIP]
> **Hướng dẫn copy sang Word:**
> 1. Copy nội dung từng mục và paste vào vị trí tương ứng trong file Word
> 2. Các bảng có thể cần chỉnh lại format trong Word (Insert → Table)
> 3. Các công thức toán học ($...$) cần chuyển sang MathType hoặc Equation Editor trong Word
> 4. Sơ đồ quy trình có thể vẽ lại bằng công cụ Draw/SmartArt trong Word
