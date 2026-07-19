# ĐATN PPE Detection

## Cách lấy checkpoint 
Vào Dagshub và chọn experiment muốn lấy checkpoint. Sau đó tìm phần Artifacts và tại checkpoint trong Weights.

Tạo folder checkpoint và đưa file vừa tải vào đây và sửa trong phần config đường dẫn tới model.

## Cách chạy code DEMO
Cài đặt các thư viện cần thiết (cần cài torch cuda để có thể dùng GPU)
```
uv venv
```

```
uv pip install -r requirements.txt
```
Kích hoạt môi trường ảo
```
.venv\Scripts\activate
```
Chạy code demo 
```
streamlit run demo.py
```

## Các notebook còn lại
Notebook được dùng để train và eval model chạy trên colab 

Cần có token của Dagshub và authenticate để có thể track được experiment lên dagshub 
