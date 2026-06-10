# Sơ Đồ Dự Án

File này dùng để nhìn nhanh cấu trúc và vai trò từng phần trong dự án.

## Chạy Dự Án

```powershell
python -m ml_coursework.co2
python -m ml_coursework.churn
```

Nếu muốn hiện biểu đồ:

```powershell
python -m ml_coursework.co2 --show
python -m ml_coursework.churn --show
```

## Code Chính

- ml_coursework/co2.py: bài hồi quy CO2.
- ml_coursework/churn.py: bài phân loại Churn.
- ml_coursework/data_loader.py: nạp dữ liệu.
- ml_coursework/settings.py: cấu hình chung.
- ml_coursework/visual_style.py: giao diện biểu đồ.

## Notebook

- notebooks/co2/co2.ipynb: notebook bài CO2.
- notebooks/churn/churn.ipynb: notebook bài Churn.

Notebook là bản báo cáo có thể chạy được. Nó chứa giải thích, code, bảng kết quả, biểu đồ và kết luận.

## Kết Quả

- reports/figures/co2: biểu đồ bài CO2.
- reports/figures/churn: biểu đồ bài Churn.
- reports/tables/co2: bảng kết quả bài CO2.
- reports/tables/churn: bảng kết quả bài Churn.
- reports/documents: tài liệu Word hỗ trợ báo cáo và vấn đáp.

## Bản Nộp

- submissions/co2: bản nộp bài CO2.
- submissions/churn: bản nộp bài Churn.

Mỗi thư mục bản nộp gồm notebook, dữ liệu, figures, tables, requirements, VIVA_NOTES và VIVA_QA.

## Luồng Hoạt Động

```text
Lệnh chạy
→ nạp dữ liệu
→ EDA
→ xử lý dữ liệu
→ chia train/test
→ Pipeline tiền xử lý
→ cross-validation
→ tuning
→ đánh giá test
→ lưu bảng và biểu đồ
```
