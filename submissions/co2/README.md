# Bài Toán Hồi Quy CO2

Thư mục này chứa bộ tài liệu độc lập cho bài toán dự đoán lượng phát thải CO2 của xe. Mục tiêu là xây dựng mô hình hồi quy dự đoán `CO2EMISSIONS` dựa trên thông tin kỹ thuật xe và mức tiêu thụ nhiên liệu.

## Nội Dung Thư Mục

| Thành phần | Mô tả |
| --- | --- |
| `co2.ipynb` | Notebook trình bày quy trình phân tích, huấn luyện và đánh giá mô hình. |
| `data/FuelConsumptionCo2.csv` | Bộ dữ liệu gốc dùng cho bài toán CO2. |
| `figures/` | Biểu đồ EDA, so sánh mô hình, residual analysis và feature importance. |
| `tables/` | Bảng metric, kết quả cross-validation, tuning và phân tích lỗi. |
| `requirements.txt` | Danh sách thư viện cần thiết để chạy notebook. |

## Cách Chạy

Từ thư mục gốc của dự án:

```powershell
python -m ml_coursework.co2
```

Nếu muốn mở trình xem biểu đồ:

```powershell
python -m ml_coursework.co2 --show
```

## Kết Quả Chính

Mô hình cuối cùng là Linear Regression. Kết quả trên tập test:

| Chỉ số | Giá trị |
| --- | ---: |
| MAE | 3.56 |
| RMSE | 5.45 |
| R2 | 0.9921 |

Kết quả cho thấy lượng phát thải CO2 có quan hệ rất mạnh với mức tiêu thụ nhiên liệu và các đặc trưng kỹ thuật của xe.
