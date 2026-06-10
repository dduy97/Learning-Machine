# Bài Toán Phân Loại Churn

Thư mục này chứa bộ tài liệu độc lập cho bài toán dự đoán khách hàng rời bỏ dịch vụ viễn thông. Mục tiêu là xây dựng mô hình phân loại nhị phân dự đoán biến mục tiêu `Churn`.

## Nội Dung Thư Mục

| Thành phần | Mô tả |
| --- | --- |
| `churn.ipynb` | Notebook trình bày quy trình phân tích, huấn luyện và đánh giá mô hình. |
| `data/Telco-Customer-Churn.csv` | Bộ dữ liệu gốc dùng cho bài toán churn. |
| `figures/` | Biểu đồ EDA, so sánh mô hình, ROC/PR curve, confusion matrix và feature importance. |
| `tables/` | Bảng metric, classification report, cross-validation, tuning và phân tích lỗi. |
| `requirements.txt` | Danh sách thư viện cần thiết để chạy notebook. |

## Cách Chạy

Từ thư mục gốc của dự án:

```powershell
python -m ml_coursework.churn
```

Nếu muốn mở trình xem biểu đồ:

```powershell
python -m ml_coursework.churn --show
```

## Kết Quả Chính

Mô hình cuối cùng là Logistic Regression với ngưỡng dự đoán được tối ưu theo F1-score. Kết quả trên tập test:

| Chỉ số | Giá trị |
| --- | ---: |
| Accuracy | 0.7495 |
| Precision | 0.5182 |
| Recall | 0.7661 |
| F1-score | 0.6182 |
| ROC-AUC | 0.8398 |

Kết quả cho thấy mô hình ưu tiên phát hiện khách hàng có nguy cơ rời bỏ, phù hợp với mục tiêu hỗ trợ giữ chân khách hàng.
