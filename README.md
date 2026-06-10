# Dự Án Học Máy: Hồi Quy CO2 Và Phân Loại Khách Hàng Rời Bỏ

Repository này gồm hai dự án học máy hoàn chỉnh, được xây dựng theo cùng một quy trình: khám phá dữ liệu, tiền xử lý, so sánh mô hình, tối ưu siêu tham số, đánh giá trên tập kiểm tra và xuất kết quả báo cáo.

Mỗi bài toán có dữ liệu riêng, pipeline xử lý riêng, notebook riêng và thư mục kết quả riêng. Toàn bộ dự án có thể chạy lại trực tiếp bằng lệnh Python, giúp đảm bảo tính tái lập của kết quả.

## Tổng Quan Dự Án

| Dự án | Loại bài toán | Mục tiêu | Mô hình cuối cùng |
| --- | --- | --- | --- |
| Dự đoán phát thải CO2 của xe | Hồi quy | Dự đoán lượng CO2 phát thải dựa trên thông tin kỹ thuật và mức tiêu thụ nhiên liệu của xe. | Linear Regression |
| Dự đoán khách hàng rời bỏ dịch vụ | Phân loại nhị phân | Dự đoán khả năng khách hàng viễn thông rời bỏ dịch vụ. | Logistic Regression |

## Kết Quả Chính

### Bài toán CO2

Bài toán hồi quy CO2 đạt kết quả cao vì lượng phát thải có quan hệ chặt với mức tiêu thụ nhiên liệu, dung tích động cơ và các đặc trưng kỹ thuật của xe.

| Chỉ số | Giá trị |
| --- | ---: |
| MAE | 3.56 |
| RMSE | 5.45 |
| R2 | 0.9921 |

### Bài toán Churn

Bài toán phân loại churn tập trung vào việc phát hiện nhóm khách hàng có nguy cơ rời bỏ dịch vụ. Ngưỡng dự đoán cuối cùng được lựa chọn dựa trên hiệu quả cân bằng giữa Recall và F1-score.

| Chỉ số | Giá trị |
| --- | ---: |
| Accuracy | 0.7495 |
| Precision | 0.5182 |
| Recall | 0.7661 |
| F1-score | 0.6182 |
| ROC-AUC | 0.8398 |

## Quy Trình Thực Hiện

Hai dự án được triển khai theo cùng một quy trình học máy:

1. Nạp dữ liệu gốc từ thư mục `data/raw`.
2. Kiểm tra dữ liệu, kiểu dữ liệu, giá trị thiếu và phân phối biến.
3. Thực hiện phân tích khám phá dữ liệu bằng biểu đồ EDA.
4. Chia dữ liệu thành tập huấn luyện và tập kiểm tra theo tỷ lệ 80/20.
5. Xây dựng pipeline tiền xử lý bằng scikit-learn.
6. So sánh nhiều mô hình bằng cross-validation.
7. Tối ưu siêu tham số bằng RandomizedSearchCV.
8. Đánh giá mô hình tốt nhất trên tập kiểm tra độc lập.
9. Xuất biểu đồ, bảng chỉ số, bảng so sánh mô hình và phân tích lỗi.

## Cấu Trúc Repository

```text
A49932_hocmay/
├── data/raw/                 # Dữ liệu gốc
├── ml_coursework/            # Package Python chính
│   ├── co2.py                # Pipeline hồi quy CO2
│   ├── churn.py              # Pipeline phân loại churn
│   ├── data_loader.py        # Tiện ích nạp và kiểm tra dữ liệu
│   ├── figure_viewer.py      # Trình xem biểu đồ trong một cửa sổ
│   ├── settings.py           # Cấu hình đường dẫn và tham số chung
│   └── visual_style.py       # Thiết lập giao diện biểu đồ
├── notebooks/                # Notebook phân tích có thể chạy lại
├── reports/
│   ├── documents/            # Tài liệu kỹ thuật
│   ├── figures/              # Biểu đồ được sinh ra
│   └── tables/               # Bảng kết quả được sinh ra
├── submissions/              # Thư mục nộp riêng cho từng bài toán
├── PROJECT_MAP.md            # Bản đồ cấu trúc dự án
└── requirements.txt          # Danh sách thư viện cần cài đặt
```

## Cài Đặt Môi Trường

Tạo môi trường ảo và cài đặt các thư viện cần thiết:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Cách Chạy Dự Án

Chạy bài toán hồi quy CO2:

```powershell
python -m ml_coursework.co2
```

Chạy bài toán phân loại churn:

```powershell
python -m ml_coursework.churn
```

Chạy kèm trình xem biểu đồ tương tác:

```powershell
python -m ml_coursework.co2 --show
python -m ml_coursework.churn --show
```

## Đầu Ra Của Dự Án

Sau khi chạy, kết quả được lưu trong thư mục `reports`:

| Thư mục | Nội dung |
| --- | --- |
| `reports/figures/co2` | Biểu đồ EDA, so sánh mô hình, phân tích residual và feature importance cho bài CO2. |
| `reports/figures/churn` | Biểu đồ phân phối lớp, tỷ lệ churn, ROC/PR curve, confusion matrix và feature importance cho bài churn. |
| `reports/tables/co2` | Chỉ số hồi quy, kết quả cross-validation, kết quả tuning và phân tích lỗi. |
| `reports/tables/churn` | Chỉ số phân loại, kết quả cross-validation, tối ưu ngưỡng dự đoán và phân tích lỗi. |
| `reports/documents` | Tài liệu kỹ thuật mô tả dự án. |

## Điểm Kỹ Thuật Chính

- Sử dụng random seed cố định để đảm bảo kết quả có thể tái lập.
- Tách riêng pipeline cho bài toán hồi quy và bài toán phân loại.
- Tiền xử lý biến số và biến phân loại bằng scikit-learn pipeline.
- Dùng cross-validation trước khi đánh giá trên tập test để giảm rủi ro overfitting.
- Có bước tối ưu siêu tham số để lựa chọn cấu hình mô hình phù hợp hơn.
- Tách rõ source code, dữ liệu gốc, notebook, báo cáo và thư mục nộp.

## Thư Viện Sử Dụng

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Ghi Chú

Dự án được xây dựng cho mục đích học thuật và trình bày quy trình triển khai một bài toán học máy hoàn chỉnh. Dữ liệu được lưu trực tiếp trong repository để thuận tiện cho việc chạy lại và kiểm chứng kết quả.
