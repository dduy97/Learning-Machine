# Bản Đồ Dự Án

Tài liệu này mô tả nhanh cấu trúc repository, vai trò của từng nhóm file và luồng thực thi chính của dự án.

## Lệnh Chạy Chính

Chạy pipeline hồi quy CO2:

```powershell
python -m ml_coursework.co2
```

Chạy pipeline phân loại churn:

```powershell
python -m ml_coursework.churn
```

Chạy pipeline kèm trình xem biểu đồ:

```powershell
python -m ml_coursework.co2 --show
python -m ml_coursework.churn --show
```

## Source Code

| File | Vai trò |
| --- | --- |
| `ml_coursework/co2.py` | Triển khai toàn bộ luồng hồi quy dự đoán phát thải CO2 của xe. |
| `ml_coursework/churn.py` | Triển khai toàn bộ luồng phân loại khách hàng rời bỏ dịch vụ. |
| `ml_coursework/data_loader.py` | Tập trung logic nạp dữ liệu và kiểm tra sự tồn tại của file dữ liệu. |
| `ml_coursework/settings.py` | Lưu cấu hình chung như đường dẫn, random seed, tỷ lệ train/test và số fold cross-validation. |
| `ml_coursework/visual_style.py` | Thiết lập phong cách biểu đồ dùng chung cho toàn bộ dự án. |
| `ml_coursework/figure_viewer.py` | Hiển thị nhiều biểu đồ trong một cửa sổ với nút chuyển qua lại. |

## Dữ Liệu

| Đường dẫn | Nội dung |
| --- | --- |
| `data/raw/FuelConsumptionCo2.csv` | Dữ liệu mức tiêu thụ nhiên liệu và phát thải CO2 của xe. |
| `data/raw/Telco-Customer-Churn.csv` | Dữ liệu khách hàng viễn thông phục vụ bài toán dự đoán churn. |

## Notebook

| Notebook | Nội dung |
| --- | --- |
| `notebooks/co2/co2.ipynb` | Notebook có thể chạy lại cho bài toán hồi quy CO2. |
| `notebooks/churn/churn.ipynb` | Notebook có thể chạy lại cho bài toán phân loại churn. |

Notebook phản ánh lại các bước chính trong pipeline Python, đồng thời trình bày kết quả theo định dạng phù hợp cho báo cáo.

## Báo Cáo Và Kết Quả

| Đường dẫn | Nội dung |
| --- | --- |
| `reports/figures/co2` | Biểu đồ EDA, so sánh mô hình, phân tích residual và feature importance cho bài CO2. |
| `reports/figures/churn` | Biểu đồ EDA, so sánh mô hình, ROC/PR curve, confusion matrix và feature importance cho bài churn. |
| `reports/tables/co2` | Chỉ số hồi quy, kết quả cross-validation, kết quả tuning và phân tích lỗi. |
| `reports/tables/churn` | Chỉ số phân loại, kết quả cross-validation, tối ưu ngưỡng và phân tích lỗi. |
| `reports/documents` | Tài liệu kỹ thuật mô tả dự án. |

## Thư Mục Nộp

| Đường dẫn | Nội dung |
| --- | --- |
| `submissions/co2` | Bộ tài liệu tách riêng cho bài toán hồi quy CO2. |
| `submissions/churn` | Bộ tài liệu tách riêng cho bài toán phân loại churn. |

Mỗi thư mục nộp gồm notebook, dữ liệu, biểu đồ, bảng kết quả và file dependency tương ứng.

## Luồng Thực Thi

```text
Dữ liệu gốc
→ Nạp và kiểm tra dữ liệu
→ Phân tích khám phá dữ liệu
→ Tiền xử lý bằng pipeline
→ Chia train/test
→ So sánh mô hình bằng cross-validation
→ Tối ưu siêu tham số
→ Đánh giá trên tập test
→ Xuất biểu đồ và bảng kết quả
```

## Ghi Chú Thiết Kế

- Hai bài toán độc lập nhưng dùng cùng một phong cách tổ chức code.
- Các giá trị cấu hình được đặt trong `settings.py` để dễ kiểm soát và tái lập.
- Code tiện ích được tách khỏi logic huấn luyện mô hình để tránh lặp lại.
- Kết quả sinh ra được lưu trong `reports` để phân biệt rõ giữa source code và output.
