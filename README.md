# Bài Tập Lớn Môn Học Máy

## Tổng Quan

Dự án gồm hai bài học máy độc lập, đúng theo yêu cầu có một bài hồi quy và một bài phân loại.

- CO2: hồi quy, dự đoán lượng phát thải CO2 của xe.
- Churn: phân loại nhị phân, dự đoán khách hàng có rời bỏ dịch vụ hay không.

Mỗi bài có dữ liệu riêng, pipeline xử lý riêng, biểu đồ EDA riêng, mô hình riêng và bảng đánh giá riêng.

## Lệnh Chạy Nhanh

Chạy bài CO2:

```powershell
python -m ml_coursework.co2
```

Chạy bài Churn:

```powershell
python -m ml_coursework.churn
```

Chạy và mở cửa sổ xem biểu đồ có nút Previous/Next:

```powershell
python -m ml_coursework.co2 --show
python -m ml_coursework.churn --show
```

## Cấu Trúc Dự Án

```text
A49932_hocmay/
├── data/raw/                      # dữ liệu gốc
├── ml_coursework/                 # code chính
│   ├── co2.py                     # pipeline hồi quy CO2
│   ├── churn.py                   # pipeline phân loại Churn
│   ├── data_loader.py             # nạp dữ liệu
│   ├── settings.py                # cấu hình đường dẫn và tham số chung
│   └── visual_style.py            # style biểu đồ
├── notebooks/
│   ├── co2/co2.ipynb
│   └── churn/churn.ipynb
├── reports/
│   ├── figures/                   # biểu đồ
│   ├── tables/                    # bảng kết quả
│   └── documents/                 # tài liệu Word hỗ trợ báo cáo
└── submissions/
    ├── co2/                       # bản nộp bài CO2
    └── churn/                     # bản nộp bài Churn
```

## File Chính

- ml_coursework/co2.py: xử lý dữ liệu, EDA, huấn luyện, tuning, đánh giá và lưu kết quả cho bài CO2.
- ml_coursework/churn.py: xử lý dữ liệu, EDA, huấn luyện, tuning threshold, đánh giá và lưu kết quả cho bài Churn.
- ml_coursework/data_loader.py: kiểm tra thư mục, tải dữ liệu nếu thiếu và đọc CSV.
- ml_coursework/settings.py: chứa đường dẫn, random seed, tỷ lệ train/test và số fold cross-validation.
- ml_coursework/visual_style.py: định dạng thống nhất cho toàn bộ biểu đồ.
- ml_coursework/figure_viewer.py: xem biểu đồ trong một cửa sổ thay vì mở nhiều tab.

## Kết Quả Chính Hiện Tại

### CO2

- Mô hình tốt nhất: Linear Regression.
- MAE: khoảng 3.56.
- RMSE: khoảng 5.45.
- R2: khoảng 0.9921.

Ý nghĩa: mô hình dự đoán CO2 rất tốt vì phát thải CO2 có quan hệ gần tuyến tính với mức tiêu thụ nhiên liệu và dung tích động cơ.

### Churn

- Mô hình tốt nhất: Logistic Regression.
- Accuracy: khoảng 0.7495.
- Precision: khoảng 0.5182.
- Recall: khoảng 0.7661.
- F1-score: khoảng 0.6182.
- ROC-AUC: khoảng 0.8398.

Ý nghĩa: mô hình ưu tiên phát hiện khách hàng có nguy cơ rời bỏ, nên Recall và F1-score quan trọng hơn Accuracy đơn thuần.

## Notebook Và Bản Nộp

Notebook chính:

- notebooks/co2/co2.ipynb
- notebooks/churn/churn.ipynb

Bản nộp đã đóng gói:

- submissions/co2
- submissions/churn

Mỗi thư mục bản nộp gồm notebook, dữ liệu, biểu đồ, bảng kết quả, requirements và ghi chú vấn đáp.

## Quy Trình Phân Tích

1. Nạp dữ liệu từ data/raw.
2. Khảo sát dữ liệu và vẽ EDA.
3. Làm sạch dữ liệu và tách biến đầu vào, biến mục tiêu.
4. Chia train/test theo tỷ lệ 80/20.
5. Xây dựng Pipeline tiền xử lý bằng scikit-learn.
6. So sánh nhiều mô hình bằng cross-validation.
7. Tối ưu siêu tham số bằng RandomizedSearchCV.
8. Đánh giá mô hình tốt nhất trên tập test.
9. Lưu biểu đồ, bảng metric, feature importance và phân tích lỗi.

## Thư Viện Sử Dụng

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Ghi Chú Vấn Đáp

Khi trình bày, nên chạy từng bài riêng bằng python -m ml_coursework.co2 và python -m ml_coursework.churn. Nếu thầy hỏi chi tiết, mở notebook tương ứng để chỉ rõ từng bước: EDA, pipeline, tuning, đánh giá và kết luận.
