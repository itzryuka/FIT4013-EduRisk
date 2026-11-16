# Tệp mẫu cho "Dự đoán theo danh sách"

Mọi dòng lấy từ **tập test** (sinh viên mô hình chưa thấy khi huấn luyện). Tạo lại bằng `python -m src.make_samples`.

| Tệp | Dòng | Chọn kịch bản | Mục đích |
|---|---|---|---|
| `01_30_sv_co_nhan_that.csv` | 30 (10 mỗi lớp) | Sau năm thứ nhất | Có cột `Target`, xem ngay dự đoán đúng/sai |
| `02_khoa_moi_nhap_hoc.csv` | 25 | Lúc nhập học | Chỉ 24 cột có lúc nhập học, không có nhãn |
| `03_sau_hoc_ky_1.csv` | 40 | Sau học kỳ 1 | 30 cột, phân tách bằng dấu phẩy |
| `04_toan_bo_tap_test.csv` | 885 | Cả 3 kịch bản | Kết quả phải khớp trang *Kết quả mô hình* (Accuracy 70,8% / Macro-F1 0,673 sau năm 1) |
| `05_co_dong_loi.csv` | 10 | Sau năm thứ nhất | 3 dòng có giá trị sai cố ý; hệ thống báo lỗi từng dòng, vẫn dự đoán các dòng còn lại |

Chọn sai kịch bản (ví dụ tệp 02 với "Sau năm thứ nhất") thì hệ thống báo thiếu cột. Đây là hành vi đúng.
