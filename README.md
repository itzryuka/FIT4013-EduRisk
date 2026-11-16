<h2 align="center">
    <a href="https://dainam.edu.vn/vi/khoa-cong-nghe-thong-tin">
    🎓 Faculty of Information Technology (DaiNam University)
    </a>
</h2>
<h2 align="center">
    Open Source Software Development
</h2>
<div align="center">
    <p align="center">
        <img src="docs/logo/aiotlab_logo.png" alt="AIoTLab Logo" width="170"/>
        <img src="docs/logo/fitdnu_logo.png" alt="AIoTLab Logo" width="180"/>
        <img src="docs/logo/dnu_logo.png" alt="DaiNam University Logo" width="200"/>
    </p>

[![AIoTLab](https://img.shields.io/badge/AIoTLab-green?style=for-the-badge)](https://www.facebook.com/DNUAIoTLab)
[![Faculty of Information Technology](https://img.shields.io/badge/Faculty%20of%20Information%20Technology-blue?style=for-the-badge)](https://dainam.edu.vn/vi/khoa-cong-nghe-thong-tin)
[![DaiNam University](https://img.shields.io/badge/DaiNam%20University-orange?style=for-the-badge)](https://dainam.edu.vn)

</div>

# EduRisk – Hệ thống cảnh báo sớm sinh viên có nguy cơ bỏ học

Bài tập lớn học phần **Nhập môn Học máy (FIT4013)** · Đề tài 8 · Trường Đại học Đại Nam.

EduRisk dự đoán một sinh viên sẽ **Bỏ học (Dropout)**, **Còn đang học (Enrolled)** hay **Tốt nghiệp (Graduate)**. Đầu vào là hồ sơ tuyển sinh, tài chính và kết quả học tập. Hệ thống gồm:

- **Quy trình học máy** viết bằng scikit-learn: so sánh K-NN, Naive Bayes và Decision Tree, hai cách tiền xử lý (Luồng A/B), ba thời điểm dự đoán.
- **Ứng dụng web Flask**: dự đoán từng sinh viên hoặc cả danh sách, có giải thích lý do; xem dữ liệu, kết quả thí nghiệm và phân tích lỗi.

| Kịch bản | Thời điểm | Mô hình | Macro-F1 (test) | Accuracy |
|---|---|---|---|---|
| `admission` | Lúc nhập học | Decision Tree · Luồng A | 0,531 | 55,8% |
| `sem1` | Sau học kỳ 1 | Decision Tree · Luồng B | 0,634 | 67,0% |
| `full` | Sau năm thứ nhất | Decision Tree · Luồng B | 0,673 | 70,8% |

Mốc so sánh (DummyClassifier, luôn đoán lớp đông nhất) chỉ đạt Macro-F1 = 0,222.

---

## 1. Yêu cầu
- Windows, macOS hoặc Linux
- **Python 3.10 trở lên** (đã thử với Python 3.14, scikit-learn 1.9, pandas 3.0, Flask 3.1)
- Khoảng 200 MB dung lượng trống, kết nối mạng để cài thư viện

## 2. Chuẩn bị dữ liệu
Dữ liệu không đưa lên repository. Các bước tải:
1. Tải file zip tại [UCI ML Repository #697 – Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success).
2. Giải nén, đổi tên file CSV thành `data.csv` và đặt vào thư mục `data/`:
   ```
   data/data.csv
   ```

Bộ dữ liệu có 4.424 sinh viên × 36 đặc trưng + cột nhãn `Target`, phân tách bằng dấu `;`. Giấy phép CC BY 4.0. Nguồn: Realinho và cộng sự (2022), *Data* 7(11):146, [doi:10.3390/data7110146](https://doi.org/10.3390/data7110146).

## 3. Cài đặt và chạy

### Cách nhanh (Windows): `Run.bat`
Nhấp đúp `Run.bat` rồi nhấn **Enter**. Script sẽ tự cài thư viện nếu thiếu, tự huấn luyện nếu chưa có mô hình, rồi mở trình duyệt tại http://127.0.0.1:5000.

| Lựa chọn | Tác dụng |
|---|---|
| `1` | Chạy nhanh: cài thư viện → huấn luyện (nếu cần) → mở web |
| `2` | Chỉ cài thư viện |
| `3` | Tạo môi trường ảo `.venv` |
| `4` | Huấn luyện lại toàn bộ (~3 phút) |
| `5` | Chỉ chạy web |
| `6` | Kiểm thử: nạp mô hình đã lưu và đối chiếu với kết quả lúc huấn luyện |

Có thể chạy thẳng một lựa chọn mà không hiện menu, ví dụ `Run.bat 4` hoặc `Run.bat 6`.

### Cách thủ công (mọi hệ điều hành)
```bash
python -m venv .venv
```
```bash
.venv\Scripts\activate
```
Trên macOS/Linux, kích hoạt bằng `source .venv/bin/activate`.
```bash
pip install -r requirements.txt
```
```bash
python -m src.train
```
Lệnh trên chạy toàn bộ thí nghiệm, lưu mô hình, kết quả và hình (~3 phút). Sau đó mở web:
```bash
python -m app.app
```
Mở http://127.0.0.1:5000 trên trình duyệt. Dừng web bằng `Ctrl+C`.

### Lệnh hữu ích khác
```bash
python -m src.test_models
```
Nạp lại 3 mô hình đã lưu, chấm lại trên tập test và báo `OK` nếu khớp kết quả lúc huấn luyện.
```bash
python -m src.make_samples
```
Tạo lại các tệp CSV mẫu trong `data/samples/`.

## 4. Sử dụng ứng dụng web
| Trang | Chức năng |
|---|---|
| **Trang chủ** | Tổng quan và lối vào 3 chức năng chính |
| **Dự đoán một sinh viên** | Chọn thời điểm → nhập thông tin (hoặc *Lấy sinh viên mẫu*) → xem mức nguy cơ, xác suất 3 lớp, khuyến nghị. Có 3 cách giải thích: yếu tố chính, luật quyết định của cây, 7 sinh viên tương tự |
| **Dự đoán theo danh sách** | Tải tệp CSV, nhận danh sách xếp theo xác suất bỏ học, lọc theo mức nguy cơ, tải kết quả về |
| **Dữ liệu** | Phân phối từng đặc trưng theo nhãn, tương quan, chất lượng dữ liệu, thống kê mô tả |
| **Kết quả mô hình** | Tóm tắt kết luận, so sánh Luồng A/B, mặc định/tinh chỉnh, bảng 39 cấu hình, khảo sát siêu tham số |
| **Phân tích lỗi** | Độ quan trọng đặc trưng, các cặp nhầm lẫn, tỉ lệ lỗi theo nhóm, các ca sai tự tin |
| **Thử nghiệm mô hình** | Tự chọn thuật toán, luồng, siêu tham số rồi huấn luyện ngay; nút kiểm thử 3 mô hình đã lưu |
| **Phương pháp** | Giả định, hạn chế, quy trình |

**Mức nguy cơ** dựa trên xác suất bỏ học: ≥ 50% là *Nguy cơ cao*, ≥ 30% là *Cần theo dõi*, còn lại là *Nguy cơ thấp*.

### Thử nhanh chức năng dự đoán theo danh sách
Thư mục [`data/samples/`](data/samples/) có 5 tệp mẫu lấy từ tập test, tức là sinh viên mô hình chưa thấy khi huấn luyện:

| Tệp | Chọn kịch bản | Ghi chú |
|---|---|---|
| `01_30_sv_co_nhan_that.csv` | Sau năm thứ nhất | Có cột `Target`, xem ngay đúng/sai |
| `02_khoa_moi_nhap_hoc.csv` | Lúc nhập học | Chỉ 24 cột có lúc nhập học |
| `03_sau_hoc_ky_1.csv` | Sau học kỳ 1 | Phân tách bằng dấu phẩy |
| `04_toan_bo_tap_test.csv` | Bất kỳ | 885 dòng, kết quả khớp bảng ở đầu trang |
| `05_co_dong_loi.csv` | Sau năm thứ nhất | 3 dòng lỗi cố ý, hệ thống báo lỗi từng dòng |

Tệp tự chuẩn bị cần có tên cột giống dữ liệu gốc, phân tách bằng `;` hoặc `,`, và đủ các cột của kịch bản đã chọn.

## 5. Phương pháp
| Thành phần | Lựa chọn |
|---|---|
| Chia dữ liệu | Train 3.539 / Test 885, `stratify=y`, `random_state=42`. Tập test chỉ dùng để báo cáo |
| Kịch bản thời điểm | `admission` 24 cột · `sem1` 30 cột · `full` 36 cột, để tránh dùng thông tin "tương lai" (rò rỉ theo thời gian) |
| Chống rò rỉ kỹ thuật | Mọi bước học tham số (OneHotEncoder, StandardScaler) nằm trong `Pipeline`, chỉ fit trên phần train |
| **Luồng A** | One-Hot toàn bộ cột mã + StandardScaler, giữ nguyên các cột |
| **Luồng B** | Bỏ 3 cột gần như hằng số; gộp mã hiếm (<1%); log(tuổi); thêm tỉ lệ đạt, tỉ lệ đạt/lần thi, cờ "không đăng ký môn", xu hướng điểm HK2−HK1, tổng số môn đạt |
| Thuật toán | K-Nearest Neighbors, Gaussian Naive Bayes, Decision Tree |
| Tinh chỉnh | `GridSearchCV`, `StratifiedKFold(5)`, tiêu chí **Macro-F1** (dữ liệu mất cân bằng 50/32/18) |
| Chọn mô hình triển khai | Cấu hình có **CV Macro-F1** cao nhất trong từng kịch bản, không chọn theo tập test |
| Baseline | `DummyClassifier(most_frequent)` và từng thuật toán với tham số mặc định |
| Diễn giải | Permutation importance (theo Macro-F1), phân tích lỗi theo nhóm, luật của cây, láng giềng gần nhất |

Tổng cộng **39 cấu hình** = 3 kịch bản × 2 luồng × 3 thuật toán × (mặc định + tinh chỉnh) + 3 Dummy.

### Kết quả chi tiết (Macro-F1 trên tập test, mô hình đã tinh chỉnh)
| Kịch bản | Dummy | K-NN A / B | Naive Bayes A / B | Decision Tree A / B |
|---|---|---|---|---|
| Lúc nhập học | 0,222 | 0,505 / 0,510 | 0,511 / 0,521 | **0,531** / 0,531 |
| Sau HK1 | 0,222 | 0,580 / 0,618 | 0,545 / 0,611 | 0,641 / **0,634** |
| Sau năm 1 | 0,222 | 0,619 / 0,678 | 0,614 / 0,665 | 0,649 / **0,673** |

Số in đậm là mô hình được triển khai, chọn theo CV. Sau năm 1, K-NN Luồng B có điểm test cao hơn một chút, nhưng cây có CV cao hơn (0,685 so với 0,679).

**Nhận xét chính**
- **Luồng B cải thiện K-NN và Naive Bayes** ở kịch bản sau HK1 và sau năm 1 (+0,04 đến +0,07); lúc nhập học chỉ tăng khoảng +0,01 vì chưa có cột điểm để tạo đặc trưng mới. Decision Tree gần như không đổi vì cây tự tìm ngưỡng trên từng cột, không dựa vào khoảng cách.
- **Naive Bayes mặc định gần bằng Dummy** (F1 ≈ 0,22) do `var_smoothing=1e-9` quá nhỏ với cột One-Hot. Tinh chỉnh lên 0,01–0,1 thì F1 tăng lên 0,51–0,66.
- **Dự đoán càng sớm càng kém chính xác:** 0,53 → 0,63 → 0,67. Kết quả học kỳ là tín hiệu mạnh nhất.
- **Lớp "Còn đang học" khó nhất** vì nằm giữa hai lớp còn lại.

## 6. Cấu trúc thư mục
```
├── Run.bat                  menu chạy nhanh trên Windows
├── requirements.txt         thư viện cần cài
├── data/
│   ├── data.csv             dữ liệu gốc (tự tải, không commit)
│   └── samples/             5 tệp CSV mẫu cho dự đoán theo danh sách
├── src/                     quy trình học máy
│   ├── config.py            đường dẫn, random_state, nhóm cột, 3 kịch bản
│   ├── data.py              đọc CSV, chia train/test
│   ├── features.py          FeatureEngineer + ColumnTransformer (Luồng A/B)
│   ├── models.py            mô hình mặc định + lưới siêu tham số
│   ├── evaluate.py          các chỉ số đánh giá
│   ├── train.py             chạy thí nghiệm, chọn mô hình, phân tích lỗi, lưu kết quả
│   ├── figures.py           hình PNG cho báo cáo
│   ├── labels.py            nhãn tiếng Việt (đã đối chiếu bảng mã UCI)
│   ├── test_models.py       kiểm thử tái lập
│   └── make_samples.py      tạo tệp CSV mẫu
├── app/                     ứng dụng web Flask
│   ├── app.py               route + REST API
│   ├── services.py          nạp mô hình, kiểm tra dữ liệu, dự đoán, giải thích
│   ├── templates/           giao diện (Jinja2)
│   └── static/              CSS, JavaScript, logo (static/img/logo.png)
├── models/                  3 mô hình .joblib + metadata.json
├── results/                 bảng kết quả, siêu tham số, phân tích lỗi, hình, log
├── report/                  báo cáo LaTeX (main.tex) và bản PDF
└── docs/logo/               logo dùng trong README
```

### REST API
| Phương thức | Đường dẫn | Mô tả |
|---|---|---|
| POST | `/api/predict` | Dự đoán một sinh viên (JSON: `scenario`, `features`) |
| POST | `/api/batch` | Dự đoán theo danh sách (form: `scenario`, `file`) |
| GET | `/api/sample` | Lấy một sinh viên mẫu từ tập test |
| GET | `/api/experiments`, `/api/hparams`, `/api/analysis` | Kết quả thí nghiệm, siêu tham số, phân tích lỗi |
| GET | `/api/eda/feature`, `/api/eda/overview` | Dữ liệu cho trang khám phá dữ liệu |
| POST | `/api/playground` | Huấn luyện thử một cấu hình |
| GET | `/api/test-saved` | Kiểm thử 3 mô hình đã lưu |

### Biên dịch báo cáo
Bản PDF: [`report/FIT4013-EduRisk-Bao_cao_De_tai_8.pdf`](report/FIT4013-EduRisk-Bao_cao_De_tai_8.pdf). Mã nguồn LaTeX ở `report/main.tex`, hình ở `report/figures/`. Có hai cách biên dịch:
- **Overleaf:** nén thư mục `report/` thành zip, chọn *New Project → Upload Project*, trình biên dịch pdfLaTeX.
- **Trên máy** (đã cài MiKTeX hoặc TeX Live): chạy lệnh dưới đây 2 lần để có mục lục.
```bash
cd report && pdflatex main.tex
```

## 7. Xử lý sự cố
| Hiện tượng | Cách xử lý |
|---|---|
| `Không tìm thấy data/data.csv` | Làm lại mục 2, kiểm tra đúng tên `data.csv` |
| Web báo chưa có mô hình | Chạy `python -m src.train` hoặc `Run.bat 4` |
| Cổng 5000 đang bận | Tắt chương trình khác đang dùng cổng 5000, hoặc sửa `port=` cuối `app/app.py` |
| `test_models` báo lệch | Phiên bản thư viện khác; chạy lại `python -m src.train` |
| Dự đoán theo danh sách báo thiếu cột | Chọn đúng kịch bản, hoặc bổ sung cột theo tệp mẫu |

## 8. Hạn chế
- Dữ liệu của một trường ở Bồ Đào Nha; cần huấn luyện lại với dữ liệu của trường trước khi dùng thật.
- Dữ liệu không có điểm danh, bài tập hay hoạt động học trực tuyến.
- Ngưỡng nguy cơ 0,5 / 0,3 chọn theo kinh nghiệm, chưa tối ưu theo chi phí.
- Kết quả chỉ dùng để **ưu tiên hỗ trợ** sinh viên, không dùng để phân loại hay xử lý kỷ luật.

## 9. Thành viên nhóm
| Họ tên | MSSV | GitHub | Nhiệm vụ |
|---|---|---|---|
| Nguyễn Đăng Quang (nhóm trưởng) | 1871020481 | [@itzryuka](https://github.com/itzryuka) | EDA, tiền xử lý dữ liệu, kỹ thuật đặc trưng |
| Nguyễn Thị Tuyết | 1871020644 | [@tuyet-nguyen-1809](https://github.com/tuyet-nguyen-1809) | Mô hình, thí nghiệm, đánh giá, phân tích lỗi |
| Đỗ Trung Kiên | 1871020344 | [@trggkien12](https://github.com/trggkien12) | Ứng dụng web Flask, báo cáo LaTeX |

Lớp CNTT 18-09. Giảng viên hướng dẫn: ThS. Lê Thị Thùy Trang.

## 10. Khai báo sử dụng AI
Xem [AI_USAGE.md](AI_USAGE.md).
