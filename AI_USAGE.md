# Khai báo sử dụng công cụ AI

Theo quy định của học phần **Nhập môn Học máy (FIT4013)**, nhóm khai báo việc sử dụng công cụ AI trong bài tập lớn Đề tài 8 – *Dự đoán sinh viên có nguy cơ bỏ học*.

## 1. Công cụ đã dùng
| Công cụ | Nhà phát triển | Hình thức |
|---|---|---|
| Claude Code (Claude Opus) | Anthropic | Trợ lý lập trình chạy trong ứng dụng Claude trên máy tính, làm việc trực tiếp trên thư mục dự án |

## 2. AI đã hỗ trợ những gì
| Phần việc | Mức độ hỗ trợ |
|---|---|
| Phân tích yêu cầu đề bài, đề xuất hướng làm (3 kịch bản thời điểm, 2 luồng tiền xử lý, Macro-F1) | Gợi ý, nhóm lựa chọn |
| Mã nguồn quy trình học máy `src/` (đọc dữ liệu, `FeatureEngineer`, Pipeline, GridSearchCV, phân tích lỗi, vẽ hình) | Sinh mã ban đầu, sửa lỗi |
| Ứng dụng web Flask `app/` (API, kiểm tra đầu vào, giải thích dự đoán, giao diện) | Sinh mã ban đầu, chỉnh giao diện theo yêu cầu của nhóm |
| Đối chiếu nhãn tiếng Việt `src/labels.py` với bảng mã của bài báo gốc | Hỗ trợ tra cứu và đối chiếu |
| Tệp CSV mẫu `data/samples/` | Sinh tệp từ tập kiểm tra |
| README, tài liệu ôn tập, báo cáo LaTeX | Soạn bản nháp, trình bày, biên dịch PDF |
| Tìm hiểu các dự án tương tự trên mạng để so sánh | Tìm kiếm và tóm tắt |

## 3. Nhóm đã tự thực hiện và kiểm chứng
- Chọn đề tài, bộ dữ liệu UCI #697 và quyết định các hướng làm chính do AI đề xuất.
- Đưa ra yêu cầu cho từng phần: giao diện sáng, thanh bên theo phong cách doanh nghiệp, dùng logo Trường Đại học Đại Nam, các chức năng chính gọn và dễ hiểu, dự đoán theo danh sách.
- Chạy lại toàn bộ quy trình bằng `python -m src.train` và kiểm thử bằng `python -m src.test_models`; đối chiếu các con số trong README và báo cáo với `results/experiments.csv` và `results/train_log.txt`.
- Dùng thử ứng dụng web với các tệp mẫu, phát hiện lỗi giao diện (ô tải tệp bị vỡ khung) và yêu cầu sửa.
- Đọc lại, sửa và bổ sung nội dung báo cáo: thông tin nhóm, cấu trúc, phụ lục; tự quyết định giữ hoặc bỏ các phần.
- Phân công: Nguyễn Đăng Quang phụ trách EDA và tiền xử lý; Nguyễn Thị Tuyết phụ trách mô hình, thí nghiệm, đánh giá; Đỗ Trung Kiên phụ trách ứng dụng web và báo cáo. Mỗi thành viên đọc hiểu và giải thích được phần mình phụ trách.

## 4. Các yêu cầu (prompt) chính đã dùng
1. "Phân tích kĩ hướng làm bài tập lớn theo đề cương và rubric của học phần, đề tài 8 dự đoán sinh viên có nguy cơ bỏ học."
2. "Làm bằng Python kèm Flask để dựng giao diện."
3. "Cho tôi biết thêm về dữ liệu mẫu… trong dữ liệu mẫu là dự đoán sinh viên theo các dữ kiện nào?"
4. "Hệ thống web các chức năng chính trông gọn gàng dễ hiểu hơn, gam màu nền sáng, thanh công cụ bên cạnh màu phù hợp như một doanh nghiệp, dùng logo trường Đại Nam."
5. "Làm bản Word giải thích kĩ quy trình hoạt động, tại sao lại như vậy."
6. "Tìm và tham khảo các dự án khác trên mạng, tóm tắt cách họ làm và so sánh."
7. "Kiếm vài file dữ liệu mẫu để thử dự đoán theo danh sách; tổng hợp ngắn gọn để chuẩn bị vấn đáp."
8. "Viết README giới thiệu và hướng dẫn cài, chạy dự án; rồi làm báo cáo LaTeX."
9. "Trong báo cáo có cần thêm ảnh của dự án không? – Chụp và chèn luôn."

## 5. Cam kết
Nhóm hiểu và giải thích được toàn bộ mã nguồn, kết quả đã nộp, và chịu trách nhiệm về tính đúng đắn của nội dung. Mọi kết quả trong báo cáo được tạo ra bằng cách chạy chương trình trên máy của nhóm, không lấy từ AI.
