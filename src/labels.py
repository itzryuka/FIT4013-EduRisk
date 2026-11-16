"""Nhãn tiếng Việt cho tên cột và mã giá trị (theo mô tả biến của UCI #697).

Mô hình chỉ dùng mã số; các nhãn này chỉ để hiển thị trên giao diện.
"""

FEATURE_VI = {
    "Marital status": "Tình trạng hôn nhân",
    "Application mode": "Hình thức xét tuyển",
    "Application order": "Thứ tự nguyện vọng (0 = NV1)",
    "Course": "Ngành học",
    "Daytime/evening attendance": "Hệ học",
    "Previous qualification": "Trình độ trước khi nhập học",
    "Previous qualification (grade)": "Điểm trình độ trước (0–200)",
    "Nacionality": "Quốc tịch",
    "Mother's qualification": "Trình độ học vấn của mẹ",
    "Father's qualification": "Trình độ học vấn của cha",
    "Mother's occupation": "Nghề nghiệp của mẹ",
    "Father's occupation": "Nghề nghiệp của cha",
    "Admission grade": "Điểm xét tuyển (0–200)",
    "Displaced": "Học xa nhà",
    "Educational special needs": "Nhu cầu giáo dục đặc biệt",
    "Debtor": "Đang nợ",
    "Tuition fees up to date": "Đóng học phí đúng hạn",
    "Gender": "Giới tính",
    "Scholarship holder": "Có học bổng",
    "Age at enrollment": "Tuổi khi nhập học",
    "International": "Sinh viên quốc tế",
    "Unemployment rate": "Tỉ lệ thất nghiệp (%)",
    "Inflation rate": "Tỉ lệ lạm phát (%)",
    "GDP": "Tăng trưởng GDP (%)",
}
_SEM_VI = {
    "credited": "Số môn được công nhận",
    "enrolled": "Số môn đăng ký",
    "evaluations": "Số lần đánh giá",
    "approved": "Số môn đạt",
    "grade": "Điểm trung bình (0–20)",
    "without evaluations": "Số môn không đánh giá",
}
for _k, _v in _SEM_VI.items():
    FEATURE_VI[f"Curricular units 1st sem ({_k})"] = f"HK1 – {_v}"
    FEATURE_VI[f"Curricular units 2nd sem ({_k})"] = f"HK2 – {_v}"

ENGINEERED_VI = {
    "s1_approval_rate": "HK1 – Tỉ lệ môn đạt",
    "s1_eval_success": "HK1 – Tỉ lệ đạt trên số lần thi",
    "s1_no_enrolment": "HK1 – Không đăng ký môn",
    "s2_approval_rate": "HK2 – Tỉ lệ môn đạt",
    "s2_eval_success": "HK2 – Tỉ lệ đạt trên số lần thi",
    "s2_no_enrolment": "HK2 – Không đăng ký môn",
    "grade_trend": "Xu hướng điểm HK2 − HK1",
    "approved_total": "Tổng số môn đạt năm 1",
}

BINARY_VI = {
    "Daytime/evening attendance": {1: "Ban ngày", 0: "Buổi tối"},
    "Gender": {1: "Nam", 0: "Nữ"},
}
DEFAULT_BINARY = {1: "Có", 0: "Không"}

CODE_VI = {
    "Marital status": {1: "Độc thân", 2: "Đã kết hôn", 3: "Góa", 4: "Ly hôn",
                       5: "Sống chung không đăng ký", 6: "Ly thân"},
    "Course": {
        33: "Công nghệ sản xuất nhiên liệu sinh học", 171: "Thiết kế hoạt hình & đa phương tiện",
        8014: "Công tác xã hội (buổi tối)", 9003: "Nông học", 9070: "Thiết kế truyền thông",
        9085: "Điều dưỡng thú y", 9119: "Kỹ thuật tin học", 9130: "Chăn nuôi ngựa",
        9147: "Quản trị", 9238: "Công tác xã hội", 9254: "Du lịch", 9500: "Điều dưỡng",
        9556: "Vệ sinh răng miệng", 9670: "Quản trị quảng cáo & marketing",
        9773: "Báo chí & truyền thông", 9853: "Giáo dục cơ bản", 9991: "Quản trị (buổi tối)",
    },
    "Application mode": {
        1: "Đợt 1 – chỉ tiêu chung", 2: "Nghị định 612/93", 5: "Đợt 1 – chỉ tiêu đặc biệt (Azores)",
        7: "Đã có bằng đại học khác", 10: "Nghị định 854-B/99", 15: "Sinh viên quốc tế (cử nhân)",
        16: "Đợt 1 – chỉ tiêu đặc biệt (Madeira)", 17: "Đợt 2 – chỉ tiêu chung",
        18: "Đợt 3 – chỉ tiêu chung", 26: "Nghị định 533-A/99 b2 (khác chương trình)",
        27: "Nghị định 533-A/99 b3 (trường khác)", 39: "Trên 23 tuổi", 42: "Chuyển trường",
        43: "Chuyển ngành", 44: "Có bằng chuyên môn kỹ thuật", 51: "Chuyển trường/ngành",
        53: "Có bằng chu kỳ ngắn", 57: "Chuyển trường/ngành (quốc tế)",
    },
    "Previous qualification": {
        1: "Trung học phổ thông", 2: "Cử nhân", 3: "Đại học", 4: "Thạc sĩ", 5: "Tiến sĩ",
        6: "Đã học đại học (chưa xong)", 9: "Lớp 12 chưa hoàn thành", 10: "Lớp 11 chưa hoàn thành",
        12: "Khác – lớp 11", 14: "Lớp 10", 15: "Lớp 10 chưa hoàn thành",
        19: "Giáo dục cơ bản chu kỳ 3 (lớp 9–11)", 38: "Giáo dục cơ bản chu kỳ 2 (lớp 6–8)",
        39: "Khóa chuyên môn công nghệ", 40: "Đại học (chu kỳ 1)",
        42: "Khóa kỹ thuật chuyên nghiệp bậc cao", 43: "Thạc sĩ (chu kỳ 2)",
    },
    "Nacionality": {
        1: "Bồ Đào Nha", 2: "Đức", 6: "Tây Ban Nha", 11: "Ý", 13: "Hà Lan", 14: "Anh",
        17: "Litva", 21: "Angola", 22: "Cape Verde", 24: "Guinea", 25: "Mozambique",
        26: "São Tomé", 32: "Thổ Nhĩ Kỳ", 41: "Brazil", 62: "Romania", 100: "Moldova",
        101: "Mexico", 103: "Ukraina", 105: "Nga", 108: "Cuba", 109: "Colombia",
    },
}

_PARENT_QUAL = {
    1: "THPT (lớp 12)", 2: "Cử nhân", 3: "Đại học", 4: "Thạc sĩ", 5: "Tiến sĩ",
    6: "Đã học đại học (chưa xong)", 9: "Lớp 12 chưa hoàn thành", 10: "Lớp 11 chưa hoàn thành",
    11: "Lớp 7 (hệ cũ)", 12: "Khác – lớp 11", 13: "Năm 2 THPT bổ túc", 14: "Lớp 10",
    18: "Khóa thương mại tổng quát", 19: "Giáo dục cơ bản chu kỳ 3 (lớp 9–11)", 20: "THPT bổ túc",
    22: "Khóa kỹ thuật – nghề", 25: "THPT bổ túc (chưa xong)", 26: "Lớp 7",
    27: "Chu kỳ 2 THPT tổng quát", 29: "Lớp 9 chưa hoàn thành", 30: "Lớp 8",
    31: "Khóa hành chính & thương mại", 33: "Kế toán & hành chính bổ túc", 34: "Không rõ",
    35: "Không biết đọc viết", 36: "Biết đọc nhưng chưa xong lớp 4",
    37: "Giáo dục cơ bản chu kỳ 1 (lớp 4–5)", 38: "Giáo dục cơ bản chu kỳ 2 (lớp 6–8)",
    39: "Khóa chuyên môn công nghệ", 40: "Đại học (chu kỳ 1)", 41: "Khóa chuyên sâu bậc cao",
    42: "Khóa kỹ thuật chuyên nghiệp bậc cao", 43: "Thạc sĩ (chu kỳ 2)", 44: "Tiến sĩ (chu kỳ 3)",
}
_PARENT_OCC = {
    0: "Học sinh/sinh viên", 1: "Lãnh đạo, quản lý", 2: "Chuyên gia trí thức, khoa học",
    3: "Kỹ thuật viên trung cấp", 4: "Nhân viên hành chính", 5: "Dịch vụ cá nhân, bảo vệ, bán hàng",
    6: "Nông dân, lao động nông nghiệp lành nghề", 7: "Thợ lành nghề công nghiệp, xây dựng",
    8: "Vận hành máy móc, lắp ráp", 9: "Lao động phổ thông", 10: "Lực lượng vũ trang",
    90: "Tình huống khác", 99: "(Để trống)",
    101: "Sĩ quan quân đội", 102: "Hạ sĩ quan quân đội", 103: "Quân nhân khác",
    112: "Giám đốc dịch vụ hành chính, thương mại", 114: "Giám đốc khách sạn, nhà hàng, thương mại",
    121: "Chuyên gia khoa học tự nhiên, toán, kỹ thuật", 122: "Chuyên gia y tế", 123: "Giáo viên",
    124: "Chuyên gia tài chính, kế toán, hành chính", 125: "Chuyên gia CNTT-TT",
    131: "Kỹ thuật viên khoa học, kỹ thuật", 132: "Kỹ thuật viên y tế",
    134: "Kỹ thuật viên pháp lý, xã hội, thể thao, văn hóa", 135: "Kỹ thuật viên CNTT-TT",
    141: "Nhân viên văn phòng, thư ký", 143: "Nhân viên dữ liệu, kế toán, thống kê",
    144: "Nhân viên hỗ trợ hành chính khác", 151: "Nhân viên dịch vụ cá nhân", 152: "Người bán hàng",
    153: "Nhân viên chăm sóc cá nhân", 154: "Nhân viên bảo vệ, an ninh",
    161: "Nông dân sản xuất hàng hóa", 163: "Nông dân, ngư dân tự cung tự cấp",
    171: "Thợ xây dựng lành nghề", 172: "Thợ luyện kim, cơ khí lành nghề",
    173: "Thợ in, chế tác dụng cụ chính xác, kim hoàn, thủ công",
    174: "Thợ điện, điện tử lành nghề", 175: "Thợ chế biến thực phẩm, gỗ, may mặc",
    191: "Nhân viên vệ sinh",
    181: "Vận hành máy và thiết bị cố định", 182: "Công nhân lắp ráp", 183: "Lái xe, vận hành thiết bị di động",
    192: "Lao động phổ thông nông nghiệp", 193: "Lao động phổ thông công nghiệp, xây dựng, vận tải",
    194: "Phụ bếp", 195: "Bán hàng rong",
}
CODE_VI["Mother's qualification"] = CODE_VI["Father's qualification"] = _PARENT_QUAL
CODE_VI["Mother's occupation"] = CODE_VI["Father's occupation"] = _PARENT_OCC


def feature_vi(col):
    return FEATURE_VI.get(col) or ENGINEERED_VI.get(col) or col


def code_vi(col, code):
    code = int(code)
    if col in CODE_VI:
        return CODE_VI[col].get(code, f"Mã {code}")
    if col in BINARY_VI:
        return BINARY_VI[col].get(code, str(code))
    return DEFAULT_BINARY.get(code, str(code))


# Nhóm trường trên form dự đoán
FORM_GROUPS = [
    ("personal", "Thông tin cá nhân", "bi-person",
     ["Gender", "Age at enrollment", "Marital status", "Nacionality", "International",
      "Displaced", "Educational special needs"]),
    ("admission", "Tuyển sinh", "bi-mortarboard",
     ["Course", "Daytime/evening attendance", "Application mode", "Application order",
      "Previous qualification", "Previous qualification (grade)", "Admission grade"]),
    ("family", "Gia đình", "bi-house",
     ["Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation"]),
    ("finance", "Tài chính", "bi-wallet",
     ["Scholarship holder", "Tuition fees up to date", "Debtor"]),
    ("sem1", "Học kỳ 1", "bi-1-circle",
     [f"Curricular units 1st sem ({k})" for k in _SEM_VI]),
    ("sem2", "Học kỳ 2", "bi-2-circle",
     [f"Curricular units 2nd sem ({k})" for k in _SEM_VI]),
    ("macro", "Kinh tế vĩ mô (năm nhập học)", "bi-graph-up",
     ["Unemployment rate", "Inflation rate", "GDP"]),
]
