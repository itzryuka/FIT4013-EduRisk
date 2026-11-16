"""Cấu hình chung: đường dẫn, random_state, nhóm cột, kịch bản dự đoán."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "data.csv"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
SCORING = "f1_macro"  # tiêu chí chọn mô hình (dữ liệu mất cân bằng)

TARGET = "Target"
CLASSES = ["Dropout", "Enrolled", "Graduate"]
CLASS_LABELS_VI = {"Dropout": "Bỏ học", "Enrolled": "Còn đang học", "Graduate": "Tốt nghiệp"}

# ---- Nhóm cột (tên đã strip khoảng trắng) ----
NOMINAL = [
    "Marital status", "Application mode", "Course", "Previous qualification",
    "Nacionality", "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
]
BINARY = [
    "Daytime/evening attendance", "Displaced", "Educational special needs", "Debtor",
    "Tuition fees up to date", "Gender", "Scholarship holder", "International",
]
NUMERIC_BASE = [
    "Application order", "Previous qualification (grade)", "Admission grade",
    "Age at enrollment", "Unemployment rate", "Inflation rate", "GDP",
]
SEM_FIELDS = ["credited", "enrolled", "evaluations", "approved", "grade", "without evaluations"]
SEM1 = [f"Curricular units 1st sem ({f})" for f in SEM_FIELDS]
SEM2 = [f"Curricular units 2nd sem ({f})" for f in SEM_FIELDS]

# ---- Kịch bản theo thời điểm dự đoán (chống rò rỉ theo thời gian) ----
# Giả định: thông tin tài chính (Debtor, Tuition fees...) có từ đầu khóa học.
SCENARIOS = {
    "admission": {
        "name": "Lúc nhập học",
        "desc": "Chỉ dùng hồ sơ tuyển sinh, nhân khẩu học, tài chính và kinh tế vĩ mô. Cảnh báo sớm nhất.",
        "columns": NOMINAL + BINARY + NUMERIC_BASE,
    },
    "sem1": {
        "name": "Sau học kỳ 1",
        "desc": "Thêm kết quả học tập học kỳ 1. Cân bằng giữa độ sớm và độ chính xác.",
        "columns": NOMINAL + BINARY + NUMERIC_BASE + SEM1,
    },
    "full": {
        "name": "Sau năm thứ nhất",
        "desc": "Dùng toàn bộ 36 đặc trưng, gồm cả học kỳ 2. Chính xác nhất nhưng cảnh báo muộn.",
        "columns": NOMINAL + BINARY + NUMERIC_BASE + SEM1 + SEM2,
    },
}
SCENARIO_ORDER = ["admission", "sem1", "full"]

# Luồng B: các cột gần như hằng số (>= 97% một giá trị) bị loại
NEAR_CONSTANT = ["Educational special needs", "International", "Nacionality"]
