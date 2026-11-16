"""Tiền xử lý và kỹ thuật đặc trưng cho hai luồng A (cơ bản) và B (cải tiến).

Mọi bước học tham số từ dữ liệu (scaler, encoder) đều nằm trong Pipeline
nên chỉ được fit trên tập huấn luyện -> hạn chế data leakage.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import BINARY, NEAR_CONSTANT, NOMINAL, SCENARIOS, SEM1, SEM2


def _safe_ratio(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.zeros_like(a), where=b > 0)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Chọn cột theo kịch bản và (với luồng B) tạo đặc trưng mới.

    Không học tham số nào từ dữ liệu (stateless) nên không gây rò rỉ.
    """

    def __init__(self, flow="A", scenario="full"):
        self.flow = flow
        self.scenario = scenario

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cols = SCENARIOS[self.scenario]["columns"]
        out = X[cols].copy()
        if self.flow == "A":
            return out

        # ---- Luồng B ----
        out = out.drop(columns=[c for c in NEAR_CONSTANT if c in out.columns])
        out["Age at enrollment"] = np.log1p(out["Age at enrollment"])  # giảm lệch phải

        for tag, sem in (("s1", SEM1), ("s2", SEM2)):
            if sem[0] not in out.columns:
                continue
            credited, enrolled, evals, approved, grade, _ = sem
            # Tỉ lệ đạt: số môn đạt / số môn đăng ký -> chuẩn hóa theo khối lượng học
            out[f"{tag}_approval_rate"] = _safe_ratio(out[approved], out[enrolled])
            # Tỉ lệ đạt trên số lần đánh giá -> phản ánh hiệu quả thi cử
            out[f"{tag}_eval_success"] = _safe_ratio(out[approved], out[evals])
            # Không đăng ký môn nào -> điểm 0 là "missing trá hình", tách thành cờ riêng
            out[f"{tag}_no_enrolment"] = (out[enrolled] == 0).astype(int)

        if SEM2[0] in out.columns:
            out["grade_trend"] = out[SEM2[4]] - out[SEM1[4]]
            out["approved_total"] = out[SEM1[3]] + out[SEM2[3]]
        return out


def feature_lists(flow, scenario):
    """Trả về (cột danh nghĩa, cột số) sau bước FeatureEngineer."""
    cols = list(SCENARIOS[scenario]["columns"])
    if flow == "B":
        cols = [c for c in cols if c not in NEAR_CONSTANT]
        for tag, sem in (("s1", SEM1), ("s2", SEM2)):
            if sem[0] in cols:
                cols += [f"{tag}_approval_rate", f"{tag}_eval_success", f"{tag}_no_enrolment"]
        if SEM2[0] in cols:
            cols += ["grade_trend", "approved_total"]
    nominal = [c for c in cols if c in NOMINAL]
    numeric = [c for c in cols if c not in NOMINAL]
    return nominal, numeric


def build_preprocessor(flow, scenario):
    nominal, numeric = feature_lists(flow, scenario)
    if flow == "A":
        # One-Hot đầy đủ cho mọi giá trị
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        # Gộp các giá trị hiếm (< 1% mẫu train) thành nhóm "infrequent"
        # -> giảm số chiều, có lợi cho K-NN
        ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01,
                            sparse_output=False)
    return ColumnTransformer(
        [("cat", ohe, nominal), ("num", StandardScaler(), numeric)],
        verbose_feature_names_out=False,
    )


def describe_flow(flow):
    if flow == "A":
        return ("Luồng A – cơ bản: One-Hot toàn bộ biến danh nghĩa, chuẩn hóa StandardScaler "
                "cho biến số/nhị phân, giữ nguyên các cột gốc.")
    return ("Luồng B – cải tiến: loại cột gần như hằng số, gộp giá trị hiếm (<1%) khi One-Hot, "
            "log tuổi nhập học, thêm tỉ lệ đạt, tỉ lệ thi đạt, cờ không đăng ký môn, "
            "xu hướng điểm và tổng số môn đạt.")
