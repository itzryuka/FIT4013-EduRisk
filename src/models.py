"""Định nghĩa mô hình và lưới siêu tham số."""
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_STATE

MODEL_NAMES = {"knn": "K-Nearest Neighbors", "nb": "Gaussian Naive Bayes", "dt": "Decision Tree"}


def make_model(key):
    """Mô hình với cấu hình mặc định (dùng làm baseline)."""
    if key == "knn":
        return KNeighborsClassifier()
    if key == "nb":
        return GaussianNB()
    if key == "dt":
        return DecisionTreeClassifier(random_state=RANDOM_STATE)
    raise ValueError(key)


# Phạm vi tìm kiếm (giải thích trong báo cáo):
# - KNN: k lẻ từ nhỏ (dễ overfit) đến lớn (mượt), trọng số đều/khoảng cách, Manhattan/Euclid
# - DT: độ sâu từ nông (dễ diễn giải) đến không giới hạn, số mẫu tối thiểu ở lá để chống overfit,
#       class_weight='balanced' để bù lớp Enrolled ít mẫu
# - NB: var_smoothing theo thang log
PARAM_GRIDS = {
    "knn": {
        "model__n_neighbors": [3, 5, 7, 11, 15, 21, 31, 45],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    },
    "dt": {
        "model__max_depth": [3, 4, 5, 6, 8, 10, None],
        "model__min_samples_leaf": [1, 5, 10, 20, 40, 80],
        "model__criterion": ["gini", "entropy"],
        "model__class_weight": [None, "balanced"],
    },
    "nb": {
        # Mặc định 1e-9 quá nhỏ với cột One-Hot (phương sai gần 0) -> mở rộng tới 1
        "model__var_smoothing": list(np.logspace(-11, 0, 12)),
    },
}
