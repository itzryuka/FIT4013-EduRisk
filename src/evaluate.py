"""Tính các độ đo đánh giá cho bài toán phân loại 3 lớp."""
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)

from .config import CLASSES


def compute_metrics(y_true, y_pred, y_proba=None):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, zero_division=0)
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": float(p.mean()),
        "recall_macro": float(r.mean()),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, c in enumerate(CLASSES)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
    }
    if y_proba is not None:
        m["roc_auc_ovr"] = roc_auc_score(y_true, y_proba, multi_class="ovr", labels=CLASSES)
    return m
