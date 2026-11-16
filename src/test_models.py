"""Kiểm thử nhanh các mô hình đã lưu trên tập test.

Chạy:  python -m src.test_models
Kiểm tra: tệp .joblib nạp được, dự đoán chạy được, và các chỉ số khớp với lúc huấn luyện
(results/analysis.json). Dùng để xác nhận pipeline tái lập được.
"""
import json
import time

import joblib

from .config import MODELS_DIR, RESULTS_DIR, SCENARIO_ORDER, SCENARIOS
from .data import load_data, split_data
from .evaluate import compute_metrics


def evaluate_saved():
    _, X_test, _, y_test = split_data(load_data())
    with open(RESULTS_DIR / "analysis.json", encoding="utf-8") as f:
        expected = json.load(f)["deployed"]
    out = []
    for s in SCENARIO_ORDER:
        t0 = time.perf_counter()
        pipe = joblib.load(MODELS_DIR / f"{s}.joblib")
        m = compute_metrics(y_test, pipe.predict(X_test), pipe.predict_proba(X_test))
        exp = expected[s]
        out.append({
            "scenario": s, "scenario_name": SCENARIOS[s]["name"], "model": exp["model"], "flow": exp["flow"],
            "n_test": int(len(y_test)), "seconds": time.perf_counter() - t0,
            "accuracy": m["accuracy"], "f1_macro": m["f1_macro"], "roc_auc_ovr": m["roc_auc_ovr"],
            "recall_dropout": m["per_class"]["Dropout"]["recall"],
            "expected_f1": exp["f1_macro"],
            "match": abs(m["f1_macro"] - exp["f1_macro"]) < 1e-9,
            "confusion_matrix": m["confusion_matrix"], "per_class": m["per_class"],
        })
    return out


def main():
    rows = evaluate_saved()
    print(f"{'Kich ban':<12}{'Mo hinh':<10}{'Accuracy':>10}{'MacroF1':>10}{'AUC':>8}{'Rec.Drop':>10}  Khop?")
    for r in rows:
        print(f"{r['scenario']:<12}{r['model'].upper() + '-' + r['flow']:<10}{r['accuracy']:>10.3f}"
              f"{r['f1_macro']:>10.3f}{r['roc_auc_ovr']:>8.3f}{r['recall_dropout']:>10.3f}  "
              f"{'OK' if r['match'] else 'KHAC (' + format(r['expected_f1'], '.3f') + ')'}")
    ok = all(r["match"] for r in rows)
    print("\n[OK] Tat ca mo hinh tai lap dung ket qua." if ok else "\n[CANH BAO] Co mo hinh cho ket qua khac luc huan luyen.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
