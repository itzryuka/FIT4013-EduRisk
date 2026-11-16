"""Chạy toàn bộ thí nghiệm: 3 kịch bản × 2 luồng × 3 thuật toán × (mặc định, tinh chỉnh).

Chạy:  python -m src.train
Kết quả: results/experiments.csv|json, results/hparams/*.csv, results/errors_*.csv,
         results/analysis.json, results/figures/*.png, models/*.joblib, models/metadata.json
"""
import json
import logging
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from .config import (BINARY, CLASSES, CV_FOLDS, MODELS_DIR, NOMINAL, RANDOM_STATE, RESULTS_DIR,
                     SCENARIO_ORDER, SCENARIOS, SCORING, SEM1, SEM2, TARGET, TEST_SIZE)
from .data import load_data, split_data
from .evaluate import compute_metrics
from .features import FeatureEngineer, build_preprocessor, describe_flow
from .models import MODEL_NAMES, PARAM_GRIDS, make_model

log = logging.getLogger("train")

# Tham số mặc định của sklearn (để ghi vào bảng kết quả)
DEFAULT_PARAMS = {
    "knn": {"n_neighbors": 5, "weights": "uniform", "p": 2},
    "nb": {"var_smoothing": 1e-9},
    "dt": {"max_depth": None, "min_samples_leaf": 1, "criterion": "gini", "class_weight": None},
}


def build_pipeline(flow, scenario, model_key):
    return Pipeline([
        ("fe", FeatureEngineer(flow=flow, scenario=scenario)),
        ("prep", build_preprocessor(flow, scenario)),
        ("model", make_model(model_key)),
    ])


def _clean_params(params):
    out = {}
    for k, v in params.items():
        k = k.replace("model__", "")
        if isinstance(v, (np.floating,)):
            v = float(v)
        elif isinstance(v, (np.integer,)):
            v = int(v)
        out[k] = v
    return out


def run_experiments(X_train, X_test, y_train, y_test, cv):
    rows = []
    fitted = {}
    hp_dir = RESULTS_DIR / "hparams"
    hp_dir.mkdir(parents=True, exist_ok=True)

    for scenario in SCENARIO_ORDER:
        # Mốc "không học gì": luôn đoán lớp đa số
        dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
        m = compute_metrics(y_test, dummy.predict(X_test), dummy.predict_proba(X_test))
        rows.append({"scenario": scenario, "flow": "-", "model": "dummy", "config": "baseline",
                     "params": {"strategy": "most_frequent"}, "cv_mean": None, "cv_std": None,
                     "fit_seconds": 0.0, **m})

        for flow in ("A", "B"):
            for key in ("knn", "nb", "dt"):
                tag = f"{scenario}/{flow}/{key}"
                # 1) Cấu hình mặc định (baseline)
                t0 = time.perf_counter()
                pipe = build_pipeline(flow, scenario, key)
                cvs = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
                pipe.fit(X_train, y_train)
                m = compute_metrics(y_test, pipe.predict(X_test), pipe.predict_proba(X_test))
                rows.append({"scenario": scenario, "flow": flow, "model": key, "config": "default",
                             "params": DEFAULT_PARAMS[key],
                             "cv_mean": float(cvs.mean()), "cv_std": float(cvs.std()),
                             "fit_seconds": time.perf_counter() - t0, **m})

                # 2) Tinh chỉnh bằng GridSearchCV (chỉ trên tập train)
                t0 = time.perf_counter()
                gs = GridSearchCV(build_pipeline(flow, scenario, key), PARAM_GRIDS[key],
                                  scoring=SCORING, cv=cv, n_jobs=-1, refit=True)
                gs.fit(X_train, y_train)
                best = gs.best_estimator_
                i = gs.best_index_
                m = compute_metrics(y_test, best.predict(X_test), best.predict_proba(X_test))
                rows.append({"scenario": scenario, "flow": flow, "model": key, "config": "tuned",
                             "params": _clean_params(gs.best_params_),
                             "cv_mean": float(gs.cv_results_["mean_test_score"][i]),
                             "cv_std": float(gs.cv_results_["std_test_score"][i]),
                             "fit_seconds": time.perf_counter() - t0, **m})
                fitted[(scenario, flow, key)] = best

                cvr = pd.DataFrame(gs.cv_results_["params"]).rename(columns=lambda c: c.replace("model__", ""))
                cvr["mean_f1_macro"] = gs.cv_results_["mean_test_score"]
                cvr["std_f1_macro"] = gs.cv_results_["std_test_score"]
                cvr["rank"] = gs.cv_results_["rank_test_score"]
                cvr.to_csv(hp_dir / f"{scenario}_{flow}_{key}.csv", index=False)
                log.info("%-18s default F1=%.3f | tuned CV=%.3f±%.3f test F1=%.3f %s",
                         tag, rows[-2]["f1_macro"], rows[-1]["cv_mean"], rows[-1]["cv_std"],
                         rows[-1]["f1_macro"], rows[-1]["params"])
    return rows, fitted


def error_analysis(pipe, X_test, y_test, scenario):
    """Phân tích lỗi trên tập test cho mô hình triển khai."""
    proba = pipe.predict_proba(X_test)
    pred = pipe.classes_[proba.argmax(1)]
    df = X_test.copy()
    df["true"] = y_test.values
    df["pred"] = pred
    for j, c in enumerate(pipe.classes_):
        df[f"p_{c}"] = proba[:, j]
    df["correct"] = df["true"] == df["pred"]

    wrong = df[~df["correct"]].copy()
    wrong["confidence"] = wrong[[f"p_{c}" for c in pipe.classes_]].max(axis=1)
    keep = ["true", "pred", "confidence"] + [f"p_{c}" for c in pipe.classes_] + [
        "Course", "Age at enrollment", "Admission grade", "Scholarship holder", "Debtor",
        "Tuition fees up to date"] + [c for c in (SEM1[3], SEM1[4], SEM2[3], SEM2[4])]
    wrong.sort_values("confidence", ascending=False)[keep].to_csv(
        RESULTS_DIR / f"errors_{scenario}.csv", index_label="row_id")

    def rate_by(col, bins=None, labels=None):
        s = pd.cut(df[col], bins=bins, labels=labels) if bins is not None else df[col]
        g = df.groupby(s, observed=True)["correct"].agg(["mean", "size"])
        return [{"group": str(k), "error_rate": float(1 - v["mean"]), "n": int(v["size"])}
                for k, v in g.iterrows()]

    pairs = (df[~df["correct"]].groupby(["true", "pred"]).size()
             .sort_values(ascending=False).reset_index(name="n"))
    return {
        "n_test": int(len(df)),
        "n_wrong": int((~df["correct"]).sum()),
        "confusion_pairs": pairs.to_dict("records"),
        "by_course": sorted(rate_by("Course"), key=lambda r: -r["error_rate"]),
        "by_age": rate_by("Age at enrollment", bins=[0, 19, 22, 26, 35, 100],
                          labels=["≤19", "20–22", "23–26", "27–35", ">35"]),
        "by_scholarship": rate_by("Scholarship holder"),
        "by_debtor": rate_by("Debtor"),
        "by_tuition": rate_by("Tuition fees up to date"),
        "confident_mistakes": int((wrong["confidence"] >= 0.8).sum()),
    }


def build_metadata(df, X_train):
    """Thông tin cột cho giao diện (giá trị hợp lệ, mặc định, khoảng)."""
    meta = {"columns": {}, "class_counts": df[TARGET].value_counts().to_dict(),
            "n_rows": int(len(df)), "n_features": int(df.shape[1] - 1)}
    for c in X_train.columns:
        s = X_train[c]
        if c in NOMINAL:
            vc = s.value_counts()
            meta["columns"][c] = {"type": "nominal", "default": int(vc.index[0]),
                                  "options": [int(v) for v in vc.index]}
        elif c in BINARY:
            meta["columns"][c] = {"type": "binary", "default": int(s.mode()[0])}
        else:
            is_int = bool((s % 1 == 0).all())
            meta["columns"][c] = {"type": "int" if is_int else "float",
                                  "min": float(s.min()), "max": float(s.max()),
                                  "default": float(s.median()),
                                  "mean": float(s.mean())}
    # Trung bình theo lớp để so sánh trong phần giải thích dự đoán
    num = [c for c in X_train.columns if c not in NOMINAL]
    meta["class_means"] = df.groupby(TARGET)[num].mean().round(3).to_dict("index")
    return meta


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(RESULTS_DIR / "train_log.txt", "w", "utf-8")])
    t_start = time.perf_counter()
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    log.info("Dữ liệu %s | train %d | test %d | random_state=%d", df.shape, len(X_train),
             len(X_test), RANDOM_STATE)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    rows, fitted = run_experiments(X_train, X_test, y_train, y_test, cv)

    flat = pd.DataFrame([{k: (json.dumps(v, ensure_ascii=False) if k == "params" else v)
                          for k, v in r.items() if k not in ("per_class", "confusion_matrix")}
                         | {f"recall_{c}": r["per_class"][c]["recall"] for c in CLASSES}
                         | {f"f1_{c}": r["per_class"][c]["f1"] for c in CLASSES}
                         for r in rows])
    flat.to_csv(RESULTS_DIR / "experiments.csv", index=False)

    # ---- Chọn mô hình triển khai cho từng kịch bản theo CV macro-F1 (không nhìn tập test) ----
    analysis = {"deployed": {}, "importance": {}, "errors": {}}
    tuned = [r for r in rows if r["config"] == "tuned"]
    for scenario in SCENARIO_ORDER:
        best = max((r for r in tuned if r["scenario"] == scenario), key=lambda r: r["cv_mean"])
        pipe = fitted[(scenario, best["flow"], best["model"])]
        joblib.dump(pipe, MODELS_DIR / f"{scenario}.joblib")
        analysis["deployed"][scenario] = {k: best[k] for k in (
            "flow", "model", "params", "cv_mean", "cv_std", "accuracy", "f1_macro", "roc_auc_ovr")}
        log.info("Triển khai [%s]: %s/%s CV=%.3f test F1=%.3f", scenario, best["flow"],
                 best["model"], best["cv_mean"], best["f1_macro"])

        cols = SCENARIOS[scenario]["columns"]
        pi = permutation_importance(pipe, X_test, y_test, scoring=SCORING, n_repeats=8,
                                    random_state=RANDOM_STATE, n_jobs=-1)
        imp = sorted(({"feature": c, "mean": float(pi.importances_mean[X_test.columns.get_loc(c)]),
                       "std": float(pi.importances_std[X_test.columns.get_loc(c)])} for c in cols),
                     key=lambda d: -d["mean"])
        analysis["importance"][scenario] = imp
        analysis["errors"][scenario] = error_analysis(pipe, X_test, y_test, scenario)

    analysis["setup"] = {
        "random_state": RANDOM_STATE, "test_size": TEST_SIZE, "cv_folds": CV_FOLDS,
        "scoring": SCORING, "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "flows": {"A": describe_flow("A"), "B": describe_flow("B")},
        "param_grids": {k: {p.replace("model__", ""): [None if v is None else
                                                       (float(v) if isinstance(v, np.floating) else v)
                                                       for v in vals]
                            for p, vals in g.items()} for k, g in PARAM_GRIDS.items()},
        "model_names": MODEL_NAMES,
        "train_seconds": round(time.perf_counter() - t_start, 1),
    }
    with open(RESULTS_DIR / "experiments.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1, default=str)
    with open(RESULTS_DIR / "analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=1, default=str)
    with open(MODELS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(build_metadata(df, X_train), f, ensure_ascii=False, indent=1)

    from .figures import make_all
    make_all(df, rows, analysis)
    log.info("Hoàn tất sau %.1f giây", time.perf_counter() - t_start)


if __name__ == "__main__":
    main()
