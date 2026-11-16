"""Tầng dịch vụ cho Flask: nạp mô hình, dự đoán, giải thích, thống kê EDA."""
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier

from src.config import (BINARY, CLASS_LABELS_VI, CLASSES, MODELS_DIR, NOMINAL, RESULTS_DIR,
                        SCENARIO_ORDER, SCENARIOS, SEM1, SEM2, TARGET)
from src.data import load_data, split_data
from src.labels import code_vi, feature_vi
from src.models import MODEL_NAMES


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def risk_level(p_dropout):
    if p_dropout >= 0.5:
        return {"key": "high", "label": "Nguy cơ cao", "advice":
                "Ưu tiên liên hệ cố vấn học tập ngay, xem xét hỗ trợ học phí/học bổng và kèm cặp học thuật."}
    if p_dropout >= 0.3:
        return {"key": "medium", "label": "Cần theo dõi", "advice":
                "Theo dõi kết quả các kỳ tiếp theo, mời tham gia buổi tư vấn và nhóm học tập."}
    return {"key": "low", "label": "Nguy cơ thấp", "advice":
            "Chưa cần can thiệp đặc biệt; tiếp tục theo dõi định kỳ."}


class Service:
    def __init__(self):
        self.df = load_data()
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(self.df)
        self.meta = _read_json(MODELS_DIR / "metadata.json")
        self.analysis = _read_json(RESULTS_DIR / "analysis.json")
        self.experiments = _read_json(RESULTS_DIR / "experiments.json")
        self.models = {s: joblib.load(MODELS_DIR / f"{s}.joblib") for s in SCENARIO_ORDER}
        # Chỉ mục láng giềng gần nhất (trong không gian đặc trưng đã biến đổi của từng mô hình)
        self.nn = {}
        for s, pipe in self.models.items():
            Z = self._transform(pipe, self.X_train)
            self.nn[s] = NearestNeighbors(n_neighbors=7).fit(Z)
        # Khởi động trước (lần gọi đầu của sklearn chậm ~1-2 giây)
        for s in SCENARIO_ORDER:
            self.predict({}, s)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _transform(pipe, X):
        return pipe.named_steps["prep"].transform(pipe.named_steps["fe"].transform(X))

    def scenario_info(self):
        out = []
        for s in SCENARIO_ORDER:
            d = self.analysis["deployed"][s]
            out.append({"key": s, "name": SCENARIOS[s]["name"], "desc": SCENARIOS[s]["desc"],
                        "n_features": len(SCENARIOS[s]["columns"]), "model": d["model"],
                        "model_name": MODEL_NAMES[d["model"]], "flow": d["flow"],
                        "f1": d["f1_macro"], "acc": d["accuracy"], "auc": d.get("roc_auc_ovr"),
                        "cv": d["cv_mean"], "params": d["params"]})
        return out

    def field_spec(self, col):
        m = self.meta["columns"][col]
        spec = {"name": col, "label": feature_vi(col), "type": m["type"], "default": m["default"]}
        if m["type"] == "nominal":
            spec["options"] = [{"value": v, "label": code_vi(col, v)} for v in m["options"]]
        elif m["type"] == "binary":
            spec["options"] = [{"value": v, "label": code_vi(col, v)} for v in (1, 0)]
        else:
            spec.update(min=m["min"], max=m["max"],
                        step=1 if m["type"] == "int" else 0.01)
        return spec

    def build_row(self, features: dict, scenario: str) -> pd.DataFrame:
        """Tạo 1 dòng DataFrame đủ 36 cột; cột thiếu lấy giá trị mặc định (không ảnh hưởng
        vì kịch bản không dùng tới)."""
        row = {}
        for col, m in self.meta["columns"].items():
            v = features.get(col, m["default"])
            if v in (None, ""):
                v = m["default"]
            row[col] = int(float(v)) if m["type"] in ("nominal", "binary", "int") else float(v)
        return pd.DataFrame([row], columns=list(self.meta["columns"]))

    def validate(self, features: dict, scenario: str):
        errors = {}
        for col in SCENARIOS[scenario]["columns"]:
            m = self.meta["columns"][col]
            v = features.get(col)
            if v in (None, ""):
                continue
            try:
                x = float(v)
            except (TypeError, ValueError):
                errors[col] = "Giá trị không phải số"
                continue
            if m["type"] not in ("int", "float"):
                continue
            span = m["max"] - m["min"]
            lo = m["min"] - 0.5 * span if m["min"] < 0 else 0.0
            hi = m["max"] + 0.5 * span
            if not (lo <= x <= hi):
                errors[col] = f"Ngoài khoảng hợp lý (dữ liệu huấn luyện: {m['min']:g}–{m['max']:g})"
        sem_checks = (("1st", SEM1), ("2nd", SEM2))
        for tag, sem in sem_checks:
            if sem[0] not in SCENARIOS[scenario]["columns"]:
                continue
            try:
                enrolled, approved = float(features.get(sem[1], 0)), float(features.get(sem[3], 0))
                if approved > enrolled + float(features.get(sem[0], 0)):
                    errors[sem[3]] = "Số môn đạt lớn hơn số môn đăng ký + được công nhận"
                if not 0 <= float(features.get(sem[4], 0)) <= 20:
                    errors[sem[4]] = "Điểm trung bình phải trong khoảng 0–20"
            except (TypeError, ValueError):
                pass
        return errors

    # ------------------------------------------------------------------ predict
    def predict(self, features: dict, scenario: str):
        pipe = self.models[scenario]
        X = self.build_row(features, scenario)
        proba = pipe.predict_proba(X)[0]
        probs = {c: float(p) for c, p in zip(pipe.classes_, proba)}
        pred = max(probs, key=probs.get)
        return {
            "scenario": scenario,
            "prediction": pred,
            "prediction_vi": CLASS_LABELS_VI[pred],
            "probabilities": [{"cls": c, "label": CLASS_LABELS_VI[c], "p": probs[c]} for c in CLASSES],
            "risk": risk_level(probs["Dropout"]),
            "p_dropout": probs["Dropout"],
            "indicators": self._indicators(X, scenario),
            "neighbors": self._neighbors(X, scenario),
            "rules": self._decision_path(X, scenario),
            "model": self.scenario_info()[SCENARIO_ORDER.index(scenario)],
        }

    def _indicators(self, X, scenario, k=6):
        """So sánh các đặc trưng quan trọng nhất với trung bình của nhóm Bỏ học / Tốt nghiệp."""
        cm = self.meta["class_means"]
        out = []
        for imp in self.analysis["importance"][scenario]:
            col = imp["feature"]
            if col in NOMINAL or imp["mean"] <= 0:
                continue
            v = float(X[col].iloc[0])
            d, g = cm["Dropout"][col], cm["Graduate"][col]
            closer = "Dropout" if abs(v - d) < abs(v - g) else "Graduate"
            if col in BINARY:
                text_v = code_vi(col, v)
            else:
                text_v = f"{v:g}"
            out.append({"feature": col, "label": feature_vi(col), "value": v, "value_text": text_v,
                        "dropout_mean": d, "graduate_mean": g, "enrolled_mean": cm["Enrolled"][col],
                        "closer": closer, "binary": col in BINARY, "importance": imp["mean"]})
            if len(out) >= k:
                break
        return out

    def _neighbors(self, X, scenario):
        pipe = self.models[scenario]
        dist, idx = self.nn[scenario].kneighbors(self._transform(pipe, X))
        rows = self.X_train.iloc[idx[0]]
        labels = self.y_train.iloc[idx[0]].tolist()
        items = []
        for (i, r), lab, d in zip(rows.iterrows(), labels, dist[0]):
            items.append({"outcome": lab, "outcome_vi": CLASS_LABELS_VI[lab], "distance": float(d),
                          "course": code_vi("Course", r["Course"]), "age": int(r["Age at enrollment"]),
                          "admission": float(r["Admission grade"]),
                          "s1_approved": int(r[SEM1[3]]), "s1_enrolled": int(r[SEM1[1]]),
                          "s1_grade": float(r[SEM1[4]])})
        counts = {c: labels.count(c) for c in CLASSES}
        return {"items": items, "counts": counts}

    def _decision_path(self, X, scenario):
        """Với Decision Tree: trả về chuỗi luật quyết định dẫn tới dự đoán (theo thang đo gốc)."""
        pipe = self.models[scenario]
        tree = pipe.named_steps["model"]
        if not isinstance(tree, DecisionTreeClassifier):
            return None
        prep = pipe.named_steps["prep"]
        names = prep.get_feature_names_out()
        scaler = prep.named_transformers_["num"]
        num_cols = list(scaler.feature_names_in_)
        z = self._transform(pipe, X)
        node_ids = tree.decision_path(z).indices
        t = tree.tree_
        rules = []
        for node in node_ids:
            if t.children_left[node] == t.children_right[node]:
                continue
            f = t.feature[node]
            name = names[f]
            thr = t.threshold[node]
            goes_left = z[0, f] <= thr
            if name in num_cols:
                j = num_cols.index(name)
                raw_thr = thr * scaler.scale_[j] + scaler.mean_[j]
                if name == "Age at enrollment" and pipe.named_steps["fe"].flow == "B":
                    raw_thr = float(np.expm1(raw_thr))  # luồng B dùng log1p(tuổi)
                name_vi = feature_vi(name)
                if name in BINARY:
                    val = 0 if goes_left else 1
                    text = f"{name_vi}: {code_vi(name, val)}"
                else:
                    text = f"{name_vi} {'≤' if goes_left else '>'} {raw_thr:.2f}"
            else:
                if name.endswith("_infrequent_sklearn"):
                    col, label = name[: -len("_infrequent_sklearn")], "nhóm giá trị hiếm"
                else:
                    col, _, code = name.rpartition("_")
                    label = code_vi(col, float(code))
                text = f"{feature_vi(col)} {'≠' if goes_left else '='} {label}"
            rules.append(text)
        leaf = node_ids[-1]
        dist = t.value[leaf][0]
        return {"rules": rules, "leaf_samples": int(t.n_node_samples[leaf]),
                "leaf_dist": {c: float(v / dist.sum()) for c, v in zip(tree.classes_, dist)}}

    # ------------------------------------------------------------------ playground
    def playground(self, scenario, flow, model, params, run_cv=False):
        """Huấn luyện nhanh một cấu hình tùy chọn trên tập train, đánh giá trên train và test."""
        import time
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from src.config import CV_FOLDS, RANDOM_STATE
        from src.evaluate import compute_metrics
        from src.train import build_pipeline

        if flow not in ("A", "B"):
            raise ValueError("Luồng phải là A hoặc B")
        if model not in MODEL_NAMES:
            raise ValueError("Mô hình không hợp lệ")
        clean = self._clean_params(model, params)
        pipe = build_pipeline(flow, scenario, model)
        pipe.set_params(**{f"model__{k}": v for k, v in clean.items()})

        t0 = time.perf_counter()
        pipe.fit(self.X_train, self.y_train)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        test = compute_metrics(self.y_test, pipe.predict(self.X_test), pipe.predict_proba(self.X_test))
        pred_s = time.perf_counter() - t0
        train = compute_metrics(self.y_train, pipe.predict(self.X_train))
        out = {"scenario": scenario, "flow": flow, "model": model, "params": clean,
               "fit_seconds": fit_s, "predict_seconds": pred_s, "test": test,
               "train": {k: train[k] for k in ("accuracy", "f1_macro")},
               "n_features_out": int(len(pipe.named_steps["prep"].get_feature_names_out()))}
        if isinstance(pipe.named_steps["model"], DecisionTreeClassifier):
            t = pipe.named_steps["model"]
            out["tree"] = {"depth": int(t.get_depth()), "leaves": int(t.get_n_leaves())}
        if run_cv:
            cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            sc = cross_val_score(build_pipeline(flow, scenario, model).set_params(
                **{f"model__{k}": v for k, v in clean.items()}),
                self.X_train, self.y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
            out["cv"] = {"mean": float(sc.mean()), "std": float(sc.std()), "folds": sc.tolist()}
        dep = self.analysis["deployed"][scenario]
        out["deployed"] = {"model": dep["model"], "flow": dep["flow"], "f1_macro": dep["f1_macro"]}
        return out

    @staticmethod
    def _clean_params(model, p):
        def num(key, lo, hi, cast, default):
            v = p.get(key, default)
            if v in (None, ""):
                return default
            try:
                v = cast(v)
            except (TypeError, ValueError):
                raise ValueError(f"{key} phải là số")
            if not lo <= v <= hi:
                raise ValueError(f"{key} phải trong khoảng {lo}–{hi}")
            return v

        def choice(key, options, default):
            v = p.get(key, default)
            if v not in options:
                raise ValueError(f"{key} phải là một trong {options}")
            return v

        if model == "knn":
            return {"n_neighbors": num("n_neighbors", 1, 300, int, 5),
                    "weights": choice("weights", ["uniform", "distance"], "uniform"),
                    "p": int(choice("p", [1, 2, "1", "2"], 2))}
        if model == "dt":
            depth = p.get("max_depth")
            return {"max_depth": None if depth in (None, "", "None") else num("max_depth", 1, 60, int, None),
                    "min_samples_leaf": num("min_samples_leaf", 1, 1000, int, 1),
                    "criterion": choice("criterion", ["gini", "entropy", "log_loss"], "gini"),
                    "class_weight": None if p.get("class_weight") in (None, "", "None")
                    else choice("class_weight", ["balanced"], None)}
        return {"var_smoothing": num("var_smoothing", 1e-12, 10.0, float, 1e-9)}

    def sample(self, scenario, cls=None, rng=None):
        rng = rng or np.random.default_rng()
        idx = self.y_test.index if not cls else self.y_test[self.y_test == cls].index
        i = rng.choice(idx)
        row = self.X_test.loc[i]
        feats = {c: (int(row[c]) if self.meta["columns"][c]["type"] in ("nominal", "binary", "int")
                     else float(row[c])) for c in self.X_test.columns}
        return {"features": feats, "true": self.y_test.loc[i],
                "true_vi": CLASS_LABELS_VI[self.y_test.loc[i]], "row_id": int(i)}

    def predict_batch(self, df: pd.DataFrame, scenario: str):
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        need = SCENARIOS[scenario]["columns"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError("Thiếu cột: " + ", ".join(missing[:8]) + ("…" if len(missing) > 8 else ""))
        for c, m in self.meta["columns"].items():
            if c not in df.columns:
                df[c] = m["default"]
        X = df[list(self.meta["columns"])].apply(pd.to_numeric, errors="coerce")
        bad = X[need].isna().any(axis=1)
        if bad.all():
            raise ValueError("Không có dòng hợp lệ (giá trị không phải số).")
        pipe = self.models[scenario]
        out = []
        P = pipe.predict_proba(X[~bad])
        good_idx = X.index[~bad]
        pmap = {i: p for i, p in zip(good_idx, P)}
        for i in X.index:
            rec = {"row": int(i) + 1}
            if TARGET in df.columns:
                rec["true"] = str(df.loc[i, TARGET])
            if i not in pmap:
                rec["error"] = "Dữ liệu không hợp lệ"
            else:
                p = dict(zip(pipe.classes_, pmap[i]))
                pred = max(p, key=p.get)
                rec.update(pred=pred, pred_vi=CLASS_LABELS_VI[pred],
                           p_dropout=float(p["Dropout"]), p_enrolled=float(p["Enrolled"]),
                           p_graduate=float(p["Graduate"]), risk=risk_level(p["Dropout"])["key"],
                           course=code_vi("Course", X.loc[i, "Course"]))
            out.append(rec)
        return out


# ---------------------------------------------------------------------- EDA
@lru_cache(maxsize=1)
def eda_overview(service: "Service"):
    df = service.df
    X = df.drop(columns=[TARGET])
    y_num = df[TARGET].map({"Dropout": 0, "Enrolled": 1, "Graduate": 2})
    corr = X.corrwith(y_num).sort_values(key=abs, ascending=False)
    zero_share = (X == 0).mean()
    near_const = [{"feature": c, "label": feature_vi(c),
                   "share": float(X[c].value_counts(normalize=True).iloc[0])}
                  for c in X.columns if X[c].value_counts(normalize=True).iloc[0] >= 0.97]
    num_cols = [c for c in X.columns if c not in NOMINAL and c not in BINARY]
    outliers = []
    for c in num_cols:
        q1, q3 = X[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        n = int(((X[c] < q1 - 1.5 * iqr) | (X[c] > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
        outliers.append({"feature": c, "label": feature_vi(c), "n": n, "share": n / len(X)})
    outliers.sort(key=lambda d: -d["n"])
    return {
        "n_rows": int(len(df)), "n_cols": int(X.shape[1]),
        "missing": int(df.isna().sum().sum()), "duplicates": int(df.duplicated().sum()),
        "class_counts": {c: int((df[TARGET] == c).sum()) for c in CLASSES},
        "type_counts": {"Danh nghĩa (mã số)": len(NOMINAL), "Nhị phân": len(BINARY),
                        "Số": len(num_cols)},
        "correlations": [{"feature": c, "label": feature_vi(c), "r": float(v)} for c, v in corr.items()],
        "near_constant": near_const,
        "outliers": outliers[:10],
        "zero_heavy": [{"feature": c, "label": feature_vi(c), "share": float(v)}
                       for c, v in zero_share.sort_values(ascending=False).items() if v >= 0.5],
        "sem2_zero_enrolled": {c: int(v) for c, v in
                               df[df[SEM2[1]] == 0][TARGET].value_counts().items()},
        "describe": X[num_cols].describe().T.round(2).reset_index().rename(
            columns={"index": "feature"}).assign(label=lambda d: d["feature"].map(feature_vi)
                                                 ).to_dict("records"),
    }


def eda_feature(service: "Service", col: str):
    df = service.df
    if col not in df.columns or col == TARGET:
        raise KeyError(col)
    s = df[col]
    res = {"feature": col, "label": feature_vi(col)}
    if col in NOMINAL or col in BINARY or s.nunique() <= 12:
        vc = s.value_counts()
        top = list(vc.index[:12])
        cats = top + (["Khác"] if len(vc) > 12 else [])
        grp = s.where(s.isin(top), other=-99999)
        ct = pd.crosstab(grp, df[TARGET]).reindex(columns=CLASSES, fill_value=0)
        res["kind"] = "categorical"
        res["categories"] = [
            "Khác" if c == "Khác" else (code_vi(col, c) if col in NOMINAL or col in BINARY else f"{c:g}")
            for c in cats]
        res["series"] = {cls: [int(ct.loc[c if c != "Khác" else -99999, cls]) for c in cats]
                         for cls in CLASSES}
        res["counts"] = [int(vc.get(c, 0)) if c != "Khác" else int(vc.iloc[12:].sum()) for c in cats]
    else:
        lo, hi = float(s.min()), float(s.max())
        if (s % 1 == 0).all() and s.nunique() <= 30:
            bins = np.arange(lo - 0.5, hi + 1.5)  # biến đếm: mỗi giá trị nguyên một cột
        else:
            bins = np.linspace(lo, hi, 26)
        res["kind"] = "numeric"
        res["bins"] = [round(float((a + b) / 2), 2) for a, b in zip(bins[:-1], bins[1:])]
        res["series"] = {cls: np.histogram(s[df[TARGET] == cls], bins=bins)[0].tolist() for cls in CLASSES}
        res["stats"] = {cls: {k: float(v) for k, v in s[df[TARGET] == cls].describe().items()}
                        for cls in CLASSES}
    return res
