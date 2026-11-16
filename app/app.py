"""Ứng dụng web Flask – Hệ thống cảnh báo sớm sinh viên có nguy cơ bỏ học.

Chạy (từ thư mục gốc dự án):  python -m app.app   ->  http://127.0.0.1:5000
"""
import io

import pandas as pd
from flask import Flask, abort, jsonify, render_template, request, send_file

from src.config import CLASS_LABELS_VI, CLASSES, SCENARIO_ORDER, SCENARIOS
from src.labels import FORM_GROUPS, feature_vi
from src.models import MODEL_NAMES

from .services import Service, eda_feature, eda_overview

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB cho file CSV
service = Service()


@app.context_processor
def inject_globals():
    return {"CLASSES": CLASSES, "CLASS_VI": CLASS_LABELS_VI, "MODEL_NAMES": MODEL_NAMES,
            "SCENARIOS": SCENARIOS, "SCENARIO_ORDER": SCENARIO_ORDER, "fvi": feature_vi,
            "nav": request.endpoint}


# ---------------------------------------------------------------- pages
@app.route("/")
def index():
    return render_template("index.html", scenarios=service.scenario_info(),
                           eda=eda_overview(service), setup=service.analysis["setup"])


@app.route("/eda")
def eda():
    cols = [c for c in service.df.columns if c != "Target"]
    return render_template("eda.html", eda=eda_overview(service),
                           columns=[{"name": c, "label": feature_vi(c)} for c in cols])


@app.route("/experiments")
def experiments():
    return render_template("experiments.html", setup=service.analysis["setup"],
                           scenarios=service.scenario_info())


@app.route("/analysis")
def analysis():
    return render_template("analysis.html", scenarios=service.scenario_info())


@app.route("/predict")
def predict_page():
    groups = []
    for key, title, icon, cols in FORM_GROUPS:
        groups.append({"key": key, "title": title, "icon": icon,
                       "fields": [service.field_spec(c) for c in cols]})
    col_scen = {c: [s for s in SCENARIO_ORDER if c in SCENARIOS[s]["columns"]]
                for c in service.meta["columns"]}
    return render_template("predict.html", groups=groups, scenarios=service.scenario_info(),
                           col_scen=col_scen)


@app.route("/batch")
def batch_page():
    return render_template("batch.html", scenarios=service.scenario_info())


@app.route("/playground")
def playground():
    return render_template("playground.html", scenarios=service.scenario_info(),
                           setup=service.analysis["setup"])


@app.route("/about")
def about():
    return render_template("about.html", setup=service.analysis["setup"],
                           scenarios=service.scenario_info())


# ---------------------------------------------------------------- API
def _scenario(value):
    if value not in SCENARIOS:
        abort(400, description="Kịch bản không hợp lệ")
    return value


@app.post("/api/predict")
def api_predict():
    body = request.get_json(silent=True) or {}
    scenario = _scenario(body.get("scenario", "full"))
    features = body.get("features") or {}
    errors = service.validate(features, scenario)
    if errors:
        return jsonify({"errors": errors, "labels": {k: feature_vi(k) for k in errors}}), 422
    return jsonify(service.predict(features, scenario))


@app.get("/api/sample")
def api_sample():
    scenario = _scenario(request.args.get("scenario", "full"))
    cls = request.args.get("cls") or None
    if cls and cls not in CLASSES:
        abort(400)
    return jsonify(service.sample(scenario, cls))


@app.get("/api/eda/feature")
def api_eda_feature():
    try:
        return jsonify(eda_feature(service, request.args.get("name", "")))
    except KeyError:
        abort(404)


@app.get("/api/eda/overview")
def api_eda_overview():
    return jsonify(eda_overview(service))


@app.get("/api/experiments")
def api_experiments():
    return jsonify({"rows": service.experiments, "deployed": service.analysis["deployed"],
                    "setup": service.analysis["setup"]})


@app.get("/api/hparams")
def api_hparams():
    s = _scenario(request.args.get("scenario", "full"))
    flow = request.args.get("flow", "B")
    model = request.args.get("model", "knn")
    if flow not in ("A", "B") or model not in MODEL_NAMES:
        abort(400)
    from src.config import RESULTS_DIR
    df = pd.read_csv(RESULTS_DIR / "hparams" / f"{s}_{flow}_{model}.csv")
    return jsonify(df.astype(object).where(df.notna(), None).to_dict("records"))


@app.get("/api/analysis")
def api_analysis():
    s = _scenario(request.args.get("scenario", "full"))
    from src.config import RESULTS_DIR
    errs = pd.read_csv(RESULTS_DIR / f"errors_{s}.csv").head(40)
    imp = [dict(d, label=feature_vi(d["feature"])) for d in service.analysis["importance"][s]]
    e = dict(service.analysis["errors"][s])
    from src.labels import code_vi
    e["by_course"] = [dict(r, label=code_vi("Course", float(r["group"]))) for r in e["by_course"]]
    errs["Course"] = errs["Course"].map(lambda v: code_vi("Course", v))
    return jsonify({"importance": imp, "errors": e,
                    "mistakes": errs.astype(object).where(errs.notna(), None).to_dict("records"),
                    "deployed": service.analysis["deployed"][s]})


@app.post("/api/batch")
def api_batch():
    scenario = _scenario(request.form.get("scenario", "full"))
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "Chưa chọn tệp CSV"}), 400
    raw = f.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    sep = ";" if text.splitlines()[0].count(";") > text.splitlines()[0].count(",") else ","
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep)
        rows = service.predict_batch(df, scenario)
    except Exception as e:  # trả lỗi thân thiện cho người dùng
        return jsonify({"error": str(e)}), 400
    return jsonify({"rows": rows, "scenario": scenario})


@app.get("/api/batch/template")
def api_batch_template():
    """Tệp mẫu: 20 dòng ngẫu nhiên từ tập test (giữ cột Target để đối chiếu)."""
    sample = service.X_test.sample(20, random_state=1).copy()
    sample["Target"] = service.y_test.loc[sample.index]
    buf = io.BytesIO(sample.to_csv(index=False, sep=";").encode("utf-8-sig"))
    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="mau_du_lieu.csv")


@app.post("/api/playground")
def api_playground():
    body = request.get_json(silent=True) or {}
    scenario = _scenario(body.get("scenario", "full"))
    try:
        return jsonify(service.playground(scenario, body.get("flow", "B"), body.get("model", "dt"),
                                          body.get("params") or {}, bool(body.get("cv"))))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.get("/api/test-saved")
def api_test_saved():
    from src.test_models import evaluate_saved
    return jsonify(evaluate_saved())


@app.errorhandler(400)
@app.errorhandler(404)
def handle_err(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": getattr(e, "description", str(e))}), e.code
    return render_template("error.html", code=e.code, message=e.description), e.code


if __name__ == "__main__":
    app.run(debug=True, port=5000)
