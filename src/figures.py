"""Sinh hình PNG phục vụ báo cáo LaTeX (results/figures)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CLASSES, FIGURES_DIR, RESULTS_DIR, SCENARIO_ORDER, SCENARIOS, SEM1, SEM2, TARGET

COLORS = {"Dropout": "#d9534f", "Enrolled": "#f0ad4e", "Graduate": "#2e8b57"}
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9})


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def eda_figures(df):
    counts = df[TARGET].value_counts().reindex(CLASSES)
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.bar(counts.index, counts.values, color=[COLORS[c] for c in CLASSES])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 30, f"{v}\n({v / len(df):.0%})", ha="center")
    ax.set_title("Phân phối nhãn")
    ax.set_ylim(0, counts.max() * 1.25)
    _save(fig, "eda_target_distribution")

    feats = ["Admission grade", "Age at enrollment", SEM1[4], SEM2[4], SEM1[3], SEM2[3]]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5))
    for ax, c in zip(axes.ravel(), feats):
        for cls in CLASSES:
            ax.hist(df.loc[df[TARGET] == cls, c], bins=25, alpha=0.55, color=COLORS[cls],
                    label=cls, density=True)
        ax.set_title(c, fontsize=8)
    axes[0, 0].legend(fontsize=7)
    _save(fig, "eda_histograms_by_target")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, c in zip(axes, ["Age at enrollment", "Admission grade", SEM1[1]]):
        ax.boxplot([df.loc[df[TARGET] == k, c] for k in CLASSES], tick_labels=CLASSES)
        ax.set_title(c, fontsize=8)
    _save(fig, "eda_boxplots")

    y = df[TARGET].map({"Dropout": 0, "Enrolled": 1, "Graduate": 2})
    num = df.drop(columns=[TARGET])
    top = num.corrwith(y).abs().sort_values(ascending=False).head(12).index
    corr = df[top].corr()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(top)), [t.replace("Curricular units ", "CU ") for t in top],
                  rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(top)), [t.replace("Curricular units ", "CU ") for t in top], fontsize=7)
    for i in range(len(top)):
        for j in range(len(top)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Tương quan giữa 12 đặc trưng liên quan nhất đến nhãn")
    _save(fig, "eda_correlation")


def result_figures(rows, analysis):
    res = pd.DataFrame(rows)
    # Luồng A vs B (kịch bản full, mô hình tinh chỉnh)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, scen in zip(axes, SCENARIO_ORDER):
        sub = res[(res.scenario == scen) & (res.config == "tuned")]
        x = np.arange(3)
        for k, flow in enumerate(("A", "B")):
            vals = [sub[(sub.flow == flow) & (sub.model == m)]["f1_macro"].iloc[0] for m in ("knn", "nb", "dt")]
            ax.bar(x + (k - 0.5) * 0.38, vals, 0.38, label=f"Luồng {flow}",
                   color=["#6c8ebf", "#2e6da4"][k])
            for xi, v in zip(x, vals):
                ax.text(xi + (k - 0.5) * 0.38, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
        base = res[(res.scenario == scen) & (res.model == "dummy")]["f1_macro"].iloc[0]
        ax.axhline(base, ls="--", color="gray", lw=1, label="Dummy")
        ax.set_xticks(x, ["KNN", "NB", "DT"])
        ax.set_title(SCENARIOS[scen]["name"])
        ax.set_ylim(0, 0.9)
    axes[0].set_ylabel("Macro-F1 (test)")
    axes[0].legend(fontsize=7)
    _save(fig, "result_flow_comparison")

    # Mặc định vs tinh chỉnh (full)
    sub = res[(res.scenario == "full") & res.config.isin(["default", "tuned"])]
    labels, d, t = [], [], []
    for flow in ("A", "B"):
        for m in ("knn", "nb", "dt"):
            s = sub[(sub.flow == flow) & (sub.model == m)]
            labels.append(f"{m.upper()}-{flow}")
            d.append(s[s.config == "default"]["f1_macro"].iloc[0])
            t.append(s[s.config == "tuned"]["f1_macro"].iloc[0])
    fig, ax = plt.subplots(figsize=(7, 3))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, d, 0.4, label="Mặc định", color="#bbb")
    ax.bar(x + 0.2, t, 0.4, label="Tinh chỉnh", color="#2e6da4")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Macro-F1 (test)")
    ax.legend()
    ax.set_title("Ảnh hưởng của tinh chỉnh siêu tham số (sau năm 1)")
    _save(fig, "result_default_vs_tuned")

    # Confusion matrix của mô hình triển khai
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, scen in zip(axes, SCENARIO_ORDER):
        dep = analysis["deployed"][scen]
        r = res[(res.scenario == scen) & (res.flow == dep["flow"]) & (res.model == dep["model"]) &
                (res.config == "tuned")].iloc[0]
        cm = np.array(r["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks(range(3), CLASSES)
        ax.set_yticks(range(3), CLASSES)
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Thực tế")
        ax.set_title(f"{SCENARIOS[scen]['name']}\n{dep['model'].upper()}-{dep['flow']} | F1={dep['f1_macro']:.3f}",
                     fontsize=8)
    _save(fig, "result_confusion_matrices")

    # Đường siêu tham số
    hp = RESULTS_DIR / "hparams"
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for flow, color in (("A", "#6c8ebf"), ("B", "#2e6da4")):
        k = pd.read_csv(hp / f"full_{flow}_knn.csv")
        k = k[(k.weights == "distance") & (k.p == 1)].sort_values("n_neighbors")
        axes[0].errorbar(k.n_neighbors, k.mean_f1_macro, k.std_f1_macro, marker="o", label=f"Luồng {flow}",
                         color=color, capsize=3)
        t = pd.read_csv(hp / f"full_{flow}_dt.csv")
        t["max_depth"] = t["max_depth"].fillna(25)
        t = t.groupby("max_depth")["mean_f1_macro"].max()
        axes[1].plot(t.index, t.values, marker="o", label=f"Luồng {flow}", color=color)
    axes[0].set_xlabel("k (weights=distance, p=1)")
    axes[0].set_ylabel("CV Macro-F1")
    axes[0].set_title("K-NN: ảnh hưởng của k")
    axes[1].set_xlabel("max_depth (25 = không giới hạn)")
    axes[1].set_title("Decision Tree: ảnh hưởng của độ sâu (tốt nhất theo các tham số khác)")
    for ax in axes:
        ax.legend()
    _save(fig, "result_hyperparameters")

    # Độ quan trọng đặc trưng
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, scen in zip(axes, SCENARIO_ORDER):
        imp = analysis["importance"][scen][:10][::-1]
        ax.barh([d["feature"].replace("Curricular units ", "CU ") for d in imp],
                [d["mean"] for d in imp], xerr=[d["std"] for d in imp], color="#2e6da4")
        ax.set_title(SCENARIOS[scen]["name"], fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("Permutation importance (giảm Macro-F1 khi xáo trộn đặc trưng)")
    _save(fig, "result_permutation_importance")


def make_all(df, rows, analysis):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    eda_figures(df)
    result_figures(rows, analysis)
