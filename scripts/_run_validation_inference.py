#!/usr/bin/env python3
"""
Standalone script that reproduces what notebooks/08_validation_inference.ipynb does
but runs in the current Python process (no Jupyter kernel needed).

Run from the repo root:
    python scripts/_run_validation_inference.py
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from src.inference.predict import load_pipeline, predict_single

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HANDWRITTEN_XLSX = ROOT / "data" / "handwritten_validation_set.xlsx"
REGULAR_XLSX     = ROOT / "data" / "regular_documents_validation_set.xlsx"
HANDWRITTEN_DIR  = ROOT / "data" / "validation_set" / "handwritten"
REGULAR_DIR      = ROOT / "data" / "validation_set" / "regular_documents"
REPORT_DIR       = ROOT / "validation_report"
REPORT_DIR.mkdir(exist_ok=True)

MODELS = {
    "ResNet50": {
        "checkpoint":  ROOT / "checkpoints" / "baseline" / "best_resnet50.pt",
        "calibrator":  ROOT / "checkpoints" / "baseline" / "calibrator_resnet50.pkl",
        "model_type": "resnet50",
        "config":      ROOT / "configs" / "baseline.yaml",
    },
    "ViT-Base": {
        "checkpoint":  ROOT / "checkpoints" / "baseline" / "best_vit_base_patch16_224.pt",
        "calibrator":  ROOT / "checkpoints" / "baseline" / "calibrator_vit_base_patch16_224.pkl",
        "model_type": "vit",
        "config":      ROOT / "configs" / "baseline.yaml",
    },
    "DiT": {
        "checkpoint":  ROOT / "checkpoints" / "dit" / "best_model.pt",
        "calibrator":  ROOT / "checkpoints" / "dit" / "calibrator.pkl",
        "model_type": "dit",
        "config":      ROOT / "configs" / "dit.yaml",
    },
}

N_WORST = 10

# ---------------------------------------------------------------------------
# 1. Load manifests
# ---------------------------------------------------------------------------
print("Loading manifests ...")
df_hw = pd.read_excel(HANDWRITTEN_XLSX)[["file name"]].rename(columns={"file name": "filename"})
df_hw["true_label"] = 1
df_hw["source"]     = "handwritten"
df_hw["pdf_path"]   = df_hw["filename"].apply(lambda f: HANDWRITTEN_DIR / f)

df_reg = pd.read_excel(REGULAR_XLSX)[["file name"]].rename(columns={"file name": "filename"})
df_reg["true_label"] = 0
df_reg["source"]     = "regular_documents"
df_reg["pdf_path"]   = df_reg["filename"].apply(lambda f: REGULAR_DIR / f)

df_all = pd.concat([df_hw, df_reg], ignore_index=True)
df_all["file_exists"] = df_all["pdf_path"].apply(lambda p: p.exists())
missing = df_all[~df_all["file_exists"]]
print(f"  Total: {len(df_all)}  Found: {df_all['file_exists'].sum()}  Missing: {len(missing)}")

# ---------------------------------------------------------------------------
# 2. Inference per model
# ---------------------------------------------------------------------------
all_results: dict[str, list[dict]] = {}

for model_name, cfg in MODELS.items():
    print(f"\n=== {model_name} ===")
    mdl, calibrator, thresholds, device = load_pipeline(
        checkpoint_path=str(cfg["checkpoint"]),
        calibrator_path=str(cfg["calibrator"]),
        model_type=cfg["model_type"],
        config_path=str(cfg["config"]),
    )
    print(f"  device={device}  T_low={thresholds['T_low']:.3f}  "
          f"T_high={thresholds['T_high']:.3f}  T={calibrator.temperature:.4f}")

    results = []
    for _, row in tqdm(df_all.iterrows(), total=len(df_all), desc=f"  {model_name}"):
        if not row["file_exists"]:
            results.append({
                "filename": row["filename"], "source": row["source"],
                "true_label": row["true_label"],
                "probability": float("nan"), "predicted_category": "error",
                "raw_logit": float("nan"), "error": "file_not_found",
            })
            continue
        try:
            resp = predict_single(
                pdf_path=str(row["pdf_path"]),
                model=mdl, calibrator=calibrator,
                thresholds=thresholds, dpi=150, device=device,
            )
            results.append({
                "filename": row["filename"], "source": row["source"],
                "true_label": row["true_label"],
                "probability": resp.confidence,
                "predicted_category": resp.risk_category.value,
                "raw_logit": resp.raw_logit, "error": None,
            })
        except Exception as exc:
            results.append({
                "filename": row["filename"], "source": row["source"],
                "true_label": row["true_label"],
                "probability": float("nan"), "predicted_category": "error",
                "raw_logit": float("nan"), "error": str(exc),
            })

    all_results[model_name] = results
    rdf = pd.DataFrame(results)
    print(f"  Completed: {len(rdf)}  Errors: {rdf['error'].notna().sum()}")
    del mdl
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

# ---------------------------------------------------------------------------
# 3. Metrics
# ---------------------------------------------------------------------------
model_metrics: dict[str, dict] = {}

for model_name, results in all_results.items():
    rdf = pd.DataFrame(results)
    valid_df = rdf[rdf["error"].isna()].copy()

    ck = torch.load(str(MODELS[model_name]["checkpoint"]), map_location="cpu", weights_only=False)
    T_LOW  = ck["thresholds"]["T_low"]
    T_HIGH = ck["thresholds"]["T_high"]

    y_true = valid_df["true_label"].values
    y_prob = valid_df["probability"].values
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred,
                               target_names=["safe (0)", "risky (1)"],
                               output_dict=True)
    risky_mask = y_true == 1
    fsr = float(((y_prob < T_LOW) & risky_mask).sum() / risky_mask.sum())

    safe_df  = valid_df[valid_df["true_label"] == 0].sort_values("probability", ascending=False)
    risky_df = valid_df[valid_df["true_label"] == 1].sort_values("probability", ascending=True)
    worst_fp = safe_df.head(N_WORST).copy()
    worst_fp["true_label_name"] = "safe"
    worst_fn = risky_df.head(N_WORST).copy()
    worst_fn["true_label_name"] = "risky"

    model_metrics[model_name] = {
        "valid_df": valid_df, "y_true": y_true, "y_prob": y_prob, "y_pred": y_pred,
        "cm": cm, "cr": cr, "T_LOW": T_LOW, "T_HIGH": T_HIGH, "fsr": fsr,
        "worst_fp": worst_fp, "worst_fn": worst_fn,
    }

    tn, fp, fn, tp = cm.ravel()
    acc = float((y_pred == y_true).mean())
    print(f"{model_name}: acc={acc:.1%}  FSR={fsr:.1%}  TN={tn} FP={fp} FN={fn} TP={tp}")

# ---------------------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------------------
def build_confusion_matrix_fig(cm, model_name):
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    z    = [[tn, fp], [fn, tp]]
    text = [
        [f"<b>{tn}</b><br>({tn/total:.1%})", f"<b>{fp}</b><br>({fp/total:.1%})"],
        [f"<b>{fn}</b><br>({fn/total:.1%})", f"<b>{tp}</b><br>({tp/total:.1%})"],
    ]
    fig = go.Figure(go.Heatmap(
        z=z, x=["Predicted: Safe", "Predicted: Risky"],
        y=["True: Safe", "True: Risky"],
        text=text, texttemplate="%{text}",
        textfont={"size": 18}, colorscale="Blues", showscale=False,
        hovertemplate="%{y} → %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"Confusion Matrix — {model_name}", font=dict(size=20), x=0.5),
        xaxis=dict(title="Predicted Label", side="bottom", tickfont=dict(size=13)),
        yaxis=dict(title="True Label", tickfont=dict(size=13)),
        height=460, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=80, r=40, t=80, b=60),
    )
    return fig


def build_classification_report_fig(cr, model_name):
    rows_to_show = ["safe (0)", "risky (1)", "macro avg", "weighted avg"]
    cr_rows = {k: cr[k] for k in rows_to_show if k in cr}
    header_vals = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    cell_vals = [
        list(cr_rows.keys()),
        [f"{v['precision']:.3f}" for v in cr_rows.values()],
        [f"{v['recall']:.3f}"    for v in cr_rows.values()],
        [f"{v['f1-score']:.3f}"  for v in cr_rows.values()],
        [int(v["support"])       for v in cr_rows.values()],
    ]
    n = len(cr_rows)
    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{h}</b>" for h in header_vals],
            fill_color="#1a1a2e", font=dict(color="white", size=14),
            align="center", height=36,
        ),
        cells=dict(
            values=cell_vals,
            fill_color=[["#f8f9fa" if i % 2 == 0 else "white" for i in range(n)]] * 5,
            font=dict(color="#222", size=13), align="center", height=32,
        ),
    ))
    fig.update_layout(
        title=dict(text=f"Classification Report — {model_name}", font=dict(size=20), x=0.5),
        height=300, paper_bgcolor="white", margin=dict(l=20, r=20, t=80, b=20),
    )
    return fig


def build_histogram_fig(y_true, y_prob, T_LOW, T_HIGH, model_name):
    safe_probs  = y_prob[y_true == 0]
    risky_probs = y_prob[y_true == 1]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=safe_probs, name="Safe (label=0)", nbinsx=30, opacity=0.75,
        marker_color="#2563eb",
        hovertemplate="Prob: %{x:.2f}<br>Count: %{y}<extra>Safe</extra>",
    ))
    fig.add_trace(go.Histogram(
        x=risky_probs, name="Risky (label=1)", nbinsx=30, opacity=0.75,
        marker_color="#dc2626",
        hovertemplate="Prob: %{x:.2f}<br>Count: %{y}<extra>Risky</extra>",
    ))
    fig.add_vline(x=T_LOW,  line_dash="dash", line_color="#f59e0b", line_width=2,
                  annotation_text=f"T_low={T_LOW:.3f}",
                  annotation_position="top right", annotation_font_size=11)
    fig.add_vline(x=T_HIGH, line_dash="dot",  line_color="#7c3aed", line_width=2,
                  annotation_text=f"T_high={T_HIGH:.3f}",
                  annotation_position="top right", annotation_font_size=11)
    fig.update_layout(
        title=dict(text=f"Probability Distribution — {model_name}", font=dict(size=20), x=0.5),
        xaxis=dict(title="Risk Probability", range=[0, 1], tickfont=dict(size=13)),
        yaxis=dict(title="Count", tickfont=dict(size=13)),
        barmode="overlay", legend=dict(font=dict(size=13), x=0.75, y=0.95),
        height=480, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=60, r=40, t=80, b=60),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb")
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb")
    return fig


def build_worst_cases_fig(worst_fp, worst_fn, model_name):
    def wc_table(df, color):
        n = len(df)
        return go.Table(
            header=dict(
                values=[f"<b>{h}</b>" for h in
                        ["Filename", "Source", "True Label", "Risk Probability", "Predicted Category"]],
                fill_color="#1a1a2e", font=dict(color="white", size=13),
                align=["left", "center", "center", "center", "center"], height=34,
            ),
            cells=dict(
                values=[
                    df["filename"].tolist(), df["source"].tolist(),
                    df["true_label_name"].tolist(),
                    [f"{p:.3f}" for p in df["probability"].tolist()],
                    df["predicted_category"].tolist(),
                ],
                fill_color=[[
                    "#fef2f2" if i % 2 == 0 else "white" for i in range(n)
                ] if color == "red" else [
                    "#eff6ff" if i % 2 == 0 else "white" for i in range(n)
                ]] * 5,
                font=dict(color="#222", size=12),
                align=["left", "center", "center", "center", "center"], height=30,
            ),
        )
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            f"False Positives — Safe docs with Highest Risk Score (top {N_WORST})",
            f"False Negatives — Risky docs with Lowest Risk Score (top {N_WORST})",
        ],
        vertical_spacing=0.12,
        specs=[[{"type": "table"}], [{"type": "table"}]],
    )
    fig.add_trace(wc_table(worst_fp, "red"),  row=1, col=1)
    fig.add_trace(wc_table(worst_fn, "blue"), row=2, col=1)
    fig.update_layout(
        title=dict(text=f"Worst Case Predictions — {model_name}", font=dict(size=20), x=0.5),
        height=800, paper_bgcolor="white", margin=dict(l=20, r=20, t=100, b=20),
    )
    return fig


model_figs: dict[str, dict] = {}
for model_name, m in model_metrics.items():
    model_figs[model_name] = {
        "confusion_matrix":      build_confusion_matrix_fig(m["cm"], model_name),
        "classification_report": build_classification_report_fig(m["cr"], model_name),
        "probability_histogram": build_histogram_fig(
            m["y_true"], m["y_prob"], m["T_LOW"], m["T_HIGH"], model_name),
        "worst_cases":           build_worst_cases_fig(m["worst_fp"], m["worst_fn"], model_name),
    }
    print(f"{model_name}: figures built")

# ---------------------------------------------------------------------------
# 5. Export report_data.json
# ---------------------------------------------------------------------------
def fig_to_dict(fig):
    return json.loads(pio.to_json(fig))


report_data: dict = {"models": {}}

for model_name, m in model_metrics.items():
    tn, fp, fn, tp = m["cm"].ravel()
    acc = float((m["y_pred"] == m["y_true"]).mean())
    # Extra: confusion matrix at the operational T_low (cost-weighted τ*)
    y_prob_arr = np.asarray(m["y_prob"], dtype=float)
    y_true_arr = np.asarray(m["y_true"], dtype=int)
    pred_tlow = (y_prob_arr >= m["T_LOW"]).astype(int)
    tp_t = int(((pred_tlow == 1) & (y_true_arr == 1)).sum())
    tn_t = int(((pred_tlow == 0) & (y_true_arr == 0)).sum())
    fp_t = int(((pred_tlow == 1) & (y_true_arr == 0)).sum())
    fn_t = int(((pred_tlow == 0) & (y_true_arr == 1)).sum())
    acc_t = float((pred_tlow == y_true_arr).mean())

    report_data["models"][model_name] = {
        "summary": {
            "total":           int(len(m["valid_df"])),
            "n_safe":          int((m["valid_df"]["true_label"] == 0).sum()),
            "n_risky":         int((m["valid_df"]["true_label"] == 1).sum()),
            "accuracy":        acc,
            "false_safe_rate": m["fsr"],
            "t_low":           float(m["T_LOW"]),
            "t_high":          float(m["T_HIGH"]),
            "threshold_for_binary": 0.5,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "accuracy_at_tlow": acc_t,
            "tn_at_tlow": tn_t, "fp_at_tlow": fp_t,
            "fn_at_tlow": fn_t, "tp_at_tlow": tp_t,
        },
        "per_doc": {
            "y_true":      y_true_arr.tolist(),
            "y_prob":      y_prob_arr.tolist(),
            "source":      m["valid_df"]["source"].tolist(),
        },
        "confusion_matrix":      fig_to_dict(model_figs[model_name]["confusion_matrix"]),
        "classification_report": fig_to_dict(model_figs[model_name]["classification_report"]),
        "probability_histogram": fig_to_dict(model_figs[model_name]["probability_histogram"]),
        "worst_cases":           fig_to_dict(model_figs[model_name]["worst_cases"]),
    }

json_path = REPORT_DIR / "report_data.json"
with open(json_path, "w", encoding="utf-8") as fh:
    json.dump(report_data, fh, ensure_ascii=False, indent=2)
print(f"\nreport_data.json → {json_path}  ({json_path.stat().st_size / 1024:.0f} KB)")

# ---------------------------------------------------------------------------
# 6. Build HTML
# ---------------------------------------------------------------------------
HTML_TEMPLATE = open(Path(__file__).parent / "_html_template_nb08.html", encoding="utf-8").read()

report_json_str = json.dumps(report_data, ensure_ascii=False)
html_content = HTML_TEMPLATE.replace("__REPORT_DATA__", report_json_str)

html_path = REPORT_DIR / "index.html"
with open(html_path, "w", encoding="utf-8") as fh:
    fh.write(html_content)
print(f"index.html → {html_path}  ({html_path.stat().st_size / 1024:.0f} KB)")

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
print("\n=== External Validation Summary — All Models ===")
print(f"  {'Model':<12} {'Acc':>7} {'Macro F1':>9} {'FSR':>8}   CM")
print("  " + "-" * 55)
for model_name, m in model_metrics.items():
    tn, fp, fn, tp = m["cm"].ravel()
    acc = float((m["y_pred"] == m["y_true"]).mean())
    f1  = m["cr"].get("macro avg", {}).get("f1-score", float("nan"))
    print(f"  {model_name:<12} {acc:>6.1%}  {f1:>8.3f} {m['fsr']:>7.1%}   "
          f"TN={tn} FP={fp} FN={fn} TP={tp}")
print(f"\nReport: {html_path.resolve()}")
print("Open index.html in a browser and use the model dropdown to compare ResNet50 / ViT-Base / DiT.")
