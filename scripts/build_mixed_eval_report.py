#!/usr/bin/env python3
"""Build a self-contained HTML report comparing Gemini vs DiT classification
on the mixed_results evaluation set, styled like reports/finetune_report.html."""

import json, math
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "eval" / "mixed_results - Sheet1.csv"
OUT_PATH = ROOT / "reports" / "mixed_eval_report.html"


def load_and_binarise(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["classification_gemini"]).copy()

    df["y_true"] = (df["expected output"] == "high_hallucination_risk").astype(int)

    df["y_gemini"] = df["classification_gemini"].map(
        {"typed": 0, "handwritten": 1, "mixed": 1}
    ).astype(int)

    df["y_dit"] = df["classification_idan"].map(
        {"safe_for_extraction": 0, "high_hallucination_risk": 1, "review": 1}
    ).astype(int)

    return df


def confusion(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return dict(tp=tp, tn=tn, fp=fp, fn=fn)


def metrics(y_true, y_pred, confidence=None):
    cm = confusion(y_true, y_pred)
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    n = tp + tn + fp + fn

    accuracy = (tp + tn) / n if n else 0
    precision_risky = tp / (tp + fp) if (tp + fp) else 0
    recall_risky = tp / (tp + fn) if (tp + fn) else 0
    precision_safe = tn / (tn + fn) if (tn + fn) else 0
    recall_safe = tn / (tn + fp) if (tn + fp) else 0
    f1_risky = 2 * precision_risky * recall_risky / (precision_risky + recall_risky) if (precision_risky + recall_risky) else 0
    f1_safe = 2 * precision_safe * recall_safe / (precision_safe + recall_safe) if (precision_safe + recall_safe) else 0
    macro_f1 = (f1_risky + f1_safe) / 2

    fsr = fn / (tn + fn) if (tn + fn) else 0
    false_flag_rate = fp / (tp + fp) if (tp + fp) else 0

    result = dict(
        n=n, accuracy=accuracy,
        precision_risky=precision_risky, recall_risky=recall_risky, f1_risky=f1_risky,
        precision_safe=precision_safe, recall_safe=recall_safe, f1_safe=f1_safe,
        macro_f1=macro_f1,
        fsr=fsr, false_flag_rate=false_flag_rate,
        **cm,
    )

    if confidence is not None:
        conf_correct = confidence[(y_true == y_pred)].mean()
        conf_wrong = confidence[(y_true != y_pred)].mean()
        result["avg_confidence_correct"] = float(conf_correct) if not np.isnan(conf_correct) else None
        result["avg_confidence_wrong"] = float(conf_wrong) if not np.isnan(conf_wrong) else None

    return result


def build_confusion_plotly(cm, title, model_color):
    labels = ["Safe", "Risky"]
    z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    text = [
        [f"TN={cm['tn']}", f"FP={cm['fp']}"],
        [f"FN={cm['fn']}", f"TP={cm['tp']}"],
    ]
    data = [dict(
        type="heatmap", z=z, x=["Pred Safe", "Pred Risky"], y=["True Safe", "True Risky"],
        text=text, texttemplate="%{text}", textfont=dict(size=16, color="white"),
        colorscale=[[0, "#dbeafe"], [1, model_color]], showscale=False,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    )]
    layout = dict(
        title=dict(text=title, font=dict(size=15), x=0.5),
        xaxis=dict(title="Predicted", side="bottom"),
        yaxis=dict(title="Actual", autorange="reversed"),
        height=360, margin=dict(l=80, r=30, t=60, b=60),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return dict(data=data, layout=layout)


def build_error_table_data(df, pred_col, model_name):
    """Collect false-negatives and false-positives for the worst-cases chart."""
    fn_mask = (df["y_true"] == 1) & (df[pred_col] == 0)
    fp_mask = (df["y_true"] == 0) & (df[pred_col] == 1)

    fn_df = df[fn_mask][["file", "classification_gemini", "classification_idan", "expected output"]].copy()
    fn_df["error_type"] = "FN (risky→safe)"
    fp_df = df[fp_mask][["file", "classification_gemini", "classification_idan", "expected output"]].copy()
    fp_df["error_type"] = "FP (safe→risky)"

    errors = pd.concat([fn_df, fp_df], ignore_index=True)
    return errors.to_dict(orient="records")


def build_agreement_data(df):
    """Compute agreement/disagreement stats between models."""
    both_correct = int(((df["y_gemini"] == df["y_true"]) & (df["y_dit"] == df["y_true"])).sum())
    gemini_only = int(((df["y_gemini"] == df["y_true"]) & (df["y_dit"] != df["y_true"])).sum())
    dit_only = int(((df["y_gemini"] != df["y_true"]) & (df["y_dit"] == df["y_true"])).sum())
    both_wrong = int(((df["y_gemini"] != df["y_true"]) & (df["y_dit"] != df["y_true"])).sum())
    return dict(both_correct=both_correct, gemini_only=gemini_only,
                dit_only=dit_only, both_wrong=both_wrong, total=len(df))


def build_confidence_histogram(df):
    """Confidence histogram for Gemini, coloured by correctness."""
    correct = df[df["y_gemini"] == df["y_true"]]["Gemini Confidence"].dropna().tolist()
    wrong = df[df["y_gemini"] != df["y_true"]]["Gemini Confidence"].dropna().tolist()

    data = [
        dict(type="histogram", x=correct, name="Correct", opacity=0.7,
             marker=dict(color="#059669"), nbinsx=20),
        dict(type="histogram", x=wrong, name="Wrong", opacity=0.7,
             marker=dict(color="#dc2626"), nbinsx=20),
    ]
    layout = dict(
        title=dict(text="Gemini Confidence Distribution by Correctness", font=dict(size=16), x=0.5),
        xaxis=dict(title="Confidence (%)", showgrid=True, gridcolor="#e5e7eb"),
        yaxis=dict(title="Count", showgrid=True, gridcolor="#e5e7eb"),
        barmode="overlay", height=400,
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=60, r=30, t=60, b=80),
    )
    return dict(data=data, layout=layout)


def build_per_class_breakdown(df):
    """Breakdown by original label classes for each model."""
    result = {}
    for model_name, pred_col, orig_col in [
        ("Gemini", "y_gemini", "classification_gemini"),
        ("DiT", "y_dit", "classification_idan"),
    ]:
        rows = []
        for cls in df[orig_col].dropna().unique():
            mask = df[orig_col] == cls
            sub = df[mask]
            correct = int((sub["y_true"] == sub[pred_col]).sum())
            total = len(sub)
            acc = correct / total if total else 0

            n_risky_true = int(sub["y_true"].sum())
            n_safe_true = int((sub["y_true"] == 0).sum())

            rows.append(dict(
                original_class=cls, n=total,
                n_risky_true=n_risky_true, n_safe_true=n_safe_true,
                correct=correct, accuracy=acc,
            ))
        rows.sort(key=lambda r: r["n"], reverse=True)
        result[model_name] = rows
    return result


def build_report_data(df):
    """Assemble the full REPORT JSON object."""
    gemini_metrics = metrics(df["y_true"].values, df["y_gemini"].values, df["Gemini Confidence"].values)
    dit_metrics = metrics(df["y_true"].values, df["y_dit"].values)

    gemini_color = "#2563eb"
    dit_color = "#dc2626"

    report = dict(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        n_total=len(df),
        n_safe=int((df["y_true"] == 0).sum()),
        n_risky=int((df["y_true"] == 1).sum()),
        n_dropped=4,

        class_distribution=dict(
            expected=df["expected output"].value_counts().to_dict(),
            gemini=df["classification_gemini"].value_counts().to_dict(),
            dit=df["classification_idan"].value_counts().to_dict(),
        ),

        metrics=dict(Gemini=gemini_metrics, DiT=dit_metrics),

        confusion_figs=dict(
            Gemini=build_confusion_plotly(gemini_metrics, "Gemini", gemini_color),
            DiT=build_confusion_plotly(dit_metrics, "DiT", dit_color),
        ),

        agreement=build_agreement_data(df),

        gemini_confidence_hist=build_confidence_histogram(df),

        per_class=build_per_class_breakdown(df),

        errors=dict(
            Gemini=build_error_table_data(df, "y_gemini", "Gemini"),
            DiT=build_error_table_data(df, "y_dit", "DiT"),
        ),
    )
    return report


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mixed Results Evaluation — Gemini vs DiT</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  :root {
    --bg: #f4f6fb; --surface: #ffffff; --primary: #1a1a2e;
    --accent: #2563eb; --accent-strong: #1d4ed8;
    --muted: #6b7280; --border: #e5e7eb;
    --good: #059669; --bad: #dc2626; --warn: #d97706;
    --gemini: #2563eb; --dit: #dc2626;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--primary); min-height: 100vh; line-height: 1.55;
  }
  code { font-size: 0.85em; background: #f1f5f9; padding: 2px 6px; border-radius: 4px; }
  header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: white; padding: 32px 48px 28px; border-bottom: 3px solid var(--accent);
  }
  header .eyebrow {
    font-size: 0.72rem; color: #60a5fa; text-transform: uppercase;
    letter-spacing: 1.5px; font-weight: 600; margin-bottom: 6px;
  }
  header h1 { font-size: 1.9rem; font-weight: 700; letter-spacing: -0.5px; }
  header .subtitle { font-size: 0.95rem; color: #cbd5e1; margin-top: 6px; max-width: 900px; }
  header .meta { font-size: 0.78rem; color: #94a3b8; margin-top: 14px; }

  .tabs {
    display: flex; gap: 4px; padding: 18px 48px 0;
    background: var(--bg); border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 10;
  }
  .tab-btn {
    padding: 11px 22px; border: 1px solid var(--border); border-bottom: none;
    background: #f3f4f6; color: var(--muted); font-size: 0.875rem;
    font-weight: 500; cursor: pointer; border-radius: 8px 8px 0 0; transition: all 0.15s;
  }
  .tab-btn:hover { background: #e0e7ff; color: var(--accent); }
  .tab-btn.active {
    background: var(--surface); color: var(--accent); font-weight: 600;
    box-shadow: 0 -2px 0 var(--accent);
  }

  .pages { padding: 32px 48px 48px; max-width: 1400px; margin: 0 auto; }
  .page { display: none; }
  .page.active { display: block; }

  .card {
    background: var(--surface); border-radius: 12px; border: 1px solid var(--border);
    box-shadow: 0 1px 4px rgba(0,0,0,0.05); padding: 28px 32px; margin-bottom: 24px;
  }
  .card h2 {
    font-size: 1.25rem; font-weight: 700; color: var(--primary);
    margin-bottom: 16px; border-left: 4px solid var(--accent);
    padding-left: 12px;
  }
  .card h3 {
    font-size: 1.0rem; font-weight: 600; color: var(--primary);
    margin-top: 20px; margin-bottom: 10px;
  }
  .card p { color: #334155; font-size: 0.92rem; margin-bottom: 10px; }
  .card ul, .card ol { margin-left: 24px; color: #334155; font-size: 0.92rem; }
  .card ul li, .card ol li { margin-bottom: 6px; }

  .kpi-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 14px; margin-top: 6px;
  }
  .kpi {
    background: #f8fafc; border: 1px solid var(--border); border-radius: 10px;
    padding: 16px 18px; display: flex; flex-direction: column; gap: 2px;
  }
  .kpi .label {
    font-size: 0.7rem; color: var(--muted); text-transform: uppercase;
    letter-spacing: 0.5px; font-weight: 600;
  }
  .kpi .value { font-size: 1.5rem; font-weight: 700; }
  .kpi .value.gemini-color { color: var(--gemini); }
  .kpi .value.dit-color { color: var(--dit); }
  .kpi .delta { font-size: 0.78rem; font-weight: 600; }
  .kpi .delta.good { color: var(--good); }
  .kpi .delta.bad  { color: var(--bad); }
  .kpi .delta.flat { color: var(--muted); }

  table.data {
    width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 0.9rem;
  }
  table.data th, table.data td {
    padding: 10px 14px; text-align: left;
    border-bottom: 1px solid var(--border);
  }
  table.data th {
    background: #1a1a2e; color: white; font-weight: 600;
    text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.5px;
  }
  table.data tbody tr:nth-child(even) { background: #f8fafc; }
  table.data tbody tr:hover { background: #eff6ff; }
  table.data td.num { text-align: right; font-variant-numeric: tabular-nums; }
  table.data td.center { text-align: center; }
  .best { color: var(--good); font-weight: 700; }
  .worst { color: var(--bad); font-weight: 600; }
  .good-cell { color: var(--good); font-weight: 600; }
  .bad-cell  { color: var(--bad);  font-weight: 600; }

  .callout {
    padding: 14px 18px; border-radius: 10px; margin: 16px 0;
    font-size: 0.9rem; border-left: 4px solid;
  }
  .callout.info { background: #eff6ff; border-color: var(--accent); color: #1e3a8a; }
  .callout.good { background: #ecfdf5; border-color: var(--good);   color: #065f46; }
  .callout.warn { background: #fffbeb; border-color: var(--warn);   color: #92400e; }

  .plot-card {
    background: var(--surface); border-radius: 10px; border: 1px solid var(--border);
    box-shadow: 0 1px 4px rgba(0,0,0,0.05); padding: 24px; margin-bottom: 20px;
  }
  .plot-card .caption {
    margin-top: 10px; font-size: 0.82rem; color: var(--muted); text-align: center;
  }
  .plot-card h3 {
    font-size: 1.0rem; margin: 0 0 14px; color: var(--primary); font-weight: 600;
  }
  .plotly-chart { width: 100%; min-height: 380px; }

  .two-col {
    display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;
  }
  @media (max-width: 900px) {
    .two-col { grid-template-columns: 1fr; }
  }

  .model-selector {
    display: flex; gap: 8px; margin-bottom: 18px; flex-wrap: wrap;
  }
  .model-btn {
    padding: 8px 16px; border: 1px solid var(--border); background: var(--surface);
    color: var(--muted); font-size: 0.85rem; font-weight: 500; border-radius: 8px;
    cursor: pointer; transition: all 0.15s;
  }
  .model-btn:hover { border-color: var(--accent); color: var(--accent); }
  .model-btn.active {
    background: var(--accent); border-color: var(--accent); color: white; font-weight: 600;
  }
  .model-btn.active[data-model="DiT"] {
    background: var(--dit); border-color: var(--dit);
  }

  .legend-bar {
    display: flex; gap: 20px; margin-bottom: 18px; font-size: 0.85rem; color: var(--muted);
  }
  .legend-bar .swatch {
    display: inline-block; width: 14px; height: 14px; border-radius: 3px;
    vertical-align: middle; margin-right: 6px;
  }

  footer {
    text-align: center; padding: 28px; font-size: 0.78rem; color: var(--muted);
    border-top: 1px solid var(--border); margin-top: 28px;
  }

  .error-table { max-height: 500px; overflow-y: auto; }
  .error-table table.data { font-size: 0.82rem; }
  .error-table td { padding: 6px 10px; }
  .error-table th { padding: 8px 10px; position: sticky; top: 0; z-index: 2; }

  @media (max-width: 768px) {
    header { padding: 18px 20px 16px; } header h1 { font-size: 1.35rem; }
    .tabs { padding: 12px 12px 0; overflow-x: auto; flex-wrap: nowrap; }
    .tab-btn { white-space: nowrap; padding: 9px 14px; font-size: 0.8rem; flex-shrink: 0; }
    .pages { padding: 16px 14px 32px; }
    .card { padding: 18px 16px; border-radius: 10px; }
  }
</style>
</head>
<body>

<header>
  <div class="eyebrow">Evaluation Report &bull; Mixed Results</div>
  <h1>Gemini vs DiT — Document Classification Comparison</h1>
  <div class="subtitle">
    Head-to-head comparison of <b>Gemini</b> (text-based classification) and
    <b>DiT</b> (Document Image Transformer, vision-based classification) on a
    <b>PLACEHOLDER_N</b>-document evaluation set with ground-truth labels.
    Documents are classified as <b>safe</b> (suitable for automated extraction) or
    <b>risky</b> (handwritten/mixed content with hallucination risk).
  </div>
  <div class="meta">Generated PLACEHOLDER_DATE &bull; Source: eval/mixed_results - Sheet1.csv</div>
</header>

<div class="tabs">
  <button class="tab-btn active" onclick="showPage('overview', this)">Overview</button>
  <button class="tab-btn" onclick="showPage('detail', this)">Detailed Metrics</button>
  <button class="tab-btn" onclick="showPage('errors', this)">Error Analysis</button>
  <button class="tab-btn" onclick="showPage('agreement', this)">Model Agreement</button>
  <button class="tab-btn" onclick="showPage('findings', this)">Findings</button>
</div>

<div class="pages">

  <!-- ========== OVERVIEW ========== -->
  <div class="page active" id="page-overview">

    <div class="card">
      <h2>Executive Summary</h2>
      <p>
        Two classifiers were evaluated on the same set of <b>PLACEHOLDER_N</b> documents:
        <b>Gemini</b> (Google's multimodal LLM, classifying via text analysis with
        <code>typed</code>/<code>handwritten</code>/<code>mixed</code> labels) and
        <b>DiT</b> (our fine-tuned Document Image Transformer, classifying via page images with
        <code>safe_for_extraction</code>/<code>review</code>/<code>high_hallucination_risk</code> labels).
      </p>
      <p>
        For this evaluation, labels are binarised:
      </p>
      <ul>
        <li><b>Ground truth:</b> <code>safe_for_extraction</code> → safe, <code>high_hallucination_risk</code> → risky</li>
        <li><b>Gemini:</b> <code>typed</code> → safe, <code>handwritten</code> and <code>mixed</code> → risky</li>
        <li><b>DiT:</b> <code>safe_for_extraction</code> → safe, <code>review</code> and <code>high_hallucination_risk</code> → risky</li>
      </ul>

      <div class="kpi-grid" style="margin-top:18px">
        <div class="kpi">
          <span class="label">Evaluation Set</span>
          <span class="value" id="kpi-n" style="color:var(--primary)"></span>
          <span class="delta flat" id="kpi-balance"></span>
        </div>
        <div class="kpi">
          <span class="label">Gemini Accuracy</span>
          <span class="value gemini-color" id="kpi-gem-acc"></span>
        </div>
        <div class="kpi">
          <span class="label">DiT Accuracy</span>
          <span class="value dit-color" id="kpi-dit-acc"></span>
        </div>
        <div class="kpi">
          <span class="label">Gemini FSR</span>
          <span class="value" id="kpi-gem-fsr"></span>
          <span class="delta" id="kpi-gem-fsr-note"></span>
        </div>
        <div class="kpi">
          <span class="label">DiT FSR</span>
          <span class="value" id="kpi-dit-fsr"></span>
          <span class="delta" id="kpi-dit-fsr-note"></span>
        </div>
        <div class="kpi">
          <span class="label">Gemini Macro F1</span>
          <span class="value gemini-color" id="kpi-gem-f1"></span>
        </div>
        <div class="kpi">
          <span class="label">DiT Macro F1</span>
          <span class="value dit-color" id="kpi-dit-f1"></span>
        </div>
        <div class="kpi">
          <span class="label">Both Models Agree</span>
          <span class="value" id="kpi-agree" style="color:var(--primary)"></span>
        </div>
      </div>
    </div>

    <div class="card">
      <h2>Label Distribution</h2>
      <div id="plot-label-dist" class="plotly-chart" style="min-height:340px"></div>
    </div>

    <div class="two-col">
      <div class="plot-card">
        <h3>Gemini — Confusion Matrix</h3>
        <div id="plot-cm-gemini" class="plotly-chart"></div>
      </div>
      <div class="plot-card">
        <h3>DiT — Confusion Matrix</h3>
        <div id="plot-cm-dit" class="plotly-chart"></div>
      </div>
    </div>
  </div>

  <!-- ========== DETAILED METRICS ========== -->
  <div class="page" id="page-detail">
    <div class="card">
      <h2>Classification Metrics — Head to Head</h2>
      <p>All metrics computed on the binarised labels (safe=0, risky=1). FSR = False Safe Rate = FN/(TN+FN), the fraction of documents predicted safe that are actually risky.</p>
      <table class="data">
        <thead>
          <tr>
            <th>Metric</th>
            <th class="num">Gemini</th>
            <th class="num">DiT</th>
            <th class="center">Better</th>
          </tr>
        </thead>
        <tbody id="detail-metrics-tbody"></tbody>
      </table>
    </div>

    <div class="card">
      <h2>Confusion Matrix Comparison</h2>
      <table class="data">
        <thead>
          <tr>
            <th>Model</th>
            <th class="num">TN</th>
            <th class="num">FP</th>
            <th class="num">FN</th>
            <th class="num">TP</th>
            <th class="num">FSR</th>
            <th class="num">False Flag</th>
          </tr>
        </thead>
        <tbody id="detail-cm-tbody"></tbody>
      </table>
    </div>

    <div class="card">
      <h2>Per Original-Class Accuracy</h2>
      <p>Breakdown by the <em>original</em> multi-class labels before binarisation, showing how each model handles each sub-category.</p>

      <h3 style="color:var(--gemini)">Gemini</h3>
      <table class="data">
        <thead>
          <tr>
            <th>Original Class</th>
            <th class="num">Count</th>
            <th class="num">True Safe</th>
            <th class="num">True Risky</th>
            <th class="num">Correct</th>
            <th class="num">Accuracy</th>
          </tr>
        </thead>
        <tbody id="per-class-gemini-tbody"></tbody>
      </table>

      <h3 style="color:var(--dit); margin-top:24px">DiT</h3>
      <table class="data">
        <thead>
          <tr>
            <th>Original Class</th>
            <th class="num">Count</th>
            <th class="num">True Safe</th>
            <th class="num">True Risky</th>
            <th class="num">Correct</th>
            <th class="num">Accuracy</th>
          </tr>
        </thead>
        <tbody id="per-class-dit-tbody"></tbody>
      </table>
    </div>

    <div class="plot-card">
      <h3>Gemini Confidence Distribution</h3>
      <div id="plot-confidence" class="plotly-chart" style="min-height:380px"></div>
      <p class="caption">Gemini self-reported confidence, split by whether the prediction was correct or wrong.</p>
    </div>
  </div>

  <!-- ========== ERROR ANALYSIS ========== -->
  <div class="page" id="page-errors">
    <div class="card">
      <h2>Error Analysis</h2>
      <p>
        All misclassified documents for each model. <b>FN (risky→safe)</b> are the dangerous
        errors — risky documents incorrectly marked safe. <b>FP (safe→risky)</b> are
        false alarms — safe documents flagged as risky.
      </p>
    </div>

    <div class="model-selector" data-group="errors">
      <button class="model-btn active" data-model="Gemini" onclick="selectModel('errors', 'Gemini', this)">Gemini</button>
      <button class="model-btn" data-model="DiT" onclick="selectModel('errors', 'DiT', this)">DiT</button>
    </div>

    <div class="card" id="error-summary-card">
      <h3 id="error-summary-title">Error Summary</h3>
      <div class="kpi-grid" id="error-kpis" style="margin-top:6px"></div>
    </div>

    <div class="card">
      <h3 id="error-table-title">Misclassified Documents</h3>
      <div class="error-table" id="error-table-container"></div>
    </div>
  </div>

  <!-- ========== AGREEMENT ========== -->
  <div class="page" id="page-agreement">
    <div class="card">
      <h2>Model Agreement Analysis</h2>
      <p>
        How often do Gemini and DiT agree on the classification — and when they disagree,
        which one is right?
      </p>

      <div class="kpi-grid" style="margin-top:14px">
        <div class="kpi">
          <span class="label">Both Correct</span>
          <span class="value" id="agree-both-correct" style="color:var(--good)"></span>
        </div>
        <div class="kpi">
          <span class="label">Only Gemini Correct</span>
          <span class="value gemini-color" id="agree-gemini-only"></span>
        </div>
        <div class="kpi">
          <span class="label">Only DiT Correct</span>
          <span class="value dit-color" id="agree-dit-only"></span>
        </div>
        <div class="kpi">
          <span class="label">Both Wrong</span>
          <span class="value" id="agree-both-wrong" style="color:var(--bad)"></span>
        </div>
      </div>
    </div>

    <div class="plot-card">
      <div id="plot-agreement" class="plotly-chart" style="min-height:400px"></div>
      <p class="caption">Agreement breakdown: when models agree vs when only one is correct.</p>
    </div>

    <div class="plot-card">
      <div id="plot-agree-bar" class="plotly-chart" style="min-height:400px"></div>
      <p class="caption">Side-by-side accuracy comparison across the four agreement quadrants.</p>
    </div>
  </div>

  <!-- ========== FINDINGS ========== -->
  <div class="page" id="page-findings">
    <div class="card">
      <h2>Headline Findings</h2>
      <div id="findings-callouts"></div>
    </div>

    <div class="card">
      <h2>Methodology Notes</h2>
      <ul>
        <li><b>Gemini</b> classifies from the extracted text of each page, labelling
            <code>typed</code> / <code>handwritten</code> / <code>mixed</code>.
            Self-reported confidence is available (mean ~90%).</li>
        <li><b>DiT</b> classifies from the rendered page image using our fine-tuned Document Image Transformer,
            labelling <code>safe_for_extraction</code> / <code>review</code> / <code>high_hallucination_risk</code>.</li>
        <li>4 documents were dropped from the original 652-row CSV due to missing Gemini classification.</li>
        <li>Binarisation: both <code>mixed</code> and <code>handwritten</code> are treated as risky for Gemini;
            both <code>review</code> and <code>high_hallucination_risk</code> are treated as risky for DiT.</li>
      </ul>
    </div>
  </div>

</div>

<footer>
  Mixed Results Evaluation Report &bull; Generated PLACEHOLDER_DATE &bull; Gemini vs DiT Comparison
</footer>

<script>
const R = PLACEHOLDER_DATA;
const CFG = { responsive: true, displayModeBar: false, displaylogo: false };

function fmtPct(x, dp=1) {
  if (x === null || x === undefined || isNaN(x)) return '—';
  return (x * 100).toFixed(dp) + '%';
}
function fmtNum(x, dp=3) {
  return (x === null || x === undefined || isNaN(x)) ? '—' : Number(x).toFixed(dp);
}

// ---- overview KPIs ----
function renderOverviewKpis() {
  const g = R.metrics.Gemini, d = R.metrics.DiT;
  document.getElementById('kpi-n').textContent = R.n_total;
  document.getElementById('kpi-balance').textContent =
    R.n_safe + ' safe / ' + R.n_risky + ' risky';

  document.getElementById('kpi-gem-acc').textContent = fmtPct(g.accuracy);
  document.getElementById('kpi-dit-acc').textContent = fmtPct(d.accuracy);

  const gemFsr = document.getElementById('kpi-gem-fsr');
  gemFsr.textContent = fmtPct(g.fsr);
  gemFsr.style.color = g.fsr < 0.05 ? 'var(--good)' : (g.fsr < 0.15 ? 'var(--warn)' : 'var(--bad)');
  document.getElementById('kpi-gem-fsr-note').textContent = g.fn + ' missed risky docs';
  document.getElementById('kpi-gem-fsr-note').className = 'delta ' + (g.fn === 0 ? 'good' : 'bad');

  const ditFsr = document.getElementById('kpi-dit-fsr');
  ditFsr.textContent = fmtPct(d.fsr);
  ditFsr.style.color = d.fsr < 0.05 ? 'var(--good)' : (d.fsr < 0.15 ? 'var(--warn)' : 'var(--bad)');
  document.getElementById('kpi-dit-fsr-note').textContent = d.fn + ' missed risky docs';
  document.getElementById('kpi-dit-fsr-note').className = 'delta ' + (d.fn === 0 ? 'good' : 'bad');

  document.getElementById('kpi-gem-f1').textContent = fmtNum(g.macro_f1, 3);
  document.getElementById('kpi-dit-f1').textContent = fmtNum(d.macro_f1, 3);

  const agreeRate = (R.agreement.both_correct + R.agreement.both_wrong) / R.agreement.total;
  document.getElementById('kpi-agree').textContent = fmtPct(agreeRate);
}

// ---- label distribution chart ----
function drawLabelDist() {
  const cats = ['safe_for_extraction', 'high_hallucination_risk'];
  const gemCats = ['typed', 'mixed', 'handwritten'];
  const ditCats = ['safe_for_extraction', 'review', 'high_hallucination_risk'];

  const gt = cats.map(c => R.class_distribution.expected[c] || 0);
  const gem = gemCats.map(c => R.class_distribution.gemini[c] || 0);
  const dit = ditCats.map(c => R.class_distribution.dit[c] || 0);

  const data = [
    { x: cats, y: gt, name: 'Ground Truth', type: 'bar',
      marker: { color: ['#059669', '#dc2626'] } },
    { x: gemCats, y: gem, name: 'Gemini', type: 'bar',
      marker: { color: ['#93c5fd', '#fbbf24', '#f87171'] } },
    { x: ditCats, y: dit, name: 'DiT', type: 'bar',
      marker: { color: ['#6ee7b7', '#fde68a', '#fca5a5'] } },
  ];
  Plotly.newPlot('plot-label-dist', data, {
    title: { text: 'Label Distribution — Ground Truth vs Predictions (original classes)', font: { size: 16 }, x: 0.5 },
    xaxis: { title: '', showgrid: false },
    yaxis: { title: 'Count', showgrid: true, gridcolor: '#e5e7eb' },
    barmode: 'group', height: 380,
    plot_bgcolor: 'white', paper_bgcolor: 'white',
    legend: { orientation: 'h', y: -0.18 },
    margin: { l: 60, r: 30, t: 60, b: 80 },
  }, CFG);
}

// ---- confusion matrices ----
function drawConfusion() {
  Plotly.newPlot('plot-cm-gemini', R.confusion_figs.Gemini.data, R.confusion_figs.Gemini.layout, CFG);
  Plotly.newPlot('plot-cm-dit', R.confusion_figs.DiT.data, R.confusion_figs.DiT.layout, CFG);
}

// ---- detail metrics table ----
function renderDetailMetrics() {
  const g = R.metrics.Gemini, d = R.metrics.DiT;
  const rows = [
    ['Accuracy', g.accuracy, d.accuracy, 'higher'],
    ['Macro F1', g.macro_f1, d.macro_f1, 'higher'],
    ['Precision (risky)', g.precision_risky, d.precision_risky, 'higher'],
    ['Recall (risky)', g.recall_risky, d.recall_risky, 'higher'],
    ['F1 (risky class)', g.f1_risky, d.f1_risky, 'higher'],
    ['Precision (safe)', g.precision_safe, d.precision_safe, 'higher'],
    ['Recall (safe)', g.recall_safe, d.recall_safe, 'higher'],
    ['F1 (safe class)', g.f1_safe, d.f1_safe, 'higher'],
    ['False Safe Rate (FSR)', g.fsr, d.fsr, 'lower'],
    ['False Flag Rate', g.false_flag_rate, d.false_flag_rate, 'lower'],
  ];

  let html = '';
  for (const [name, gv, dv, dir] of rows) {
    let winner = '';
    const isFsr = dir === 'lower';
    if (Math.abs(gv - dv) < 0.001) {
      winner = 'Tie';
    } else if (isFsr ? gv < dv : gv > dv) {
      winner = '<span style="color:var(--gemini);font-weight:600">Gemini</span>';
    } else {
      winner = '<span style="color:var(--dit);font-weight:600">DiT</span>';
    }
    const gCls = (isFsr ? gv < dv : gv > dv) ? 'best' : (Math.abs(gv-dv)<0.001 ? '' : 'worst');
    const dCls = (isFsr ? dv < gv : dv > gv) ? 'best' : (Math.abs(gv-dv)<0.001 ? '' : 'worst');
    const fmt = name.includes('Rate') || name.includes('FSR') ? fmtPct : (v => fmtNum(v, 3));
    html += `<tr>
      <td><b>${name}</b></td>
      <td class="num ${gCls}">${fmt(gv)}</td>
      <td class="num ${dCls}">${fmt(dv)}</td>
      <td class="center">${winner}</td>
    </tr>`;
  }
  document.getElementById('detail-metrics-tbody').innerHTML = html;

  let ch = '';
  for (const [model, m] of Object.entries(R.metrics)) {
    const fsrCls = m.fsr < 0.05 ? 'good-cell' : (m.fsr < 0.15 ? '' : 'bad-cell');
    ch += `<tr>
      <td><b>${model}</b></td>
      <td class="num">${m.tn}</td>
      <td class="num">${m.fp}</td>
      <td class="num bad-cell">${m.fn}</td>
      <td class="num">${m.tp}</td>
      <td class="num ${fsrCls}">${fmtPct(m.fsr, 2)}</td>
      <td class="num">${fmtPct(m.false_flag_rate, 2)}</td>
    </tr>`;
  }
  document.getElementById('detail-cm-tbody').innerHTML = ch;
}

// ---- per-class tables ----
function renderPerClass() {
  for (const [model, key] of [['gemini', 'Gemini'], ['dit', 'DiT']]) {
    let html = '';
    for (const row of R.per_class[key]) {
      const accCls = row.accuracy >= 0.9 ? 'good-cell' : (row.accuracy < 0.7 ? 'bad-cell' : '');
      html += `<tr>
        <td><code>${row.original_class}</code></td>
        <td class="num">${row.n}</td>
        <td class="num">${row.n_safe_true}</td>
        <td class="num">${row.n_risky_true}</td>
        <td class="num">${row.correct}</td>
        <td class="num ${accCls}">${fmtPct(row.accuracy)}</td>
      </tr>`;
    }
    document.getElementById('per-class-' + model + '-tbody').innerHTML = html;
  }
}

// ---- confidence histogram ----
function drawConfidence() {
  Plotly.newPlot('plot-confidence', R.gemini_confidence_hist.data,
                 R.gemini_confidence_hist.layout, CFG);
}

// ---- error analysis ----
function renderErrors(modelKey) {
  const errs = R.errors[modelKey];
  const m = R.metrics[modelKey];

  document.getElementById('error-summary-title').textContent = modelKey + ' — Error Summary';

  const fnList = errs.filter(e => e.error_type.startsWith('FN'));
  const fpList = errs.filter(e => e.error_type.startsWith('FP'));

  document.getElementById('error-kpis').innerHTML = `
    <div class="kpi">
      <span class="label">Total Errors</span>
      <span class="value" style="color:var(--bad)">${m.fn + m.fp}</span>
    </div>
    <div class="kpi">
      <span class="label">False Negatives (risky→safe)</span>
      <span class="value" style="color:var(--bad)">${m.fn}</span>
      <span class="delta bad">Dangerous: missed risky docs</span>
    </div>
    <div class="kpi">
      <span class="label">False Positives (safe→risky)</span>
      <span class="value" style="color:var(--warn)">${m.fp}</span>
      <span class="delta flat">Wasteful: unnecessary review</span>
    </div>
    <div class="kpi">
      <span class="label">Error Rate</span>
      <span class="value" style="color:var(--bad)">${fmtPct(1 - m.accuracy)}</span>
    </div>
  `;

  document.getElementById('error-table-title').textContent = modelKey + ' — Misclassified Documents';

  let html = '<table class="data"><thead><tr>';
  html += '<th>File</th><th>Error Type</th><th>Gemini Class</th><th>DiT Class</th><th>Expected</th>';
  html += '</tr></thead><tbody>';
  for (const e of errs) {
    const rowCls = e.error_type.startsWith('FN') ? 'style="background:#fef2f2"' : '';
    html += `<tr ${rowCls}>
      <td>${e.file}</td>
      <td><b class="${e.error_type.startsWith('FN') ? 'bad-cell' : ''}">${e.error_type}</b></td>
      <td>${e.classification_gemini}</td>
      <td>${e.classification_idan}</td>
      <td>${e['expected output']}</td>
    </tr>`;
  }
  html += '</tbody></table>';
  document.getElementById('error-table-container').innerHTML = html;
}

// ---- agreement ----
function renderAgreement() {
  const a = R.agreement;
  document.getElementById('agree-both-correct').textContent = a.both_correct + ' (' + fmtPct(a.both_correct / a.total) + ')';
  document.getElementById('agree-gemini-only').textContent = a.gemini_only + ' (' + fmtPct(a.gemini_only / a.total) + ')';
  document.getElementById('agree-dit-only').textContent = a.dit_only + ' (' + fmtPct(a.dit_only / a.total) + ')';
  document.getElementById('agree-both-wrong').textContent = a.both_wrong + ' (' + fmtPct(a.both_wrong / a.total) + ')';

  const pieData = [{
    type: 'pie',
    labels: ['Both Correct', 'Only Gemini Correct', 'Only DiT Correct', 'Both Wrong'],
    values: [a.both_correct, a.gemini_only, a.dit_only, a.both_wrong],
    marker: { colors: ['#059669', '#2563eb', '#dc2626', '#6b7280'] },
    textinfo: 'label+value+percent',
    textposition: 'inside',
    hole: 0.35,
  }];
  Plotly.newPlot('plot-agreement', pieData, {
    title: { text: 'Agreement / Disagreement Breakdown', font: { size: 16 }, x: 0.5 },
    height: 420, margin: { l: 30, r: 30, t: 60, b: 30 },
    paper_bgcolor: 'white',
  }, CFG);

  const barData = [
    { x: ['Gemini', 'DiT'], y: [R.metrics.Gemini.accuracy, R.metrics.DiT.accuracy],
      type: 'bar', name: 'Accuracy',
      marker: { color: ['#2563eb', '#dc2626'] },
      text: [fmtPct(R.metrics.Gemini.accuracy), fmtPct(R.metrics.DiT.accuracy)],
      textposition: 'outside' },
  ];
  Plotly.newPlot('plot-agree-bar', barData, {
    title: { text: 'Overall Accuracy Comparison', font: { size: 16 }, x: 0.5 },
    yaxis: { title: 'Accuracy', range: [0, 1.05], showgrid: true, gridcolor: '#e5e7eb',
             tickformat: ',.0%' },
    height: 400, plot_bgcolor: 'white', paper_bgcolor: 'white',
    showlegend: false, margin: { l: 60, r: 30, t: 60, b: 60 },
  }, CFG);
}

// ---- findings ----
function renderFindings() {
  const g = R.metrics.Gemini, d = R.metrics.DiT;
  const a = R.agreement;

  let html = '';

  const accWinner = g.accuracy > d.accuracy ? 'Gemini' : 'DiT';
  const accBetter = g.accuracy > d.accuracy ? g : d;
  const accWorse = g.accuracy > d.accuracy ? d : g;
  html += `<div class="callout ${accWinner === 'Gemini' ? 'info' : 'good'}">
    <b>${accWinner} leads on accuracy:</b> ${fmtPct(accBetter.accuracy)} vs ${fmtPct(accWorse.accuracy)}.
  </div>`;

  const fsrWinner = g.fsr < d.fsr ? 'Gemini' : 'DiT';
  const fsrBetter = g.fsr < d.fsr ? g : d;
  const fsrWorse = g.fsr < d.fsr ? d : g;
  const fsrClass = fsrBetter.fsr < 0.05 ? 'good' : 'warn';
  html += `<div class="callout ${fsrClass}">
    <b>${fsrWinner} has the lower False Safe Rate:</b> ${fmtPct(fsrBetter.fsr)} vs ${fmtPct(fsrWorse.fsr)}.
    ${fsrBetter.fsr < 0.05 ? 'This is within the <5% safety target.' : 'Neither model meets the <5% safety target.'}
    ${fsrWinner} misses only ${fsrBetter.fn} risky document(s) vs ${fsrWorse.fn}.
  </div>`;

  if (a.both_wrong > 0) {
    html += `<div class="callout warn">
      <b>${a.both_wrong} document(s) are misclassified by both models.</b>
      These are the hardest cases and likely warrant manual review or additional training data.
    </div>`;
  }

  if (g.avg_confidence_wrong !== null && g.avg_confidence_wrong !== undefined) {
    html += `<div class="callout info">
      <b>Gemini confidence is weakly calibrated:</b> average confidence on correct predictions is
      ${fmtNum(g.avg_confidence_correct, 1)}% vs ${fmtNum(g.avg_confidence_wrong, 1)}% on errors.
      ${Math.abs(g.avg_confidence_correct - g.avg_confidence_wrong) < 5
        ? 'The gap is small — confidence alone cannot reliably flag errors.'
        : 'The gap suggests confidence may help triage uncertain predictions.'}
    </div>`;
  }

  document.getElementById('findings-callouts').innerHTML = html;
}

// ---- tabs ----
const TAB_INIT = { overview: false, detail: false, errors: false, agreement: false, findings: false };

function showPage(key, btn) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.getElementById('page-' + key).classList.add('active');
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');

  if (!TAB_INIT[key]) {
    if (key === 'overview')   { drawLabelDist(); drawConfusion(); }
    if (key === 'detail')     { renderDetailMetrics(); renderPerClass(); drawConfidence(); }
    if (key === 'errors')     renderErrors('Gemini');
    if (key === 'agreement')  renderAgreement();
    if (key === 'findings')   renderFindings();
    TAB_INIT[key] = true;
  }
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function selectModel(group, modelKey, btn) {
  document.querySelectorAll(`.model-selector[data-group="${group}"] .model-btn`)
    .forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  if (group === 'errors') renderErrors(modelKey);
}

// ---- init ----
renderOverviewKpis();
drawLabelDist();
drawConfusion();
</script>
</body>
</html>"""


def main():
    df = load_and_binarise(CSV_PATH)
    report = build_report_data(df)

    report_json = json.dumps(report, ensure_ascii=False, default=str)
    gen_date = report["generated_at"]

    html = HTML_TEMPLATE
    html = html.replace("PLACEHOLDER_DATA", report_json)
    html = html.replace("PLACEHOLDER_N", str(report["n_total"]))
    html = html.replace("PLACEHOLDER_DATE", gen_date)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"Report written to {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
