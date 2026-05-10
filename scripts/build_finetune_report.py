#!/usr/bin/env python3
"""
Build a single-file, self-contained HTML report of the Phase 6 fine-tune results
with Phase-6c cost-weighted thresholds.

The report shows, per model:

    * Training curves (loss, F1, ROC-AUC, ECE) on the validation set.
    * Held-out test confusion matrices side-by-side: binary-0.5 (old) vs the
      cost-weighted τ* (new production policy).
    * External-set confusion matrices, FSR comparison (Phase 5 / @0.5 / @τ*),
      and per-source breakdown at τ*.
    * Worst-case predictions reused from validation_report/report_data.json.

Inputs:
    - logs/baseline/baseline_resnet50.json
    - logs/baseline/baseline_vit_base_patch16_224.json
    - logs/dit/dit.json
    - checkpoints/baseline/best_resnet50.pt
    - checkpoints/baseline/best_vit_base_patch16_224.pt
    - checkpoints/dit/best_model.pt              (thresholds now contain τ*)
    - validation_report/report_data.json        (needs `per_doc` arrays —
      re-run scripts/_run_validation_inference.py after any threshold change)
    - data/metadata.csv, data/splits/*.csv
    - reports/_finetune_report_template.html

Output:
    - reports/finetune_report.html              (single file, share via email)

Run from repo root:
    python scripts/build_finetune_report.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import torch
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
MODELS = [
    {
        "name":       "ResNet50",
        "log":        ROOT / "logs" / "baseline" / "baseline_resnet50.json",
        "checkpoint": ROOT / "checkpoints" / "baseline" / "best_resnet50.pt",
    },
    {
        "name":       "ViT-Base",
        "log":        ROOT / "logs" / "baseline" / "baseline_vit_base_patch16_224.json",
        "checkpoint": ROOT / "checkpoints" / "baseline" / "best_vit_base_patch16_224.pt",
    },
    {
        "name":       "DiT",
        "log":        ROOT / "logs" / "dit" / "dit.json",
        "checkpoint": ROOT / "checkpoints" / "dit" / "best_model.pt",
    },
]

# Phase 5 (pre-Phase-6) external FSR baseline.
PHASE5_EXT_FSR = {"ResNet50": 0.362, "ViT-Base": 0.500, "DiT": 0.309}

# Phase 5 thresholds, for the "old T_low" column of the test table.
PHASE5_T_LOW = {"ResNet50": 0.827, "ViT-Base": 0.521, "DiT": 0.805}

SOURCE_FAMILY = {
    "regular_forms":                "regular_form",
    "regular_forms_edge_cases":     "regular_form_edge",
    "handwritten":                  "questionnaire_named / _uuid",
    "handwritten_edge_cases":       "handwritten_edge",
    "handwritten_and_questioniers": "mixed_hwq",
    # Legacy key — kept so reports rebuilt from a pre-Phase-6d metadata
    # snapshot (e.g. data/metadata_v2_precorrection.csv) still render.
    "spaciel_font":                 "special_font",
}

COLOR = {"ResNet50": "#2563eb", "ViT-Base": "#059669", "DiT": "#dc2626"}

TEMPLATE_PATH = ROOT / "reports" / "_finetune_report_template.html"
OUTPUT_PATH   = ROOT / "reports" / "finetune_report.html"
VAL_REPORT    = ROOT / "validation_report" / "report_data.json"


# -----------------------------------------------------------------------------
# 1. Corpus + splits
# -----------------------------------------------------------------------------
def build_corpus() -> tuple[list[dict], int, list[dict]]:
    meta = pd.read_csv(ROOT / "data" / "metadata.csv")
    counts = meta["source_folder"].value_counts()
    rows = []
    for src, n in counts.items():
        label = int(meta[meta["source_folder"] == src]["label_binary"].iloc[0])
        rows.append({
            "source_folder":   src,
            "pages":           int(n),
            "label":           label,
            "template_family": SOURCE_FAMILY.get(src, "—"),
        })
    splits_rows = []
    for split_name in ["train", "val", "test"]:
        sdf = pd.read_csv(ROOT / "data" / "splits" / f"{split_name}.csv")
        vc = sdf["label_binary"].value_counts()
        splits_rows.append({
            "split": split_name,
            "total": int(len(sdf)),
            "safe":  int(vc.get(0, 0)),
            "risky": int(vc.get(1, 0)),
        })
    return rows, int(len(meta)), splits_rows


# -----------------------------------------------------------------------------
# 2. Training curves
# -----------------------------------------------------------------------------
def build_training_data() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for m in MODELS:
        log = json.load(open(m["log"]))
        ck  = torch.load(str(m["checkpoint"]), map_location="cpu", weights_only=False)

        if m["name"] == "DiT":
            stage_max_epoch: dict[str, int] = {}
            for rec in log:
                phase = rec.get("phase", "")
                if "/" not in phase:
                    continue
                stage, _ = phase.split("/")
                stage_max_epoch[stage] = max(stage_max_epoch.get(stage, 0), rec["epoch"])
            stage_order = ["stage1", "stage2", "stage3"]
            stage_offset: dict[str, int] = {}
            stage_boundaries: list[float] = []
            cum = 0
            for s in stage_order:
                stage_offset[s] = cum
                if s in stage_max_epoch:
                    cum += stage_max_epoch[s]
                    if s != stage_order[-1]:
                        stage_boundaries.append(cum + 0.5)

            train_records: dict[int, float] = {}
            val_records:   dict[int, dict]  = {}
            for rec in log:
                phase = rec.get("phase", "")
                if "/" not in phase:
                    continue
                stage, sub = phase.split("/")
                e = rec["epoch"] + stage_offset[stage]
                if sub == "train":
                    train_records[e] = rec.get("loss")
                elif sub == "val":
                    val_records[e] = rec

            epochs = sorted(val_records.keys())
            best_epoch_abs = None
            if "stage" in ck and "epoch" in ck:
                best_stage = ck.get("best_stage") or ck.get("stage")
                if best_stage in stage_offset:
                    best_epoch_abs = ck["epoch"] + stage_offset[best_stage]

            out[m["name"]] = {
                "epochs":           epochs,
                "train_loss":       [train_records.get(e) for e in epochs],
                "val_loss":         [val_records[e].get("loss")    for e in epochs],
                "val_f1":           [val_records[e].get("f1")      for e in epochs],
                "val_roc_auc":      [val_records[e].get("roc_auc") for e in epochs],
                "val_ece":          [val_records[e].get("ece")     for e in epochs],
                "best_epoch":       best_epoch_abs,
                "best_val_f1":      ck.get("val_f1"),
                "stage_boundaries": stage_boundaries,
            }
        else:
            train_records = {}
            val_records   = {}
            for rec in log:
                phase = rec.get("phase", "")
                e = rec["epoch"]
                if phase == "train":
                    train_records[e] = rec.get("loss")
                elif phase == "val":
                    val_records[e] = rec
            epochs = sorted(val_records.keys())
            out[m["name"]] = {
                "epochs":           epochs,
                "train_loss":       [train_records.get(e) for e in epochs],
                "val_loss":         [val_records[e].get("loss")    for e in epochs],
                "val_f1":           [val_records[e].get("f1")      for e in epochs],
                "val_roc_auc":      [val_records[e].get("roc_auc") for e in epochs],
                "val_ece":          [val_records[e].get("ece")     for e in epochs],
                "best_epoch":       ck.get("epoch"),
                "best_val_f1":      ck.get("val_f1"),
                "stage_boundaries": [],
            }
    return out


# -----------------------------------------------------------------------------
# 3. Held-out test metrics + figures (binary 0.5 vs τ*)
# -----------------------------------------------------------------------------
def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


def _cm(y_prob: np.ndarray, y_true: np.ndarray, tau: float) -> dict:
    pred = (y_prob >= tau).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp, "threshold": float(tau),
        "accuracy": float((tp + tn) / len(y_true)) if len(y_true) else float("nan"),
        "fsr": float(fn / (fn + tp)) if (fn + tp) else float("nan"),
    }


def build_test_metrics() -> tuple[dict, dict]:
    metrics: dict[str, dict] = {}
    figs:    dict[str, dict] = {}

    for m in MODELS:
        ck   = torch.load(str(m["checkpoint"]), map_location="cpu", weights_only=False)
        tm   = ck.get("test_metrics", {}) or {}
        th   = ck.get("thresholds", {})   or {}
        temp = float(ck.get("temperature", 1.0)) or 1.0

        logits = np.asarray(ck["test_logits"]).ravel()
        labels = np.asarray(ck["test_labels"]).ravel().astype(int)
        probs  = _sigmoid(logits / temp)

        tau = float(th.get("T_low", 0.5))
        cm_old = _cm(probs, labels, 0.5)
        cm_new = _cm(probs, labels, tau)

        metrics[m["name"]] = {
            "f1":           float(tm.get("f1", float("nan"))),
            "roc_auc":      float(tm.get("roc_auc", float("nan"))),
            "pr_auc":       float(tm.get("pr_auc",  float("nan"))),
            "ece":          float(tm.get("ece",     float("nan"))),
            "temperature":  temp,
            "t_low_old":    float(PHASE5_T_LOW.get(m["name"], float("nan"))),
            "tau":          tau,
            "t_high":       float(th.get("T_high", float("nan"))),
            "cm_old":       cm_old,
            "cm_new":       cm_new,
        }

        figs[m["name"]] = {
            "cm_old": _fig_to_dict(_confusion_fig(
                cm_old, f"{m['name']} — Test @ 0.5 (old policy)")),
            "cm_new": _fig_to_dict(_confusion_fig(
                cm_new, f"{m['name']} — Test @ τ*={tau:.3f} (cost-weighted)",
                accent="#059669")),
            "histogram": _fig_to_dict(_hist_fig(
                probs[labels == 0], probs[labels == 1],
                tau_new=tau, t_high=float(th.get("T_high", 1.0)),
                title=f"{m['name']} — Test Set Probability Distribution")),
        }
    return metrics, figs


# -----------------------------------------------------------------------------
# 4. External validation — uses per_doc arrays written by
#    scripts/_run_validation_inference.py
# -----------------------------------------------------------------------------
def build_external() -> tuple[dict, dict, dict, dict]:
    if not VAL_REPORT.exists():
        raise FileNotFoundError(
            f"{VAL_REPORT} not found. "
            "Run `python scripts/_run_validation_inference.py` first."
        )
    vr = json.load(open(VAL_REPORT))

    # Older report_data.json versions lack per_doc — guard.
    missing_per_doc = [name for name, block in vr["models"].items()
                       if "per_doc" not in block]
    if missing_per_doc:
        raise RuntimeError(
            f"report_data.json is missing `per_doc` arrays for: {missing_per_doc}. "
            "Re-run `python scripts/_run_validation_inference.py` to regenerate."
        )

    ext_metrics: dict[str, dict] = {}
    ext_figs:    dict[str, dict] = {}
    per_source:  dict[str, list[dict]] = {}

    for name, block in vr["models"].items():
        y_true = np.asarray(block["per_doc"]["y_true"], dtype=int)
        y_prob = np.asarray(block["per_doc"]["y_prob"], dtype=float)
        source = np.asarray(block["per_doc"]["source"])
        tau    = float(block["summary"]["t_low"])

        cm_old = _cm(y_prob, y_true, 0.5)
        cm_new = _cm(y_prob, y_true, tau)

        ext_metrics[name] = {
            "fsr_phase5": float(PHASE5_EXT_FSR.get(name, float("nan"))),
            "fsr_old":    cm_old["fsr"],
            "fsr_new":    cm_new["fsr"],
            "cm_old":     cm_old,
            "cm_new":     cm_new,
            "tau":        tau,
        }

        ext_figs[name] = {
            "cm_old":   _fig_to_dict(_confusion_fig(
                cm_old, f"{name} — External @ 0.5")),
            "cm_new":   _fig_to_dict(_confusion_fig(
                cm_new, f"{name} — External @ τ*={tau:.3f}",
                accent="#059669")),
            "histogram":    block["probability_histogram"],
            "worst_cases":  block["worst_cases"],
        }

        pred_new = (y_prob >= tau).astype(int)
        rows: list[dict] = []
        for src in sorted(pd.unique(source)):
            mask = source == src
            yt = y_true[mask]; yp = pred_new[mask]
            n = int(mask.sum())
            pred_safe  = int((yp == 0).sum())
            pred_risky = int((yp == 1).sum())
            acc = float((yt == yp).mean()) if n else float("nan")
            if (yt == 1).any():
                fn = int(((yp == 0) & (yt == 1)).sum())
                tp = int(((yp == 1) & (yt == 1)).sum())
                fsr = float(fn / (fn + tp)) if (fn + tp) else float("nan")
            else:
                fsr = None
            rows.append({
                "source":      src,
                "true_label":  int(yt[0]) if n else -1,
                "n":           n,
                "pred_safe":   pred_safe,
                "pred_risky":  pred_risky,
                "accuracy":    acc,
                "fsr":         fsr,
            })
        per_source[name] = rows

    # Phase 5 / @0.5 / @τ* bar chart.
    names = list(ext_metrics.keys())
    p5  = [ext_metrics[n]["fsr_phase5"] for n in names]
    p6a = [ext_metrics[n]["fsr_old"]    for n in names]
    p6b = [ext_metrics[n]["fsr_new"]    for n in names]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=[v * 100 for v in p5], name="Phase 5",
        marker_color="#94a3b8",
        text=[f"{v*100:.1f}%" for v in p5], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=names, y=[v * 100 for v in p6a], name="Phase 6 @ 0.5",
        marker_color="#f59e0b",
        text=[f"{v*100:.1f}%" for v in p6a], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=names, y=[v * 100 for v in p6b], name="Phase 6 @ τ* (cost-weighted)",
        marker_color="#2563eb",
        text=[f"{v*100:.1f}%" for v in p6b], textposition="outside",
    ))
    fig.add_hline(
        y=15, line_dash="dash", line_color="#059669",
        annotation_text="15% target", annotation_position="top right",
    )
    ymax = max(60, max([v * 100 for v in p5 + p6a + p6b]) + 10)
    fig.update_layout(
        title=dict(text="External Validation FSR — Phase 5 vs Phase 6 vs Phase 6 @ τ*",
                   font=dict(size=18), x=0.5),
        xaxis=dict(title="Model"),
        yaxis=dict(title="False-Safe Rate (%)", range=[0, ymax]),
        barmode="group", height=460, plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.22),
        margin=dict(l=60, r=30, t=70, b=90),
    )
    return ext_metrics, ext_figs, _fig_to_dict(fig), per_source


# -----------------------------------------------------------------------------
# Figure helpers
# -----------------------------------------------------------------------------
def _confusion_fig(cm: dict, title: str, accent: str = "#2563eb"):
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    total = tn + fp + fn + tp
    z = [[tn, fp], [fn, tp]]
    text = [
        [f"<b>{tn}</b><br>({tn/total:.1%})", f"<b>{fp}</b><br>({fp/total:.1%})"],
        [f"<b>{fn}</b><br>({fn/total:.1%})", f"<b>{tp}</b><br>({tp/total:.1%})"],
    ]
    fig = go.Figure(go.Heatmap(
        z=z, x=["Predicted: Safe", "Predicted: Risky"],
        y=["True: Safe", "True: Risky"],
        text=text, texttemplate="%{text}",
        textfont={"size": 17},
        colorscale=[[0, "#ffffff"], [1, accent]],
        showscale=False,
        hovertemplate="%{y} → %{x}<br>Count: %{z}<extra></extra>",
    ))
    subtitle = f"acc={cm['accuracy']:.1%} &bull; FSR={cm['fsr']:.1%}"
    fig.update_layout(
        title=dict(text=f"{title}<br><span style='font-size:12px;color:#6b7280'>"
                         f"{subtitle}</span>",
                   font=dict(size=15), x=0.5),
        xaxis=dict(title="Predicted", tickfont=dict(size=12)),
        yaxis=dict(title="Truth",     tickfont=dict(size=12)),
        height=380, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=70, r=30, t=90, b=50),
    )
    return fig


def _hist_fig(probs_safe, probs_risky, tau_new, t_high, title):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=probs_safe,  name="Safe (label=0)", nbinsx=30, opacity=0.75,
        marker_color="#2563eb",
    ))
    fig.add_trace(go.Histogram(
        x=probs_risky, name="Risky (label=1)", nbinsx=30, opacity=0.75,
        marker_color="#dc2626",
    ))
    fig.add_vline(
        x=tau_new, line_dash="dash", line_color="#059669", line_width=2,
        annotation_text=f"τ*={tau_new:.3f}", annotation_position="top right",
    )
    if t_high and t_high < 1.0:
        fig.add_vline(
            x=t_high, line_dash="dot", line_color="#7c3aed", line_width=2,
            annotation_text=f"T_high={t_high:.3f}", annotation_position="top right",
        )
    fig.update_layout(
        title=dict(text=title, font=dict(size=16), x=0.5),
        xaxis=dict(title="Calibrated Risk Probability", range=[0, 1]),
        yaxis=dict(title="Count"),
        barmode="overlay", legend=dict(x=0.75, y=0.95),
        height=420, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=60, r=30, t=70, b=60),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb")
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb")
    return fig


def _fig_to_dict(fig):
    return json.loads(pio.to_json(fig))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    print("Building Phase 6c fine-tune report ...")

    corpus_rows, corpus_total, split_rows = build_corpus()
    print(f"  corpus: {corpus_total} pages across {len(corpus_rows)} source folders")

    training = build_training_data()
    for name, d in training.items():
        print(f"  training[{name}]: {len(d['epochs'])} epochs")

    test_metrics, test_figs = build_test_metrics()
    for name, r in test_metrics.items():
        print(f"  test[{name}]: F1={r['f1']:.3f}  τ*={r['tau']:.3f}  "
              f"FSR@0.5={r['cm_old']['fsr']:.2%}  FSR@τ*={r['cm_new']['fsr']:.2%}")

    ext_metrics, ext_figs, ext_compare_fig, per_source = build_external()
    for name, r in ext_metrics.items():
        print(f"  ext[{name}]:  FSR Phase5={r['fsr_phase5']:.1%}  "
              f"@0.5={r['fsr_old']:.1%}  @τ*={r['fsr_new']:.1%}")

    report_data = {
        "generated_at":         datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "corpus":               corpus_rows,
        "corpus_total":         corpus_total,
        "splits":               split_rows,
        "training":             training,
        "test_metrics":         test_metrics,
        "test_figs":            test_figs,
        "external_metrics":     ext_metrics,
        "external_figs":        ext_figs,
        "external_fsr_compare": ext_compare_fig,
        "external_per_source":  per_source,
    }

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = (
        template
        .replace("__GENERATED_AT__", report_data["generated_at"])
        .replace("__REPORT_DATA__",  json.dumps(report_data, ensure_ascii=False))
    )
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"\nWrote {OUTPUT_PATH}  ({size_kb:,.0f} KB)")
    print("Open in a browser to view; attach the single HTML file to email / drive.")


if __name__ == "__main__":
    main()
