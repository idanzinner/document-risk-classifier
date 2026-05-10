#!/usr/bin/env python3
"""
build_dit_package.py — Assembles the DiT inference standalone package.

Creates a self-contained `dit_inference/` directory and zips it to
`dit_inference.zip` in the project root.  The zip can be sent to colleagues
who only need to install Python dependencies and run the notebook.

Usage:
    python scripts/build_dit_package.py

Output:
    <project_root>/dit_inference.zip
"""

import json
import pickle
import shutil
import sys
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent.resolve()
# Ensure src.models.calibrator is importable when loading the pickled calibrator
sys.path.insert(0, str(ROOT))

STAGING_ROOT = ROOT / "_dit_package"
PKG = STAGING_ROOT / "dit_inference"
OUT_ZIP = ROOT / "dit_inference.zip"

CHECKPOINT_SRC = ROOT / "checkpoints" / "dit" / "best_model.pt"
CALIBRATOR_SRC = ROOT / "checkpoints" / "dit" / "calibrator.pkl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_text(p: Path, content: str) -> None:
    p.write_text(content, encoding="utf-8")


def _write_init(p: Path) -> None:
    p.write_text("", encoding="utf-8")


# ---------------------------------------------------------------------------
# Re-save calibrator as plain dict (portable across environments)
# ---------------------------------------------------------------------------

def _resave_calibrator(src: Path, dst: Path) -> dict:
    """
    Loads the calibrator (may be a TemperatureCalibrator instance or already
    a dict) and re-saves as a plain dict {temperature, t_low, t_high}.
    """
    with open(src, "rb") as fh:
        obj = pickle.load(fh)

    if isinstance(obj, dict):
        state = obj
    else:
        # Full TemperatureCalibrator instance — extract attributes
        state = {
            "temperature": float(obj.temperature),
            "t_low": float(obj.t_low) if obj.t_low is not None else None,
            "t_high": float(obj.t_high) if obj.t_high is not None else None,
        }

    with open(dst, "wb") as fh:
        pickle.dump(state, fh)

    print(
        f"  calibrator: T={state['temperature']:.4f}, "
        f"T_low={state['t_low']}, T_high={state['t_high']}"
    )
    return state


# ---------------------------------------------------------------------------
# File contents
# ---------------------------------------------------------------------------

README = """\
# Hallucination-Risk Classifier — DiT Inference Package

Classifies Hebrew PDF pages as **safe for extraction**, **needs review**, or
**high hallucination risk** using a fine-tuned DiT (Document Image Transformer)
model.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On first run, `DiTClassifier` downloads the `microsoft/dit-base`
> architecture config (~400 MB) from HuggingFace and caches it locally.
> An internet connection is required the first time only.

### 2. Open the notebook

```bash
jupyter notebook notebooks/dit_inference_guide.ipynb
```

Run the cells from top to bottom.

---

## Directory Structure

```
dit_inference/
├── README.md
├── requirements.txt
├── checkpoints/
│   ├── best_model.pt       # Trained DiT weights (~343 MB)
│   └── calibrator.pkl      # Temperature calibrator + thresholds
├── configs/
│   └── inference.yaml      # Thresholds and data settings (reference)
├── notebooks/
│   └── dit_inference_guide.ipynb   # Step-by-step tutorial
├── sample_pdfs/            # Drop your PDF files here
└── src/
    ├── data/
    │   └── render_pdf.py   # PDF → 224×224 grayscale image
    ├── inference/
    │   └── service_schema.py  # RiskCategory, PredictionResponse
    └── models/
        ├── calibrator.py   # Temperature scaling calibrator
        └── dit_classifier.py  # DiT model (HuggingFace BEiT backbone)
```

---

## How It Works

1. **Render** — PyMuPDF converts the first PDF page to a 224×224 grayscale image at 150 DPI
2. **Transform** — `ToTensor` + `Normalize(mean=0.5, std=0.5)`
3. **Classify** — DiT backbone produces a raw logit (higher = more risky)
4. **Calibrate** — Temperature scaling converts logit → calibrated probability in [0, 1]
5. **Route** — Two thresholds assign a risk category:

| Category | Condition | Meaning |
|----------|-----------|---------|
| `safe_for_extraction` | prob < T_low | Page is safe for automated extraction |
| `review` | T_low ≤ prob ≤ T_high | Uncertain — route to human review |
| `high_hallucination_risk` | prob > T_high | Handwritten / questionnaire — will hallucinate |

---

## Hardware

The model runs on **CPU** by default. If a CUDA GPU or Apple MPS device is
available it is detected automatically. CPU inference takes ~1–3 seconds per
page depending on hardware.
"""

REQUIREMENTS = """\
torch>=2.1.0
torchvision>=0.16.0
transformers>=4.35.0
pymupdf>=1.23.0
Pillow>=10.0.0
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.11.0
scikit-learn>=1.3.0
pandas>=2.0.0
pydantic>=2.0.0
tqdm>=4.65.0
jupyter>=1.0.0
ipykernel>=6.0.0
"""

INFERENCE_YAML = """\
model:
  name: microsoft/dit-base
  num_classes: 1

thresholds:
  safe_upper: 0.3
  risky_lower: 0.7

data:
  dpi: 150
  image_size: 224
  grayscale: true
"""


# ---------------------------------------------------------------------------
# Notebook
# ---------------------------------------------------------------------------

def _nb_md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": f"md_{abs(hash(source)) % 10**8:08x}",
        "metadata": {},
        "source": source,
    }


def _nb_code(source: str, cell_id: str = "") -> dict:
    uid = cell_id or f"code_{abs(hash(source)) % 10**8:08x}"
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uid,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


NOTEBOOK_CELLS = [
    _nb_md("""\
# DiT Inference Guide — Hallucination-Risk Classifier

End-to-end walkthrough: **PDF file in → risk category out.**

This notebook loads the trained DiT (Document Image Transformer) checkpoint
and temperature calibrator, then shows how to classify PDF pages as:

| Category | Probability | Meaning |
|----------|-------------|---------|
| `safe_for_extraction` | prob < T_low | Safe for automated text extraction |
| `review` | T_low ≤ prob ≤ T_high | Uncertain — route to human review |
| `high_hallucination_risk` | prob > T_high | Handwritten / questionnaire — extraction will hallucinate |

**Requirements:** Run `pip install -r ../requirements.txt` before starting.
On first run, the DiT model config (~400 MB) is downloaded from HuggingFace.
"""),

    # ── Section 1: Setup ──────────────────────────────────────────────────────
    _nb_md("## 1 — Setup\n\nImports, paths, and device selection."),

    _nb_code("""\
import sys
from pathlib import Path

# Add the package root to sys.path so `src` imports work
ROOT = Path("..").resolve()
sys.path.insert(0, str(ROOT))

import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

from src.data.render_pdf import render_page
from src.models.dit_classifier import DiTClassifier
from src.models.calibrator import TemperatureCalibrator
from src.inference.service_schema import RiskCategory

# ── Paths ──────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = ROOT / "checkpoints" / "best_model.pt"
CALIBRATOR_PATH = ROOT / "checkpoints" / "calibrator.pkl"
SAMPLE_PDF_DIR  = ROOT / "sample_pdfs"

# ── Preprocessing constants (must match training) ──────────────────────────
DPI        = 150
IMAGE_SIZE = (224, 224)
GRAYSCALE  = True
MEAN, STD  = [0.5], [0.5]

print(f"ROOT : {ROOT}")
print(f"Checkpoint : {CHECKPOINT_PATH}  exists={CHECKPOINT_PATH.exists()}")
print(f"Calibrator : {CALIBRATOR_PATH}  exists={CALIBRATOR_PATH.exists()}")
""", "setup"),

    # ── Section 2: Load model & calibrator ───────────────────────────────────
    _nb_md("## 2 — Load Model & Calibrator"),

    _nb_code("""\
# ── Device ────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# ── Model ─────────────────────────────────────────────────────────────────
# DiTClassifier calls AutoModel.from_pretrained("microsoft/dit-base") which
# downloads ~400 MB from HuggingFace on the first run and caches locally.
model = DiTClassifier(model_name="microsoft/dit-base", num_classes=1)

# The checkpoint is saved with extra arrays (logits/labels) alongside weights,
# so weights_only=False is required. This file was produced by our training
# pipeline and is safe to load.
ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
state_dict = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()
print(f"Model loaded  — stage={ckpt.get('stage', 'N/A')}, "
      f"val_f1={ckpt.get('val_f1', float('nan')):.4f}")

# ── Calibrator ────────────────────────────────────────────────────────────
calibrator = TemperatureCalibrator()
calibrator.load(str(CALIBRATOR_PATH))
print(f"Calibrator    — temperature={calibrator.temperature:.4f}")
print(f"Thresholds    — T_low={calibrator.t_low:.4f}, T_high={calibrator.t_high:.4f}")

T_LOW  = calibrator.t_low
T_HIGH = calibrator.t_high
""", "load_model"),

    # ── Section 3: Preprocessing pipeline ────────────────────────────────────
    _nb_md("## 3 — Preprocessing Pipeline\n\nThe same pipeline used during training."),

    _nb_code("""\
# Grayscale ToTensor → Normalize(0.5, 0.5)
_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def preprocess(img: Image.Image) -> torch.Tensor:
    \"\"\"PIL Image (mode L) → normalised FloatTensor [1, 224, 224].\"\"\"
    return _transform(img)


def _category(prob: float) -> str:
    \"\"\"Maps a calibrated probability to a risk category string.\"\"\"
    if prob < T_LOW:
        return RiskCategory.SAFE.value
    if prob > T_HIGH:
        return RiskCategory.HIGH_RISK.value
    return RiskCategory.REVIEW.value


def predict_pdf(pdf_path: str | Path) -> dict:
    \"\"\"
    End-to-end prediction for a single PDF file.

    Returns a dict with keys: file, probability, category, raw_logit.
    \"\"\"
    img = render_page(str(pdf_path), dpi=DPI, grayscale=GRAYSCALE, target_size=IMAGE_SIZE)
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)   # [1, 1, 224, 224]

    with torch.no_grad():
        logit = model(tensor).view(-1).cpu().item()

    prob = float(calibrator.predict(np.array([logit]))[0])
    return {
        "file": Path(pdf_path).name,
        "probability": round(prob, 4),
        "category": _category(prob),
        "raw_logit": round(logit, 4),
    }

print("Pipeline ready.")
""", "pipeline"),

    # ── Section 4: Single PDF inference ──────────────────────────────────────
    _nb_md("""\
## 4 — Single PDF Inference

Replace `PDF_PATH` with the path to any PDF file you want to classify.
"""),

    _nb_code("""\
# ── Change this path to your PDF ──────────────────────────────────────────
PDF_PATH = SAMPLE_PDF_DIR / "example.pdf"   # ← replace with your file

if not Path(PDF_PATH).exists():
    print(f"[DEMO] File not found: {PDF_PATH}")
    print("Place a PDF in the sample_pdfs/ folder or update PDF_PATH above.")
else:
    result = predict_pdf(PDF_PATH)

    # Display rendered page
    img = render_page(str(PDF_PATH), dpi=DPI, grayscale=GRAYSCALE, target_size=IMAGE_SIZE)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(img, cmap="gray")
    ax.set_title(
        f"{result['file']}\\n"
        f"Category: {result['category']}\\n"
        f"Probability: {result['probability']:.4f}",
        fontsize=9,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    print(result)
""", "single_inference"),

    # ── Section 5: Batch inference ────────────────────────────────────────────
    _nb_md("""\
## 5 — Batch Inference (Folder of PDFs)

`score_directory()` walks a folder recursively, classifies every PDF, and
returns a pandas DataFrame.
"""),

    _nb_code("""\
def score_directory(pdf_dir: str | Path, glob: str = "**/*.pdf") -> pd.DataFrame:
    \"\"\"
    Classifies all PDF files under pdf_dir.

    Args:
        pdf_dir: Root directory containing PDF files (searched recursively).
        glob:    Glob pattern (default: all .pdf files, recursive).

    Returns:
        DataFrame with columns: file, path, probability, category, raw_logit.
    \"\"\"
    pdf_dir = Path(pdf_dir)
    pdf_files = sorted(pdf_dir.glob(glob))

    if not pdf_files:
        print(f"No PDF files found under {pdf_dir}")
        return pd.DataFrame()

    records = []
    for pdf_path in tqdm(pdf_files, desc="Scoring PDFs"):
        try:
            rec = predict_pdf(pdf_path)
            rec["path"] = str(pdf_path)
            records.append(rec)
        except Exception as exc:
            records.append({
                "file": pdf_path.name,
                "path": str(pdf_path),
                "probability": None,
                "category": "error",
                "raw_logit": None,
                "error": str(exc),
            })

    return pd.DataFrame(records)


# ── Run batch over sample_pdfs/ ────────────────────────────────────────────
# Change SAMPLE_PDF_DIR to any folder you want to scan.
results_df = score_directory(SAMPLE_PDF_DIR)

if results_df.empty:
    print(f"No PDFs found in {SAMPLE_PDF_DIR}. Add PDFs to sample_pdfs/ and re-run.")
else:
    print(f"Scored {len(results_df)} file(s)")
    display(results_df)
""", "batch_inference"),

    # ── Section 6: Summary & histogram ───────────────────────────────────────
    _nb_md("## 6 — Results Summary"),

    _nb_code("""\
if not results_df.empty and "probability" in results_df.columns:
    # Category counts
    print("Category counts:")
    print(results_df["category"].value_counts().to_string())
    print()

    # Histogram of calibrated probabilities
    valid = results_df["probability"].dropna().astype(float)
    if len(valid) > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(valid, bins=30, edgecolor="white", color="#3a7ebf")
        ax.axvline(T_LOW, color="orange", linestyle="--",
                   label=f"T_low = {T_LOW:.3f}")
        ax.axvline(T_HIGH, color="red", linestyle="--",
                   label=f"T_high = {T_HIGH:.3f}")
        ax.set_xlabel("Calibrated Probability (higher = more risky)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Risk Scores")
        ax.legend()
        plt.tight_layout()
        plt.show()
else:
    print("No results to summarise yet.")
""", "summary"),

    # ── Section 7: Export ─────────────────────────────────────────────────────
    _nb_md("## 7 — Export Results"),

    _nb_code("""\
if not results_df.empty:
    output_dir = ROOT / "inference_results"
    output_dir.mkdir(exist_ok=True)

    csv_path  = output_dir / "predictions.csv"
    json_path = output_dir / "predictions.json"

    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    results_df.to_json(json_path, orient="records", indent=2, force_ascii=False)

    print(f"Saved CSV  → {csv_path}")
    print(f"Saved JSON → {json_path}")
else:
    print("Nothing to export yet.")
""", "export"),
]


def _make_notebook(cells: list) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"=== Building DiT inference package ===")
    print(f"  Source root : {ROOT}")
    print(f"  Staging dir : {PKG}")
    print(f"  Output zip  : {OUT_ZIP}")
    print()

    # 0. Clean staging area
    if STAGING_ROOT.exists():
        shutil.rmtree(STAGING_ROOT)

    # 1. Create directory skeleton
    for sub in [
        PKG / "checkpoints",
        PKG / "configs",
        PKG / "notebooks",
        PKG / "sample_pdfs",
        PKG / "src" / "models",
        PKG / "src" / "data",
        PKG / "src" / "inference",
    ]:
        _mkdir(sub)

    # 2. Copy source files
    print("[1/7] Copying source files …")
    src_files = {
        ROOT / "src" / "models" / "dit_classifier.py": PKG / "src" / "models" / "dit_classifier.py",
        ROOT / "src" / "models" / "calibrator.py":     PKG / "src" / "models" / "calibrator.py",
        ROOT / "src" / "data" / "render_pdf.py":       PKG / "src" / "data" / "render_pdf.py",
        ROOT / "src" / "inference" / "service_schema.py": PKG / "src" / "inference" / "service_schema.py",
    }
    for src, dst in src_files.items():
        shutil.copy2(src, dst)
        print(f"  copied {src.relative_to(ROOT)} → {dst.relative_to(PKG)}")

    # __init__.py stubs
    for init_dir in [
        PKG / "src",
        PKG / "src" / "models",
        PKG / "src" / "data",
        PKG / "src" / "inference",
    ]:
        _write_init(init_dir / "__init__.py")

    # .gitkeep for empty folder
    _write_text(PKG / "sample_pdfs" / ".gitkeep", "")

    # 3. Copy checkpoint
    print("[2/7] Copying checkpoint (~343 MB) …")
    shutil.copy2(CHECKPOINT_SRC, PKG / "checkpoints" / "best_model.pt")
    print(f"  {CHECKPOINT_SRC.name} → checkpoints/best_model.pt")

    # 4. Re-save calibrator as plain dict
    print("[3/7] Re-saving calibrator …")
    _resave_calibrator(CALIBRATOR_SRC, PKG / "checkpoints" / "calibrator.pkl")

    # 5. Write inference.yaml
    print("[4/7] Writing configs/inference.yaml …")
    _write_text(PKG / "configs" / "inference.yaml", INFERENCE_YAML)

    # 6. Write requirements.txt + README
    print("[5/7] Writing requirements.txt and README.md …")
    _write_text(PKG / "requirements.txt", REQUIREMENTS)
    _write_text(PKG / "README.md", README)

    # 7. Generate notebook
    print("[6/7] Generating notebook …")
    nb = _make_notebook(NOTEBOOK_CELLS)
    nb_path = PKG / "notebooks" / "dit_inference_guide.ipynb"
    _write_text(nb_path, json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"  notebook → notebooks/dit_inference_guide.ipynb")

    # 8. Zip
    print("[7/7] Creating zip archive …")
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()

    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fpath in sorted(PKG.rglob("*")):
            if fpath.is_file():
                arcname = fpath.relative_to(STAGING_ROOT)
                zf.write(fpath, arcname)
                print(f"  + {arcname}")

    size_mb = OUT_ZIP.stat().st_size / (1024 ** 2)
    print()
    print(f"=== Done ===")
    print(f"  {OUT_ZIP.name}  ({size_mb:.1f} MB)")
    print()
    print("To use:")
    print("  1. Send dit_inference.zip to your colleagues")
    print("  2. They unzip it and run: pip install -r requirements.txt")
    print("  3. Open notebooks/dit_inference_guide.ipynb in Jupyter")


if __name__ == "__main__":
    main()
