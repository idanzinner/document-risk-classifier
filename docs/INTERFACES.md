# Interface Contract

## Overview

This document defines the data schemas and module APIs for the Hebrew PDF Hallucination-Risk Classifier.
All modules in `src/` are expected to conform to the contracts below.
Changes to these interfaces must be reflected here before implementation.

---

## Data Schema

### `data/metadata.csv` — Master page-level index

| Column | Type | Description |
|---|---|---|
| `file_path` | str | Path to the rendered PNG, relative to `data/rendered_pages/` |
| `page_num` | int | 1-indexed page number within the source PDF |
| `institution` | str | Institution identifier (used as grouping key for splits) |
| `template_family` | str | Document template family (e.g. `questionnaire_v1`, `standard_form`) |
| `label_binary` | int | Binary label: `0` = safe, `1` = risky |
| `D` | int | Density rubric score: 0–3 (3 = dense/structured, lower = harder to extract) |
| `H` | int | Handwriting rubric score: 0–3 (3 = heavy handwriting) |
| `S` | int | Scan quality rubric score: 0–3 (3 = poor scan) |
| `L` | int | Layout complexity rubric score: 0–3 (3 = complex layout) |
| `risk_score` | int | Composite risk: `(3 - D) + H + S + L`, range 0–12 |
| `split` | str | Assigned split: `train`, `val`, or `test` |

### `data/labels_binary.csv` — Minimal annotation file

| Column | Type | Description |
|---|---|---|
| `file_path` | str | Same path key as metadata.csv |
| `page_num` | int | 1-indexed page number |
| `label_binary` | int | `0` = safe, `1` = risky |

### `data/labels_rubric.csv` — Full rubric annotation file

| Column | Type | Description |
|---|---|---|
| `file_path` | str | Same path key as metadata.csv |
| `page_num` | int | 1-indexed page number |
| `D` | int | Density score 0–3 |
| `H` | int | Handwriting score 0–3 |
| `S` | int | Scan quality score 0–3 |
| `L` | int | Layout complexity score 0–3 |
| `risk_score` | int | Computed: `(3 - D) + H + S + L` |
| `label_ternary` | str | `safe_for_extraction`, `review`, or `high_hallucination_risk` |

---

## `src/data/render_pdf.py`

```python
def render_page(
    pdf_path: str,
    dpi: int = 150,
    grayscale: bool = True,
    target_size: tuple[int, int] = (224, 224),
) -> PIL.Image.Image:
    """
    Renders a single-page PDF to a PIL Image.
    - Grayscale conversion if grayscale=True
    - Preserves aspect ratio, pads to target_size with white
    - Deterministic (same input always gives same output)
    Returns: PIL.Image in mode 'L' (grayscale) or 'RGB'
    """

def render_all(
    pdf_dir: str,
    output_dir: str,
    dpi: int = 150,
    grayscale: bool = True,
    target_size: tuple[int, int] = (224, 224),
    skip_existing: bool = True,
) -> list[dict]:
    """
    Renders all PDFs in pdf_dir, saves PNG files to output_dir.
    Returns list of dicts with keys: pdf_path, rendered_path, status
    status is one of: 'rendered', 'skipped', 'error'
    """
```

---

## `src/data/dataset.py`

```python
class HallucinationRiskDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        metadata_csv: str,
        split: str,                     # 'train', 'val', or 'test'
        rendered_dir: str,
        transform: Optional[Callable] = None,
        augment: bool = False,
    ):
        """
        split: filters rows where metadata['split'] == split
        Returns (image_tensor: FloatTensor[C,H,W], label: FloatTensor[1])
        label is label_binary column (0.0 or 1.0)
        """

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]: ...
```

---

## `src/data/splits.py`

```python
def create_grouped_splits(
    metadata_df: pd.DataFrame,
    group_col: str,
    ratios: dict[str, float],           # e.g. {'train': 0.7, 'val': 0.15, 'test': 0.15}
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Assigns split column to metadata_df. Groups are kept whole.
    Returns metadata_df with 'split' column added.
    """

def create_kfold_splits(
    metadata_df: pd.DataFrame,
    group_col: str,
    n_folds: int = 5,
    random_state: int = 42,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Returns list of (train_df, val_df) tuples for grouped k-fold CV.
    """

def save_splits(metadata_df: pd.DataFrame, output_dir: str) -> None:
    """Saves train.csv, val.csv, test.csv to output_dir."""

def load_splits(splits_dir: str) -> dict[str, pd.DataFrame]:
    """Returns {'train': df, 'val': df, 'test': df}"""
```

---

## `src/models/` — Model interface

All model classes expose a common interface:

```python
class BaseClassifier(nn.Module):
    num_classes: int = 1    # binary: single logit output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: FloatTensor[B, C, H, W]  (C=1 or 3, H=W=224)
        returns: logits FloatTensor[B, 1]
        """
```

### `ResNetClassifier` (`src/models/resnet_baseline.py`)

```python
class ResNetClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = 'resnet50',
        pretrained: bool = True,
        num_classes: int = 1,
    ): ...
```

### `ViTClassifier` (`src/models/vit_baseline.py`)

```python
class ViTClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = 'vit_base_patch16_224',
        pretrained: bool = True,
        num_classes: int = 1,
    ): ...
```

### `DiTClassifier` (`src/models/dit_classifier.py`)

```python
class DiTClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = 'microsoft/dit-base',
        num_classes: int = 1,
    ): ...

    def freeze_backbone(self) -> None: ...
    def unfreeze_top_blocks(self, n_blocks: int = 2) -> None: ...
    def unfreeze_all(self) -> None: ...
```

### `TemperatureCalibrator` (`src/models/calibrator.py`)

```python
class TemperatureCalibrator:
    def calibrate(
        self,
        logits: np.ndarray,             # shape [N]
        labels: np.ndarray,             # shape [N], binary int
    ) -> None:
        """Fits temperature parameter via NLL minimization on validation set."""

    def predict(self, logits: np.ndarray) -> np.ndarray:
        """Returns calibrated probabilities, shape [N]."""

    def get_thresholds(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        target_false_safe_rate: float = 0.05,
    ) -> dict[str, float]:
        """Returns {'T_low': float, 'T_high': float} for safe/review/risky mapping."""

    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

---

## `src/utils/metrics.py`

```python
def compute_metrics(
    y_true: np.ndarray,                 # binary labels [N]
    y_prob: np.ndarray,                 # calibrated probabilities [N]
    thresholds: dict[str, float],       # {'T_low': float, 'T_high': float}
) -> dict:
    """
    Returns dict with keys:
    - f1, precision_safe, recall_risky, false_safe_rate, review_rate
    - roc_auc, pr_auc, ece
    """

def compute_per_institution_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    institutions: np.ndarray,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """Returns DataFrame with per-institution F1, recall, false_safe_rate."""

def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error."""
```

---

## `src/inference/service_schema.py`

```python
class RiskCategory(str, Enum):
    SAFE = "safe_for_extraction"
    REVIEW = "review"
    HIGH_RISK = "high_hallucination_risk"

class PredictionRequest(BaseModel):
    file_path: str
    page_num: int = 1

class PredictionResponse(BaseModel):
    file_path: str
    page_num: int
    risk_category: RiskCategory
    confidence: float                   # calibrated probability [0, 1]
    raw_logit: float
```

---

## Error Analysis Log Schema

Used by `src/train/evaluate.py` when writing the per-page error analysis CSV.

| Column | Type | Description |
|---|---|---|
| `file_path` | str | Source PDF path |
| `page_num` | int | 1-indexed page |
| `true_label` | int | Ground-truth binary label |
| `predicted_category` | str | One of the three RiskCategory values |
| `confidence` | float | Calibrated probability |
| `institution` | str | Institution identifier |
| `template_family` | str | Document template family |
| `risk_score` | int | Composite rubric score |
| `D` | int | Density rubric score |
| `H` | int | Handwriting rubric score |
| `S` | int | Scan quality rubric score |
| `L` | int | Layout complexity rubric score |
| `scan_quality_note` | str | Free-text annotator note on scan quality |
| `handwriting_note` | str | Free-text annotator note on handwriting |
