"""
predict.py — Inference entry point for single pages and batch PDF directories.

Loads a trained model (ResNet/ViT/DiT) + temperature calibrator from disk,
renders each requested PDF page on the fly, and returns a structured
PredictionResponse with risk category and calibrated confidence.
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from src.inference.service_schema import PredictionRequest, PredictionResponse, RiskCategory
from src.models.calibrator import TemperatureCalibrator
from src.models.dit_classifier import DiTClassifier
from src.models.resnet_baseline import ResNetClassifier
from src.models.vit_baseline import ViTClassifier
from src.data.render_pdf import render_page
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Normalisation constants — must match training (see dataset.py)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_GRAY_MEAN = [0.5]
_GRAY_STD = [0.5]

_DEFAULT_T_LOW = 0.3
_DEFAULT_T_HIGH = 0.7


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _build_inference_transform(grayscale: bool) -> transforms.Compose:
    """Returns a ToTensor + Normalize pipeline matching the training config."""
    if grayscale:
        normalize = transforms.Normalize(mean=_GRAY_MEAN, std=_GRAY_STD)
    else:
        normalize = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
    return transforms.Compose([transforms.ToTensor(), normalize])


def _pil_to_tensor(img: Image.Image, grayscale: bool) -> torch.Tensor:
    """Converts a PIL Image to a normalised FloatTensor [C, H, W]."""
    transform = _build_inference_transform(grayscale)
    return transform(img)


# ---------------------------------------------------------------------------
# Thresholding
# ---------------------------------------------------------------------------

def _logit_to_category(
    prob: float,
    t_low: float,
    t_high: float,
) -> RiskCategory:
    """Maps a calibrated probability to a RiskCategory using (T_low, T_high)."""
    if prob < t_low:
        return RiskCategory.SAFE
    if prob > t_high:
        return RiskCategory.HIGH_RISK
    return RiskCategory.REVIEW


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _build_model(model_type: str, config: dict) -> torch.nn.Module:
    """
    Instantiates the correct model class from model_type string and config dict.

    Args:
        model_type: One of 'resnet50', 'efficientnet_b0', 'vit', 'dit'.
        config:     Parsed YAML config dict (used for num_classes).

    Returns:
        Uninitialised model (weights loaded separately).
    """
    num_classes: int = config.get("model", {}).get("num_classes", 1)

    if model_type in ("resnet50", "efficientnet_b0"):
        return ResNetClassifier(
            model_name=model_type,
            pretrained=False,
            num_classes=num_classes,
        )
    if model_type == "vit":
        model_name = config.get("model", {}).get("name", "vit_base_patch16_224")
        return ViTClassifier(
            model_name=model_name,
            pretrained=False,
            num_classes=num_classes,
        )
    if model_type == "dit":
        model_name = config.get("model", {}).get("name", "microsoft/dit-base")
        return DiTClassifier(
            model_name=model_name,
            num_classes=num_classes,
        )
    raise ValueError(
        f"Unknown model_type '{model_type}'. "
        "Expected one of: 'resnet50', 'efficientnet_b0', 'vit', 'dit'."
    )


# ---------------------------------------------------------------------------
# Pipeline loader
# ---------------------------------------------------------------------------

def load_pipeline(
    checkpoint_path: str,
    calibrator_path: str,
    model_type: str,
    config_path: str,
    device: str = "auto",
) -> tuple:
    """
    Loads model + calibrator + thresholds from disk.

    Args:
        checkpoint_path: Path to the model state_dict (.pt / .pth).
        calibrator_path: Path to the pickled TemperatureCalibrator.
        model_type:      One of 'resnet50', 'efficientnet_b0', 'vit', 'dit'.
        config_path:     Path to a YAML config file (baseline.yaml / dit.yaml).
        device:          'auto' selects CUDA → MPS → CPU automatically.

    Returns:
        Tuple (model, calibrator, thresholds, device_str) where:
            model        — nn.Module in eval mode on the target device,
            calibrator   — fitted TemperatureCalibrator,
            thresholds   — dict {'T_low': float, 'T_high': float},
            device_str   — resolved device string, e.g. 'cuda', 'mps', 'cpu'.
    """
    # --- Resolve device ---
    if device == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        elif torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    else:
        device_str = device
    logger.info("Using device: %s", device_str)

    # --- Load YAML config ---
    with open(config_path, "r", encoding="utf-8") as fh:
        config: dict = yaml.safe_load(fh)

    # --- Load checkpoint ---
    raw_ckpt = torch.load(checkpoint_path, map_location=device_str, weights_only=False)

    # Patch config.model.name from the checkpoint so _build_model uses the right backbone.
    # baseline.yaml always says "resnet50", which would be wrong for ViT.
    if isinstance(raw_ckpt, dict) and "model_name" in raw_ckpt:
        if "model" not in config:
            config["model"] = {}
        config["model"]["name"] = raw_ckpt["model_name"]

    # --- Build model and load weights ---
    model = _build_model(model_type, config)
    state_dict = raw_ckpt
    # Unwrap common checkpoint wrappers (e.g. {'model_state_dict': ...})
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model = model.to(device_str)
    model.eval()

    # --- Load calibrator ---
    # Support both object-pickled and dict-pickled calibrators.
    import pickle as _pickle
    with open(calibrator_path, "rb") as _fh:
        _raw = _pickle.load(_fh)
    if isinstance(_raw, TemperatureCalibrator):
        calibrator = _raw
    else:
        calibrator = TemperatureCalibrator()
        calibrator.load(calibrator_path)

    # --- Resolve thresholds ---
    t_low: Optional[float] = getattr(calibrator, "t_low", None)
    t_high: Optional[float] = getattr(calibrator, "t_high", None)

    if t_low is None or t_high is None:
        # Fall back to inference.yaml thresholds or hard-coded defaults
        yaml_thresholds = config.get("thresholds", {})
        t_low = float(yaml_thresholds.get("safe_upper", _DEFAULT_T_LOW))
        t_high = float(yaml_thresholds.get("risky_lower", _DEFAULT_T_HIGH))
        logger.warning(
            "Calibrator has no stored thresholds; using T_low=%.3f, T_high=%.3f",
            t_low,
            t_high,
        )

    thresholds = {"T_low": float(t_low), "T_high": float(t_high)}
    logger.info("Thresholds: T_low=%.3f, T_high=%.3f", t_low, t_high)

    return model, calibrator, thresholds, device_str


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

def predict_batch(
    requests: list[PredictionRequest],
    model: torch.nn.Module,
    calibrator: TemperatureCalibrator,
    thresholds: dict[str, float],
    rendered_dir: str,
    device: str = "cpu",
    batch_size: int = 32,
) -> list[PredictionResponse]:
    """
    Run batch inference over a list of PredictionRequests.

    For each request the pre-rendered PNG is loaded from
    ``rendered_dir / request.file_path`` (with .png substituted if the path
    ends in .pdf).  Missing images are returned with confidence=0.5 and
    category=REVIEW.

    Args:
        requests:     List of PredictionRequest objects.
        model:        Loaded classifier (ResNet / ViT / DiT) in eval mode.
        calibrator:   Fitted TemperatureCalibrator.
        thresholds:   Dict {'T_low': float, 'T_high': float}.
        rendered_dir: Root directory that holds pre-rendered PNG files.
        device:       Target device string ('cpu', 'cuda', 'mps').
        batch_size:   Number of images per forward pass.

    Returns:
        List of PredictionResponse objects in the same order as *requests*.
    """
    t_low = thresholds["T_low"]
    t_high = thresholds["T_high"]
    rendered_root = Path(rendered_dir)

    # Build (tensor, request) pairs; use None tensor for missing files
    items: list[tuple[Optional[torch.Tensor], PredictionRequest]] = []
    for req in requests:
        img_path = rendered_root / req.file_path
        # Accept .pdf paths by swapping extension to .png
        if img_path.suffix.lower() == ".pdf":
            img_path = img_path.with_suffix(".png")

        if not img_path.exists():
            logger.warning("Rendered image not found: %s", img_path)
            items.append((None, req))
            continue

        try:
            img = Image.open(str(img_path))
            grayscale = img.mode == "L"
            tensor = _pil_to_tensor(img, grayscale)
            items.append((tensor, req))
        except (OSError, ValueError) as exc:
            logger.warning("Failed to load image %s: %s", img_path, exc)
            items.append((None, req))

    responses: list[PredictionResponse] = []
    # Process in batches
    for batch_start in range(0, len(items), batch_size):
        batch_items = items[batch_start : batch_start + batch_size]

        valid_tensors: list[torch.Tensor] = []
        valid_indices: list[int] = []
        for local_idx, (tensor, _req) in enumerate(batch_items):
            if tensor is not None:
                valid_tensors.append(tensor)
                valid_indices.append(local_idx)

        # Map local batch index → logit
        logit_map: dict[int, float] = {}
        if valid_tensors:
            batch_tensor = torch.stack(valid_tensors).to(device)
            with torch.no_grad():
                raw_logits: torch.Tensor = model(batch_tensor)   # [N, 1] or [N]
            raw_logits = raw_logits.view(-1).cpu().numpy()
            for mapped_idx, (logit_val, local_idx) in enumerate(
                zip(raw_logits, valid_indices)
            ):
                logit_map[local_idx] = float(logit_val)

        for local_idx, (tensor, req) in enumerate(batch_items):
            if local_idx not in logit_map:
                # Missing / unloadable image
                responses.append(
                    PredictionResponse(
                        file_path=req.file_path,
                        page_num=req.page_num,
                        risk_category=RiskCategory.REVIEW,
                        confidence=0.5,
                        raw_logit=0.0,
                    )
                )
                continue

            raw_logit = logit_map[local_idx]
            prob = float(calibrator.predict(np.array([raw_logit]))[0])
            category = _logit_to_category(prob, t_low, t_high)
            responses.append(
                PredictionResponse(
                    file_path=req.file_path,
                    page_num=req.page_num,
                    risk_category=category,
                    confidence=prob,
                    raw_logit=raw_logit,
                )
            )

    return responses


# ---------------------------------------------------------------------------
# Single PDF inference (end-to-end with on-the-fly rendering)
# ---------------------------------------------------------------------------

def predict_single(
    pdf_path: str,
    model: torch.nn.Module,
    calibrator: TemperatureCalibrator,
    thresholds: dict[str, float],
    dpi: int = 150,
    device: str = "cpu",
) -> PredictionResponse:
    """
    End-to-end prediction for a single PDF file.

    Renders the first page at *dpi* DPI, preprocesses it to a tensor,
    runs it through *model*, calibrates the logit, and applies thresholds.

    Args:
        pdf_path:   Path to the PDF file (supports Hebrew filenames).
        model:      Loaded classifier in eval mode.
        calibrator: Fitted TemperatureCalibrator.
        thresholds: Dict {'T_low': float, 'T_high': float}.
        dpi:        Rendering resolution (must match training config).
        device:     Target device string.

    Returns:
        PredictionResponse for page 1 of the PDF.
    """
    t_low = thresholds["T_low"]
    t_high = thresholds["T_high"]

    img = render_page(pdf_path, dpi=dpi, grayscale=True, target_size=(224, 224))
    tensor = _pil_to_tensor(img, grayscale=True).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        raw_logit_tensor = model(tensor)

    raw_logit = float(raw_logit_tensor.view(-1).cpu().numpy()[0])
    prob = float(calibrator.predict(np.array([raw_logit]))[0])
    category = _logit_to_category(prob, t_low, t_high)

    return PredictionResponse(
        file_path=pdf_path,
        page_num=1,
        risk_category=category,
        confidence=prob,
        raw_logit=raw_logit,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run batch hallucination-risk inference over a directory of PDFs."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the model checkpoint (.pt).",
    )
    parser.add_argument(
        "--calibrator",
        required=True,
        help="Path to the pickled TemperatureCalibrator.",
    )
    parser.add_argument(
        "--model_type",
        required=True,
        choices=["resnet50", "efficientnet_b0", "vit", "dit"],
        help="Model architecture identifier.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config file (e.g. configs/inference.yaml).",
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing pre-rendered PNG files (mirrors PDF layout).",
    )
    parser.add_argument(
        "--output_json",
        required=True,
        help="Path for the output JSON file with prediction results.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of images per forward pass (default: 32).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run on: 'auto', 'cpu', 'cuda', 'mps' (default: auto).",
    )
    args = parser.parse_args()

    # Load pipeline
    model, calibrator, thresholds, resolved_device = load_pipeline(
        checkpoint_path=args.checkpoint,
        calibrator_path=args.calibrator,
        model_type=args.model_type,
        config_path=args.config,
        device=args.device,
    )

    # Discover all PNG files in input_dir
    input_root = Path(args.input_dir)
    png_files = sorted(input_root.rglob("*.png"))
    if not png_files:
        logger.warning("No PNG files found under %s", input_root)

    requests_list: list[PredictionRequest] = [
        PredictionRequest(file_path=str(p.relative_to(input_root)))
        for p in png_files
    ]
    logger.info("Running inference on %d files …", len(requests_list))

    predictions = predict_batch(
        requests=requests_list,
        model=model,
        calibrator=calibrator,
        thresholds=thresholds,
        rendered_dir=args.input_dir,
        device=resolved_device,
        batch_size=args.batch_size,
    )

    # Serialise results
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = [resp.model_dump() for resp in predictions]
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    logger.info("Results written to %s", output_path)
