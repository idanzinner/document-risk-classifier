"""
device.py — Device selection and MPS/CUDA helpers.

Usage:
    from src.utils.device import get_device, prepare_model, prepare_input, mps_sync, mps_empty_cache

    device = get_device("auto")
    model  = prepare_model(model, device)          # .to(device)
    images = prepare_input(images, device)          # .to(device, non_blocking)
"""

import torch
import torch.nn as nn


def get_device(preference: str = "auto") -> torch.device:
    """
    Return the best available torch.device based on the requested preference.

    Args:
        preference: One of "auto" | "cuda" | "mps" | "cpu".
                    "auto" selects CUDA > MPS > CPU in that priority order.
                    Explicit values fall back to CPU if the requested backend
                    is unavailable.

    Returns:
        torch.device
    """
    preference = preference.lower().strip()

    if preference == "cpu":
        return torch.device("cpu")

    if preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if preference == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # "auto" — prefer CUDA > MPS > CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_model(model: nn.Module, device: torch.device) -> nn.Module:
    """
    Move model to *device*.

    Note: channels_last (NHWC) is intentionally NOT applied. While Metal
    operates in NHWC natively, timm backbones use internal .view() operations
    whose backward pass is incompatible with channels_last memory format.
    PyTorch >= 2.11 handles the format conversion transparently on MPS.
    """
    return model.to(device)


def prepare_input(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move a batch of images to *device* with non-blocking transfer."""
    return images.to(device, non_blocking=True)


def mps_sync() -> None:
    """
    Force MPS command-queue synchronization.

    Call after loss.backward() and before grad clipping to surface NaN
    corruption early on Apple Silicon. No-op when MPS is not available.
    """
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def mps_empty_cache() -> None:
    """
    Release MPS memory back to the OS.

    Call at the end of each epoch to prevent Metal memory pressure from
    building up over long training runs. No-op when MPS is not available.
    """
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
