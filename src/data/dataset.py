"""
dataset.py — PyTorch Dataset for hallucination-risk page images.

Loads rendered page PNGs and corresponding binary labels from
metadata.csv, applying optional augmentation transforms.
"""

import logging
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

# ImageNet normalisation constants (for RGB)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# Mid-grey normalisation constants (for grayscale)
_GRAY_MEAN = [0.5]
_GRAY_STD = [0.5]


def _build_transform(grayscale: bool, augment: bool) -> Callable:
    """Builds a torchvision transform pipeline."""
    aug_ops: list = []
    if augment:
        aug_ops = [
            transforms.RandomRotation(degrees=3, fill=255),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.RandomPerspective(distortion_scale=0.05, p=0.3, fill=255),
        ]

    to_tensor = transforms.ToTensor()
    if grayscale:
        normalize = transforms.Normalize(mean=_GRAY_MEAN, std=_GRAY_STD)
    else:
        normalize = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

    return transforms.Compose(aug_ops + [to_tensor, normalize])


class HallucinationRiskDataset(Dataset):
    """
    PyTorch Dataset that serves (image_tensor, label) pairs for a given split.

    Filters metadata.csv to rows where metadata['split'] == split.
    Returns (FloatTensor[C, H, W], FloatTensor[1]) where label is
    label_binary (0.0 = safe, 1.0 = risky).
    """

    def __init__(
        self,
        metadata_csv: str,
        split: str,
        rendered_dir: str,
        transform: Optional[Callable] = None,
        augment: bool = False,
    ) -> None:
        """
        Args:
            metadata_csv: Path to data/metadata.csv.
            split: One of 'train', 'val', 'test'.
            rendered_dir: Root directory of rendered PNG files.
            transform: Optional transform applied after loading; receives a PIL
                       Image, returns a Tensor.  When None, a default pipeline
                       (ToTensor + Normalize) is constructed automatically.
            augment: If True, apply training-time augmentations (only meaningful
                     when transform is None and split == 'train').
        """
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be 'train', 'val', or 'test'; got '{split}'")

        df = pd.read_csv(metadata_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.rendered_dir = Path(rendered_dir)
        self.split = split

        # Detect image mode from the first available image, fall back to grayscale
        self._grayscale: Optional[bool] = None

        if transform is not None:
            self.transform = transform
        else:
            # Defer mode detection to first __getitem__ call; for now build with
            # grayscale=True as default (updated lazily on first load)
            self._augment = augment
            self.transform = None  # type: ignore[assignment]

        logger.info(
            "HallucinationRiskDataset split='%s': %d samples", split, len(self.df)
        )

    def _get_transform(self, grayscale: bool) -> Callable:
        """Lazily builds (or rebuilds) the default transform once image mode is known."""
        return _build_transform(grayscale=grayscale, augment=(self._augment and self.split == "train"))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: FloatTensor[C, H, W]  (C=1 grayscale or 3 RGB)
            label: FloatTensor[1]        (0.0 or 1.0)
        """
        row = self.df.iloc[idx]
        label = torch.tensor([float(row["label_binary"])], dtype=torch.float32)

        img_path = self.rendered_dir / row["file_path"]

        if not img_path.exists():
            logger.warning("Image not found: %s — returning zero tensor", img_path)
            # Return zeros with a sensible default shape (1, H, W)
            return torch.zeros(1, 224, 224, dtype=torch.float32), label

        try:
            img = Image.open(str(img_path))
            grayscale = img.mode == "L"

            # Determine transform lazily on first successful load
            if self.transform is None:
                self._grayscale = grayscale
                self.transform = self._get_transform(grayscale)
            elif self._grayscale is None:
                # transform was externally supplied; still record mode for info
                self._grayscale = grayscale

            image_tensor = self.transform(img)
        except Exception as exc:
            logger.warning("Failed to load image %s: %s — returning zero tensor", img_path, exc)
            return torch.zeros(1, 224, 224, dtype=torch.float32), label

        return image_tensor, label
