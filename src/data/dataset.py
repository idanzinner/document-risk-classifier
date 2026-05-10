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


def _build_transform(grayscale: bool, normalize: bool, augment: bool) -> Callable:
    """Builds a torchvision transform pipeline.

    Always outputs 3-channel tensors so pretrained backbones receive the distribution they were trained on.
    If normalize is True, the tensors are normalized to have mean 0 and standard deviation 1.
    """
    pre_ops: list = []
    post_ops: list = []
    if grayscale:
        pre_ops.append(transforms.Grayscale(num_output_channels=3))
    if normalize:
        post_ops.append(transforms.Normalize(
            mean=[0,0,0], # [0.485, 0.456, 0.406], 
            std=[1,1,1])) # [0.229, 0.224, 0.225]))
    else:
        post_ops.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    aug_ops: list = []
    if augment:
        aug_ops = [
            transforms.RandomRotation(degrees=3, fill=255),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.RandomPerspective(distortion_scale=0.05, p=0.3, fill=255),
        ]

    to_tensor = transforms.ToTensor()  # scales [0,255] uint8 → [0,1] float32

    return transforms.Compose(pre_ops + aug_ops + [to_tensor] + post_ops)


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
        normalize: bool = True,
        grayscale: bool = True,
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

        self._file_index = self._build_file_index()

        self._grayscale: Optional[bool] = None
        self._normalize: Optional[bool] = None

        if transform is not None:
            self.transform = transform
        else:
            self._augment = augment
            self.transform = None  # type: ignore[assignment]

        logger.info(
            "HallucinationRiskDataset split='%s': %d samples (%d resolved on disk)",
            split, len(self.df),
            sum(1 for fp in self.df["file_path"] if self._resolve_path(fp) is not None),
        )

    def _build_file_index(self) -> dict[str, Path]:
        """
        Builds a mapping from metadata file_path values to actual PNG paths on
        disk. Metadata stores .pdf extensions; rendered files are .png.
        """
        index: dict[str, Path] = {}
        if not self.rendered_dir.is_dir():
            return index

        for fp in self.df["file_path"]:
            png_name = Path(fp).with_suffix(".png").name
            candidate = self.rendered_dir / png_name
            if candidate.exists():
                index[fp] = candidate

        n_miss = len(self.df) - len(index)
        if n_miss > 0:
            logger.warning(
                "%d/%d files in split '%s' could not be resolved to PNGs on disk",
                n_miss, len(self.df), self.split,
            )
        return index

    def _resolve_path(self, file_path: str) -> Optional[Path]:
        """Returns the resolved on-disk Path for a metadata file_path, or None."""
        return self._file_index.get(file_path)

    def _get_transform(self, grayscale: bool, normalize: bool) -> Callable:
        """Lazily builds (or rebuilds) the default transform once image mode is known."""
        return _build_transform(grayscale=grayscale, normalize=normalize, augment=(self._augment and self.split == "train"))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: FloatTensor[3, H, W]  (always 3-channel, ImageNet-normalised)
            label: FloatTensor[1]        (0.0 or 1.0)
        """
        row = self.df.iloc[idx]
        label = torch.tensor([float(row["label_binary"])], dtype=torch.float32)

        img_path = self._resolve_path(row["file_path"])

        if img_path is None or not img_path.exists():
            return torch.zeros(3, 224, 224, dtype=torch.float32), label

        try:
            img = Image.open(str(img_path))
            grayscale = img.mode == "L"
            normalize = self._normalize

            if self.transform is None:
                self._grayscale = grayscale
                self._normalize = normalize
                self.transform = self._get_transform(grayscale, normalize)
            elif self._grayscale is None:
                self._grayscale = grayscale

            image_tensor = self.transform(img)
        except Exception as exc:
            logger.warning("Failed to load image %s: %s — returning zero tensor", img_path, exc)
            return torch.zeros(3, 224, 224, dtype=torch.float32), label

        return image_tensor, label
