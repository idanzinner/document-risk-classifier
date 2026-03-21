"""
dit_classifier.py — Document Image Transformer (DiT) classifier.

Wraps microsoft/dit-base (HuggingFace Transformers) with a single-logit
head for binary hallucination-risk classification.  Exposes three-stage
fine-tuning helpers: freeze_backbone / unfreeze_top_blocks / unfreeze_all.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class DiTClassifier(nn.Module):
    """
    DiT (Document Image Transformer) with a single-logit output for binary
    hallucination-risk classification of Hebrew PDF page images.

    Input:  FloatTensor[B, C, H, W]  (C=1 or 3, H=W=224)
    Output: logits FloatTensor[B, 1]

    DiT is built on BeitModel; the backbone returns a pooler_output of shape
    [B, hidden_size] which is fed into a linear classification head.

    Staged training protocol:
      Stage 1 — freeze backbone, train head only         (freeze_backbone)
      Stage 2 — unfreeze top N transformer blocks        (unfreeze_top_blocks)
      Stage 3 — unfreeze entire backbone                 (unfreeze_all)
    """

    num_classes: int = 1

    def __init__(
        self,
        model_name: str = "microsoft/dit-base",
        num_classes: int = 1,
    ) -> None:
        """
        Args:
            model_name: HuggingFace model identifier for a DiT / BEiT model.
            num_classes: Output logit count (always 1 for binary BCE training).
        """
        super().__init__()
        self.num_classes = num_classes

        self.backbone = AutoModel.from_pretrained(model_name)

        # Determine hidden size from the model config
        hidden_size: int = self.backbone.config.hidden_size
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: FloatTensor[B, C, H, W] — C may be 1 (grayscale) or 3 (RGB).

        Returns:
            logits: FloatTensor[B, 1]
        """
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        outputs = self.backbone(pixel_values=x)

        # Prefer pooler_output (CLS-token after a dense + tanh), fall back to
        # the raw CLS token from last_hidden_state if pooler is absent.
        if outputs.pooler_output is not None:
            pooled = outputs.pooler_output          # [B, hidden_size]
        else:
            pooled = outputs.last_hidden_state[:, 0, :]  # [B, hidden_size]

        logits = self.head(pooled)  # [B, num_classes]
        return logits

    def freeze_backbone(self) -> None:
        """Freezes all backbone parameters; only the classification head trains."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = True

    def unfreeze_top_blocks(self, n_blocks: int = 2) -> None:
        """
        Unfreezes the last n_blocks transformer encoder blocks of the backbone.

        For BEiT/DiT models the encoder layers live at
        ``self.backbone.encoder.layer``.  All other backbone weights remain
        frozen (call ``freeze_backbone`` first, then this method).

        Args:
            n_blocks: Number of trailing encoder blocks to unfreeze.
        """
        encoder_layers = self.backbone.encoder.layer
        for layer in encoder_layers[-n_blocks:]:
            for param in layer.parameters():
                param.requires_grad = True

    def unfreeze_all(self) -> None:
        """Unfreezes all model parameters (backbone + head)."""
        for param in self.parameters():
            param.requires_grad = True
