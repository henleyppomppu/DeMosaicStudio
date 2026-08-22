"""Mosaic segmentation network. prd.md §5.2.2, D-03.

A small U-Net with a full-resolution decoder. Two properties are deliberate:

**Full-resolution output.** §5.11 blends on the mask, so boundary precision is the product
requirement, not IoU alone. This is the reason D-03 rejects prototype-coefficient masks even now
that their licence is available: they are coarser exactly where blending is most sensitive.

**Small.** Mosaic detection is a texture and frequency problem, not a semantic one — the network has
to notice that a region's gradient field is periodic, which is a far shallower question than
recognising an object. Starting large would hide whether the signal is learnable at all, which is
the only thing this first experiment is trying to find out.

No pretrained encoder here. `timm` arrives when the question moves from "is the signal learnable" to
"how well", because an ImageNet initialisation would confound the first answer.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with GroupNorm.

    GroupNorm rather than BatchNorm: batches here are small and, more importantly, a batch mixes
    heavily degraded and clean crops, so batch statistics are not a meaningful population.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the block."""
        return self.body(x)


class MosaicUNet(nn.Module):
    """Binary mosaic segmentation. Input ``(N, 1, H, W)``, output ``(N, 1, H, W)`` logits."""

    def __init__(self, width: int = 32, depth: int = 4) -> None:
        super().__init__()

        channels = [width * (2**i) for i in range(depth)]

        self.downs = nn.ModuleList()
        in_channels = 1
        for out_channels in channels:
            self.downs.append(ConvBlock(in_channels, out_channels))
            in_channels = out_channels

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2)

        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        in_channels = channels[-1] * 2
        for out_channels in reversed(channels):
            self.ups.append(nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2))
            self.up_convs.append(ConvBlock(out_channels * 2, out_channels))
            in_channels = out_channels

        self.head = nn.Conv2d(channels[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns per-pixel logits at the input resolution."""
        skips: list[torch.Tensor] = []

        for block in self.downs:
            x = block(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips), strict=True):
            x = up(x)
            x = conv(torch.cat([x, skip], dim=1))

        return self.head(x)

    @property
    def parameter_count(self) -> int:
        """Trainable parameters, for the record in the experiment report."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """BCE plus soft Dice.

    BCE alone is dominated by the background on crops where the mosaic covers a small fraction, and
    a detector that predicts "nothing" everywhere would score well on it. Dice supplies the gradient
    that actually cares about the region.
    """
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)

    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    union = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - (2.0 * intersection + eps) / (union + eps)

    return bce + dice.mean()


@torch.no_grad()
def mask_iou(logits: torch.Tensor, target: torch.Tensor, *, threshold: float = 0.5) -> torch.Tensor:
    """Per-sample IoU. Samples with no mosaic and no prediction score 1.0."""
    predicted = (torch.sigmoid(logits) > threshold).float()

    intersection = (predicted * target).sum(dim=(1, 2, 3))
    union = ((predicted + target) > 0).float().sum(dim=(1, 2, 3))

    return torch.where(union > 0, intersection / union, torch.ones_like(union))


@torch.no_grad()
def false_positive_area(logits: torch.Tensor, target: torch.Tensor, *, threshold: float = 0.5) -> torch.Tensor:
    """Fraction of a clean crop wrongly marked. prd.md §5.2.5a, in miniature.

    Reported separately from IoU because IoU on an empty target is uninformative, and firing on
    clean footage is the failure that actually damages a user's video.
    """
    predicted = (torch.sigmoid(logits) > threshold).float()
    wrong = (predicted * (1.0 - target)).sum(dim=(1, 2, 3))
    pixels = target[0].numel()

    return wrong / pixels
