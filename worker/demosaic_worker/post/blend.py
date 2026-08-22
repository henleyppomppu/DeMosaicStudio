"""Mask-aware compositing. prd.md §5.11.

The sequence the PRD specifies, and the reason for each step:

    segmentation mask -> controlled dilation -> edge-aware feathering
                      -> temporal alpha smoothing -> compositing

**Dilation** covers the mosaic's influence past the mask edge. A pixelated region's boundary block
is partly clean and partly not, so the detector's mask stops short of where the damage actually
ends; ``2 + ceil(block / 4)`` pixels is the allowance.

**Edge-aware feathering** follows the source's own gradients instead of a fixed Gaussian, so the
transition hides along real edges rather than cutting across them.

**Temporal alpha smoothing** stops the mask edge breathing. Without it a mask that jitters by a
pixel per frame produces a boundary that shimmers — the flicker of §5.10, arriving through the
blender rather than the model.

Compositing happens in linear light. Blending two gamma-encoded values produces a result darker than
either, which shows up as a dark seam ringing every restored region.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Base dilation, before the block-size allowance. prd.md §5.11.
BASE_DILATION_PX = 2

#: Feather width, default and range. prd.md §5.11.
DEFAULT_FEATHER_PX = 3
MIN_FEATHER_PX = 1
MAX_FEATHER_PX = 9

#: How fast the alpha map may change between frames. Lower is steadier and lags more.
DEFAULT_ALPHA_SMOOTHING = 0.4

#: sRGB-ish transfer. Not the exact piecewise curve: the difference is far below what blending needs
#: and this keeps the round trip exactly invertible, which matters more here.
GAMMA = 2.2


def dilation_for(block_size: int) -> int:
    """Dilation in pixels for a given mosaic block size."""
    return BASE_DILATION_PX + -(-block_size // 4)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation by a square structuring element."""
    if radius <= 0:
        return mask.copy()

    out = mask.copy()
    for _ in range(radius):
        padded = np.pad(out, 1, constant_values=False)
        grown = np.zeros_like(out)
        for dy in range(3):
            for dx in range(3):
                grown |= padded[dy : dy + out.shape[0], dx : dx + out.shape[1]]
        out = grown

    return out


def _box_blur(field: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur via cumulative sums."""
    if radius <= 0:
        return field.copy()

    out = field.astype(np.float64)
    for axis in (0, 1):
        padded = np.pad(out, [(radius, radius) if a == axis else (0, 0) for a in (0, 1)], mode="edge")
        cumulative = np.cumsum(padded, axis=axis)
        cumulative = np.concatenate(
            [np.zeros_like(np.take(cumulative, [0], axis=axis)), cumulative], axis=axis
        )
        upper = np.take(cumulative, range(2 * radius + 1, cumulative.shape[axis]), axis=axis)
        lower = np.take(cumulative, range(0, cumulative.shape[axis] - 2 * radius - 1), axis=axis)
        out = (upper - lower) / (2 * radius + 1)

    return out


def feather(mask: np.ndarray, source: np.ndarray, width: int = DEFAULT_FEATHER_PX) -> np.ndarray:
    """Turns a binary mask into an alpha map whose transition follows the source's edges.

    A plain blur would spread the alpha equally in every direction. Weighting the blur by local
    gradient magnitude makes the transition tighter where the picture has an edge to hide it in and
    softer across flat areas, which is where a visible seam would otherwise show.
    """
    width = int(np.clip(width, MIN_FEATHER_PX, MAX_FEATHER_PX))

    alpha = _box_blur(mask.astype(np.float64), width)

    gradient = np.abs(np.gradient(source.astype(np.float64))[0]) + np.abs(
        np.gradient(source.astype(np.float64))[1]
    )
    peak = gradient.max()
    if peak > 1e-9:
        # Where the source has a strong edge, snap the alpha back towards the hard mask.
        edge_weight = np.clip(gradient / peak, 0.0, 1.0)
        alpha = alpha * (1.0 - edge_weight) + mask.astype(np.float64) * edge_weight

    return np.clip(alpha, 0.0, 1.0)


def to_linear(image: np.ndarray) -> np.ndarray:
    """Gamma-encoded 0-255 to linear 0-1."""
    return np.power(np.clip(image, 0.0, 255.0) / 255.0, GAMMA)


def from_linear(image: np.ndarray) -> np.ndarray:
    """Linear 0-1 back to gamma-encoded 0-255."""
    return np.clip(np.power(np.clip(image, 0.0, 1.0), 1.0 / GAMMA) * 255.0, 0.0, 255.0)


@dataclass
class TemporalAlpha:
    """Smooths each track's alpha map across frames. prd.md §5.11.

    Per track rather than per frame: two regions in one frame have independent boundaries, and
    averaging them together would smear one into the other.
    """

    smoothing: float = DEFAULT_ALPHA_SMOOTHING
    _previous: dict[int, np.ndarray] = field(default_factory=dict)

    def smooth(self, track_id: int, alpha: np.ndarray) -> np.ndarray:
        """Returns the temporally smoothed alpha for one track."""
        previous = self._previous.get(track_id)

        if previous is None or previous.shape != alpha.shape:
            smoothed = alpha
        else:
            smoothed = (1.0 - self.smoothing) * previous + self.smoothing * alpha

        self._previous[track_id] = smoothed
        return smoothed

    def forget(self, track_id: int) -> None:
        """Drops a terminated track's state."""
        self._previous.pop(track_id, None)


def composite(
    original: np.ndarray,
    restored: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Blends ``restored`` over ``original`` using ``alpha``, in linear light.

    Returns gamma-encoded 0-255, matching the input convention.
    """
    if not (original.shape == restored.shape == alpha.shape):
        raise ValueError(
            f"shape mismatch: original {original.shape}, restored {restored.shape}, alpha {alpha.shape}"
        )

    blended = to_linear(original) * (1.0 - alpha) + to_linear(restored) * alpha
    return from_linear(blended)


def blend_region(
    frame: np.ndarray,
    restored: np.ndarray,
    mask: np.ndarray,
    *,
    block_size: int,
    feather_px: int = DEFAULT_FEATHER_PX,
    temporal: TemporalAlpha | None = None,
    track_id: int | None = None,
) -> np.ndarray:
    """Runs the full §5.11 sequence for one region and returns the composited frame."""
    grown = dilate(mask, dilation_for(block_size))
    alpha = feather(grown, frame, feather_px)

    if temporal is not None and track_id is not None:
        alpha = temporal.smooth(track_id, alpha)

    return composite(frame, restored, alpha)
