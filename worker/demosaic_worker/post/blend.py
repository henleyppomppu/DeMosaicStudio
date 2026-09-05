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
    """Binary dilation by a square structuring element, ``radius`` times."""
    if radius <= 0:
        return mask.copy()

    return _dilate_slices(mask, radius)


def _dilate_slices(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilation by repeated 3x3 passes; the specification the GPU pooling form must match."""
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

    # One gradient call: it returns both axes, and calling it twice computed both twice.
    gy, gx = np.gradient(source.astype(np.float64))
    gradient = np.abs(gy) + np.abs(gx)
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
    """Runs the full §5.11 sequence for one region and returns the composited frame.

    On a CUDA machine the dilation, feather and composite run as a handful of pooling and
    element-wise ops on the GPU (:func:`_blend_region_torch`); the numpy functions above are the
    specification and the fallback, and ``test_blend.py`` holds the two to each other. Profiled
    before this at 121 ms a frame - dilation, a box blur, two gradients and three ``np.power``
    calls over a region a good fraction of the frame, for every track, in numpy.
    """
    fast = _blend_region_torch(frame, restored, mask, block_size=block_size,
                               feather_px=feather_px, temporal=temporal, track_id=track_id)
    if fast is not None:
        return fast

    grown = dilate(mask, dilation_for(block_size))
    alpha = feather(grown, frame, feather_px)

    if temporal is not None and track_id is not None:
        alpha = temporal.smooth(track_id, alpha)

    return composite(frame, restored, alpha)


def _blend_region_torch(
    frame: np.ndarray,
    restored: np.ndarray,
    mask: np.ndarray,
    *,
    block_size: int,
    feather_px: int,
    temporal: TemporalAlpha | None,
    track_id: int | None,
) -> np.ndarray | None:
    """The same sequence on the GPU. ``None`` when there is no CUDA device, so the caller falls back.

    Each numpy step has an exact pooling equivalent:

    * square dilation by ``r`` = ``max_pool2d`` with kernel ``2r+1``, once - the iterated 3x3 form
      and the single large kernel produce the same set (a square is a square);
    * box blur with edge padding = replicate-pad then ``avg_pool2d``;
    * ``np.gradient`` = central differences inside, one-sided at the ends, written out;
    * the linear-light composite is the same arithmetic in float32.

    Temporal alpha smoothing stays in numpy: it is one small array per track.
    """
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None

    device = torch.device("cuda")
    f = torch.as_tensor(np.ascontiguousarray(frame, dtype=np.float32), device=device)[None, None]
    r = torch.as_tensor(np.ascontiguousarray(restored, dtype=np.float32), device=device)[None, None]
    m = torch.as_tensor(np.ascontiguousarray(mask, dtype=np.float32), device=device)[None, None]

    # dilate
    radius = dilation_for(block_size)
    grown = functional.max_pool2d(m, kernel_size=2 * radius + 1, stride=1, padding=radius) if radius > 0 else m

    # feather: box blur (edge-padded) then edge-aware snap
    width = int(np.clip(feather_px, MIN_FEATHER_PX, MAX_FEATHER_PX))
    padded = functional.pad(grown, (width, width, width, width), mode="replicate")
    alpha = functional.avg_pool2d(padded, kernel_size=2 * width + 1, stride=1)

    gy = torch.empty_like(f)
    gx = torch.empty_like(f)
    gy[..., 1:-1, :] = (f[..., 2:, :] - f[..., :-2, :]) * 0.5
    gy[..., 0, :] = f[..., 1, :] - f[..., 0, :]
    gy[..., -1, :] = f[..., -1, :] - f[..., -2, :]
    gx[..., :, 1:-1] = (f[..., :, 2:] - f[..., :, :-2]) * 0.5
    gx[..., :, 0] = f[..., :, 1] - f[..., :, 0]
    gx[..., :, -1] = f[..., :, -1] - f[..., :, -2]
    gradient = gy.abs() + gx.abs()
    peak = float(gradient.max())
    if peak > 1e-9:
        edge_weight = (gradient / peak).clamp_(0.0, 1.0)
        alpha = alpha * (1.0 - edge_weight) + grown * edge_weight
    alpha = alpha.clamp_(0.0, 1.0)

    if temporal is not None and track_id is not None:
        alpha = torch.as_tensor(
            temporal.smooth(track_id, alpha[0, 0].cpu().numpy().astype(np.float64)),
            device=device, dtype=torch.float32,
        )[None, None]

    # composite in linear light
    lin_f = (f.clamp(0.0, 255.0) / 255.0).pow(GAMMA)
    lin_r = (r.clamp(0.0, 255.0) / 255.0).pow(GAMMA)
    blended = lin_f * (1.0 - alpha) + lin_r * alpha
    out = (blended.clamp(0.0, 1.0).pow(1.0 / GAMMA) * 255.0).clamp_(0.0, 255.0)

    return out[0, 0].cpu().numpy().astype(np.float64)
