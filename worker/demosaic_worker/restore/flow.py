"""Dense optical-flow alignment with per-pixel confidence. prd.md §5.7, §5.9.4.

The Phase 0 gate measured +3.30 dB available with perfect alignment and −0.86 dB delivered by a
global translation, and named the gap Phase 2's critical path. This module is the first attempt to
close it.

Two things here are requirements rather than conveniences.

**Dense, not global.** Real shots contain parallax, independent object motion and occlusion. A single
translation per frame cannot describe them, and back-projecting through a wrong warp injects error at
exactly the spatial frequencies multi-frame restoration is trying to recover.

**Per-pixel confidence, not per-frame.** The gate found a per-frame photometric ratio admitting 86% of
neighbours in the medium-motion band while the result was still −1.76 dB: a frame whose *global*
alignment is plausible can be locally wrong everywhere that matters. Forward-backward flow
consistency gives a confidence value for every pixel, so a neighbour can contribute where it is
trustworthy and be ignored where it is not — which is what §5.7's "exclude rather than down-weight"
means once it is applied at the right granularity.

RAFT-small is used as the flow estimator: 990 K parameters, BSD-3, shipped with torchvision, trained
on FlyingChairs/FlyingThings3D. It is a stand-in for measurement purposes, not a shipping decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

#: Forward-backward disagreement, in pixels, above which a pixel is not trusted. §5.7.
DEFAULT_CONSISTENCY_PX = 1.5

#: Fraction of full resolution the flow is estimated at. **Measured**, and it is not a speed knob:
#: at these crop sizes RAFT's cost is dominated by fixed overhead, so the timing barely moves.
#: It is a *quality* knob, and a large one.
#:
#: ========  ======  ======  ======  ======
#: clip      1.00    0.75    0.50    0.35
#: ========  ======  ======  ======  ======
#: screen    +2.81   +5.39   +5.07   +4.84
#: pan1      +3.93   +4.71   +4.76   +4.74
#: fast16    +5.35   +5.91   +5.65   +5.83
#: ========  ======  ======  ======  ======
#:
#: Full resolution is worst on all three. Two mechanisms fit: the mosaic is a screen-fixed
#: high-frequency texture that competes with the content for RAFT's attention, and downscaling
#: shrinks the displacement into the range the small model handles best. The pan1 row - 0.79 px of
#: motion, and still better downscaled - favours the first. Three clips cannot separate them.
DEFAULT_FLOW_SCALE = 0.75

#: Photometric residual, in levels, below which a warped pixel counts as landing where it should.
#: Used when the backward pass is skipped. §5.7 rejects a *per-frame* photometric ratio as a
#: confidence signal, and this is not that: it stands in for the scalar "is this alignment usable
#: at all", never for the per-pixel weighting.
PHOTOMETRIC_TOLERANCE = 12.0

#: RAFT downsamples by 8 and then builds a correlation pyramid over that, so anything smaller than
#: this on either axis raises "feature maps are too small to be down-sampled".
#:
#: **Measured, not read off a spec:** 96x96 fails and 128x128 works. Small ROIs are the common case
#: for this pipeline, so the padding below is not an edge case — without it every alignment on a
#: small region fails and the router silently falls back to single-frame everywhere.
MIN_FLOW_SIZE = 128


@dataclass(frozen=True, slots=True)
class Alignment:
    """One neighbour's alignment to the target frame."""

    #: Flow from the target frame to the neighbour, ``(H, W, 2)`` in pixels.
    target_to_neighbour: np.ndarray

    #: Flow from the neighbour back to the target, or ``None`` when it was not estimated.
    #:
    #: It exists for :func:`reconstruct_flow`, which warps residuals back along it. The accumulator
    #: does not need it - it warps corrections nowhere - and it costs a second RAFT pass, which is
    #: half the alignment time. So it is optional, and absent rather than approximated: a plausible
    #: stand-in would be used without anyone noticing it was not the real thing.
    neighbour_to_target: np.ndarray | None

    #: Per-pixel confidence in ``[0, 1]``, from forward-backward consistency.
    confidence: np.ndarray

    @property
    def usable_fraction(self) -> float:
        """Fraction of pixels whose flow is self-consistent."""
        return float((self.confidence > 0.5).mean())


def _to_rgb_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    """RAFT wants 3-channel float in [-1, 1] with both sides a multiple of 8."""
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
    tensor = tensor[None, None].repeat(1, 3, 1, 1).to(device)
    return tensor * 2.0 - 1.0


def _pad_for_raft(tensor: torch.Tensor, multiple: int = 8) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pads up to RAFT's minimum size and to a multiple of its stride.

    Replicate rather than reflect: the padding is scaffolding the flow is computed on and then
    discarded, and replicating the edge produces zero apparent motion there, which is the least
    misleading thing to hand a flow estimator.
    """
    height, width = tensor.shape[-2:]

    target_h = max(height, MIN_FLOW_SIZE)
    target_w = max(width, MIN_FLOW_SIZE)
    target_h += (-target_h) % multiple
    target_w += (-target_w) % multiple

    pad_h = target_h - height
    pad_w = target_w - width

    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")

    return tensor, (height, width)


class DenseAligner:
    """Estimates dense flow between frames and warps images along it.

    One instance per process: the network is loaded once and reused. Not thread-safe.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        *,
        flow_scale: float = DEFAULT_FLOW_SCALE,
    ) -> None:
        if not 0.0 < flow_scale <= 1.0:
            raise ValueError(f"flow_scale must be in (0, 1], got {flow_scale}")

        self.flow_scale = flow_scale
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(self.device).eval()

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _effective_scale(self, shape: tuple[int, ...]) -> float:
        """Downscales only where there is resolution to spare.

        A crop already smaller than :data:`MIN_FLOW_SIZE` is padded up to it, so shrinking it first
        just replaces content with padding. Measured: a 24 px region at scale 0.75 drops from a
        usable fraction of 0.9 to 0.31 - the alignment stops working entirely, on exactly the small
        ROIs that are this pipeline's common case.
        """
        shortest = min(shape[:2])
        if shortest <= MIN_FLOW_SIZE:
            return 1.0

        return min(1.0, max(self.flow_scale, MIN_FLOW_SIZE / shortest))

    @torch.no_grad()
    def _flow(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Estimates flow from ``a`` to ``b``, at :data:`DEFAULT_FLOW_SCALE` of full resolution."""
        shape = a.shape
        scale = self._effective_scale(shape)

        first = _to_rgb_tensor(a, self.device)
        second = _to_rgb_tensor(b, self.device)

        if scale != 1.0:
            first = F.interpolate(first, scale_factor=scale, mode="bilinear", align_corners=False)
            second = F.interpolate(second, scale_factor=scale, mode="bilinear", align_corners=False)

        first, size = _pad_for_raft(first)
        second, _ = _pad_for_raft(second)

        # RAFT returns a list of refinement iterations; the last is the estimate.
        flow = self.model(first, second)[-1]

        height, width = size
        flow = flow[:, :, :height, :width]

        if scale != 1.0:
            # Back to full resolution, and the vectors scale with the geometry they describe.
            flow = F.interpolate(flow, size=shape, mode="bilinear", align_corners=False) / scale

        return flow[0].permute(1, 2, 0).cpu().numpy()

    def align(
        self,
        target: np.ndarray,
        neighbour: np.ndarray,
        *,
        consistency_px: float = DEFAULT_CONSISTENCY_PX,
        backward: bool = True,
    ) -> Alignment:
        """Estimates the flow and a per-pixel confidence for it.

        With ``backward`` (the default) the flow is estimated both ways and the confidence is
        forward-backward consistency: sample the backward flow where the forward flow says each
        pixel went, and see whether the round trip returns to the origin. That is the only per-pixel
        confidence available without a learned uncertainty head, and §5.7 asks for per-pixel.

        Without it, the confidence comes from a **photometric** residual - warp the neighbour onto
        the target and see where it lands. §5.7 rejects a photometric *ratio per frame* as a
        confidence signal, and this is not that: it is per pixel, and it stands in only for the
        scalar the accumulator actually consumes. Measured, the two give the same result
        (+2.81 dB either way) for half the time (104 ms against 49), because the second RAFT pass
        is the expensive half.
        """
        forward = self._flow(target, neighbour)

        if backward:
            reverse = self._flow(neighbour, target)
            round_trip = warp_by_flow(reverse, forward)
            disagreement = np.linalg.norm(forward + round_trip, axis=-1)
            confidence = np.clip(1.0 - disagreement / max(consistency_px, 1e-6), 0.0, 1.0)
            return Alignment(forward, reverse, confidence.astype(np.float32))

        landed = warp_by_flow(neighbour.astype(np.float64), forward)
        residual = np.abs(landed - target.astype(np.float64))
        confidence = np.clip(1.0 - residual / max(PHOTOMETRIC_TOLERANCE, 1e-6), 0.0, 1.0)

        return Alignment(forward, None, confidence.astype(np.float32))


def warp_by_flow(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Backward-warps ``image`` by ``flow`` with bilinear sampling.

    ``out[p] = image[p + flow[p]]``. Works on ``(H, W)`` and ``(H, W, C)`` alike, which is why the
    flow field itself can be warped by another flow — that is how the consistency check above works.
    """
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError(f"flow must be (H, W, 2), got {flow.shape}")
    if image.shape[:2] != flow.shape[:2]:
        raise ValueError(f"shape mismatch: image {image.shape[:2]} vs flow {flow.shape[:2]}")

    height, width = image.shape[:2]
    ys, xs = np.mgrid[0:height, 0:width]

    src_x = np.clip(xs + flow[..., 0], 0, width - 1)
    src_y = np.clip(ys + flow[..., 1], 0, height - 1)

    x0 = np.floor(src_x).astype(np.int64)
    y0 = np.floor(src_y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)

    wx = (src_x - x0)[..., None] if image.ndim == 3 else src_x - x0
    wy = (src_y - y0)[..., None] if image.ndim == 3 else src_y - y0

    top = image[y0, x0] * (1 - wx) + image[y0, x1] * wx
    bottom = image[y1, x0] * (1 - wx) + image[y1, x1] * wx

    return top * (1 - wy) + bottom * wy
