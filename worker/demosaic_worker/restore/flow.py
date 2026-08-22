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


@dataclass(frozen=True, slots=True)
class Alignment:
    """One neighbour's alignment to the target frame."""

    #: Flow from the target frame to the neighbour, ``(H, W, 2)`` in pixels.
    target_to_neighbour: np.ndarray

    #: Flow from the neighbour back to the target.
    neighbour_to_target: np.ndarray

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


def _pad_to_multiple(tensor: torch.Tensor, multiple: int = 8) -> tuple[torch.Tensor, tuple[int, int]]:
    height, width = tensor.shape[-2:]
    pad_h = (-height) % multiple
    pad_w = (-width) % multiple

    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")

    return tensor, (height, width)


class DenseAligner:
    """Estimates dense flow between frames and warps images along it.

    One instance per process: the network is loaded once and reused. Not thread-safe.
    """

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(self.device).eval()

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def _flow(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        first, size = _pad_to_multiple(_to_rgb_tensor(a, self.device))
        second, _ = _pad_to_multiple(_to_rgb_tensor(b, self.device))

        # RAFT returns a list of refinement iterations; the last is the estimate.
        flow = self.model(first, second)[-1]

        height, width = size
        return flow[0, :, :height, :width].permute(1, 2, 0).cpu().numpy()

    def align(
        self,
        target: np.ndarray,
        neighbour: np.ndarray,
        *,
        consistency_px: float = DEFAULT_CONSISTENCY_PX,
    ) -> Alignment:
        """Estimates flow both ways and derives per-pixel confidence.

        Both directions are needed anyway — one to simulate the neighbour from the target, one to
        bring a residual back — so forward-backward consistency costs nothing extra beyond the
        second RAFT pass, and it is the only per-pixel confidence available without a learned
        uncertainty head.
        """
        forward = self._flow(target, neighbour)
        backward = self._flow(neighbour, target)

        # Sample the backward flow at where the forward flow says each pixel went. If the two agree,
        # the round trip returns to the origin.
        round_trip = warp_by_flow(backward, forward)
        disagreement = np.linalg.norm(forward + round_trip, axis=-1)

        confidence = np.clip(1.0 - disagreement / max(consistency_px, 1e-6), 0.0, 1.0)

        return Alignment(forward, backward, confidence.astype(np.float32))


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
