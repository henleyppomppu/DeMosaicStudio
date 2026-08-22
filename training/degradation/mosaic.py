"""Synthetic mosaic degradation. prd.md §11.3.

This module is the forward operator the restoration model learns to invert, and it is also what the
Phase 0 feasibility gate (§1.4.3) measures against. Two properties matter more than the rest:

**Grid anchoring** (§1.4.1) is a first-class parameter, not an afterthought. Pixelation replaces
every ``B x B`` block with its mean, which destroys everything finer than ``B`` *in that frame*.
Multi-frame restoration recovers real detail only when different frames sample the hidden signal at
different grid phases:

* ``SCREEN`` — the grid is fixed to frame coordinates, so a moving subject slides across a
  stationary grid and each frame averages a different set of subject pixels. Phase diversity is
  high and multi-frame pays.
* ``OBJECT`` — the grid moves with the subject, so the same pixels land in the same block every
  frame. Averaging reduces noise and nothing else. Multi-frame buys nothing.

A generator that only produced screen-anchored mosaics would make the gate look better than reality.

**Determinism.** Every sample is reproducible from its seed, so an evaluation set can be rebuilt and
a regression can be re-run on exactly the frames that produced it (AC-11.3).

The *type* it describes — :class:`~demosaic_worker.analyze.profile.MosaicProfile` — lives in the
worker, because the worker has to reason about the same parameters at restoration time. This module
only generates. Dependencies point from `training/` into `demosaic_worker`, never the other way
(AGENTS.md layer rules).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from demosaic_worker.analyze.profile import (
    RECOVERABILITY_BANDS,
    DegradationType,
    GridAnchor,
    MosaicProfile,
    band_for,
)

__all__ = [
    "RECOVERABILITY_BANDS",
    "DegradationType",
    "GridAnchor",
    "MosaicProfile",
    "band_for",
    "phase_diversity",
    "pixelate",
    "with_random_geometry",
]


def pixelate(
    image: np.ndarray,
    spec: MosaicProfile,
    *,
    origin_x: int = 0,
    origin_y: int = 0,
) -> np.ndarray:
    """Box-averages ``image`` on the mosaic grid.

    ``image`` is ``(H, W)`` or ``(H, W, C)``. The phase comes from :meth:`MosaicProfile.phase_for`, so
    the same call produces a screen-anchored or object-anchored result depending only on the spec.

    Partial blocks at the region edges are averaged over the pixels they actually contain, which is
    what a real tool does and what makes edge blocks carry slightly more information than interior
    ones.
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"expected a 2-D or 3-D array, got shape {image.shape}")

    phase_x, phase_y = spec.phase_for(origin_x, origin_y)
    height, width = image.shape[:2]
    out = image.copy()

    # Start one block before the origin so a non-zero phase produces a genuine partial first block
    # rather than silently shifting the whole grid.
    y = -phase_y
    while y < height:
        y0, y1 = max(y, 0), min(y + spec.block_height, height)
        x = -phase_x
        while x < width:
            x0, x1 = max(x, 0), min(x + spec.block_width, width)
            if y1 > y0 and x1 > x0:
                block = image[y0:y1, x0:x1]
                out[y0:y1, x0:x1] = block.mean(axis=(0, 1), keepdims=True).astype(image.dtype)
            x += spec.block_width
        y += spec.block_height

    if spec.opacity < 1.0:
        out = (spec.opacity * out.astype(np.float32)
               + (1.0 - spec.opacity) * image.astype(np.float32)).astype(image.dtype)

    return out


def phase_diversity(spec: MosaicProfile, origins: list[tuple[int, int]]) -> float:
    """Fraction of distinct grid phases observed across a sequence of box origins.

    This is the quantity §1.4.1 says multi-frame restoration actually runs on, and the quantity
    §5.9.4 feeds into restoration confidence. It is ``~0`` for an object-anchored grid regardless of
    how much the subject moves, which is precisely the case the router must not spend a multi-frame
    budget on.

    Returns a value in ``[0, 1]``: observed distinct phases over the maximum possible for this
    block geometry and sample count.
    """
    if not origins:
        return 0.0

    phases = {spec.phase_for(x, y) for x, y in origins}
    possible = min(len(origins), spec.block_width * spec.block_height)

    return len(phases) / possible


def with_random_geometry(spec: MosaicProfile, rng: np.random.Generator) -> MosaicProfile:
    """Randomizes block geometry and phase, keeping the anchoring and type fixed.

    Non-square blocks are produced deliberately (§11.3): a square-only generator is a domain gap.
    """
    block_width = int(rng.integers(2, 33))
    block_height = block_width if rng.random() < 0.6 else int(rng.integers(2, 33))

    return replace(
        spec,
        block_width=block_width,
        block_height=block_height,
        grid_offset_x=int(rng.integers(0, block_width)),
        grid_offset_y=int(rng.integers(0, block_height)),
    )
