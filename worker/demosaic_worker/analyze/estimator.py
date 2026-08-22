"""Estimates a MosaicProfile from pixels. prd.md §5.4.

Two jobs, and the second is the one that decides whether restoration helps or hurts.

**Block geometry** (§5.4.2). Pixelation makes the gradient field periodic: every ``B`` pixels there
is a block boundary and between them the gradient is zero. Summing the gradient magnitude along each
axis gives a comb, and the comb's period is the block size while its offset is the grid phase.

**Grid anchoring** (§5.4.4). Measured, because getting it wrong is worse than not restoring at all:
`docs/phase0-report.md` §3.2 found object-anchored multi-frame scoring 0.79 to 1.50 dB *below*
single-frame even with perfect alignment. The test is whether the estimated phase stays constant in
frame coordinates (screen-anchored) or in box coordinates (object-anchored) as the box moves. A
track that has not moved cannot distinguish the two and must report UNKNOWN rather than guess.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .profile import DegradationType, GridAnchor, MosaicProfile

#: Block sizes the estimator will consider. prd.md §5.4.2 and the §1.4.2 bands.
MIN_BLOCK = 2
MAX_BLOCK = 48

#: How much stronger the periodic comb must be than the background to call it pixelation.
PIXELATION_CONTRAST = 1.6

#: Frames of movement required before an anchoring verdict is meaningful (§5.4.4).
MIN_FRAMES_FOR_ANCHORING = 4

#: Total box displacement required before the two hypotheses are distinguishable.
MIN_DISPLACEMENT_PX = 3.0


def _gradient_combs(patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column and row profiles of the absolute gradient."""
    gx = np.abs(np.diff(patch, axis=1))
    gy = np.abs(np.diff(patch, axis=0))
    return gx.sum(axis=0), gy.sum(axis=1)


def _best_period(comb: np.ndarray) -> tuple[int, int, float]:
    """Finds the period and offset that best explain a comb.

    Returns ``(period, offset, contrast)`` where contrast is the mean comb value on the predicted
    boundaries over the mean elsewhere. A flat comb — no pixelation — gives a contrast near 1.
    """
    length = len(comb)
    if length < MIN_BLOCK * 2:
        return 0, 0, 1.0

    total = float(comb.mean())
    if total <= 1e-9:
        return 0, 0, 1.0

    best = (0, 0, 1.0)

    for period in range(MIN_BLOCK, min(MAX_BLOCK, length // 2) + 1):
        for offset in range(period):
            positions = np.arange(offset, length, period)
            if len(positions) < 2:
                continue

            on = float(comb[positions].mean())
            mask = np.ones(length, dtype=bool)
            mask[positions] = False
            off = float(comb[mask].mean()) if mask.any() else 0.0

            contrast = on / off if off > 1e-9 else (on / 1e-9 if on > 0 else 1.0)
            if contrast > best[2]:
                best = (period, offset, contrast)

    return best


def estimate_geometry(patch: np.ndarray) -> tuple[MosaicProfile, float]:
    """Estimates block size, phase and degradation type from one region's pixels.

    Returns the profile and the contrast that produced it, so a caller can tell a confident
    pixelation estimate from a shrug.
    """
    if patch.ndim != 2:
        raise ValueError(f"expected a 2-D patch, got {patch.shape}")

    column_comb, row_comb = _gradient_combs(patch.astype(np.float64))

    block_w, offset_x, contrast_x = _best_period(column_comb)
    block_h, offset_y, contrast_y = _best_period(row_comb)

    contrast = min(contrast_x, contrast_y)

    if block_w < MIN_BLOCK or block_h < MIN_BLOCK or contrast < PIXELATION_CONTRAST:
        # No periodic structure: it is smooth, so it is blur rather than a grid.
        return (
            MosaicProfile(
                kind=DegradationType.GAUSSIAN_BLUR,
                block_width=max(block_w, 1),
                block_height=max(block_h, 1),
                anchor=GridAnchor.UNKNOWN,
                confidence=0.2,
            ),
            contrast,
        )

    # Converting a comb offset back to a grid phase, carefully:
    #
    #   a grid with phase p starts at -p, so its boundaries inside the patch are at (B - p) % B;
    #   diff()[i] is patch[i+1] - patch[i], so the comb peaks one pixel *before* each boundary;
    #   therefore offset = (B - p) % B - 1, and p = (B - offset - 1) % B.
    #
    # Getting this backwards produces a phase that is wrong by a block minus itself, which looks
    # plausible for p=0 and fails everywhere else.
    return (
        MosaicProfile(
            kind=DegradationType.PIXELATION,
            block_width=block_w,
            block_height=block_h,
            grid_offset_x=(block_w - offset_x - 1) % block_w,
            grid_offset_y=(block_h - offset_y - 1) % block_h,
            anchor=GridAnchor.UNKNOWN,
            confidence=float(min(1.0, (contrast - 1.0) / 4.0)),
        ),
        contrast,
    )


@dataclass(frozen=True, slots=True)
class AnchorObservation:
    """One frame's evidence about anchoring: where the box was, and the phase measured there."""

    box_origin: tuple[int, int]
    phase: tuple[int, int]


def estimate_anchor(
    observations: list[AnchorObservation],
    block: tuple[int, int],
) -> tuple[GridAnchor, float]:
    """Decides SCREEN, OBJECT or UNKNOWN from a track's phase history. prd.md §5.4.4.

    Returns the verdict and a confidence in ``[0, 1]``.

    **UNKNOWN is a real answer, not a failure.** If the box has not moved, both hypotheses predict a
    constant phase and the evidence cannot separate them. Reporting SCREEN there would let the router
    spend a multi-frame budget on content that has no phase diversity — which measured *worse* than
    single-frame.
    """
    block_w, block_h = block

    if len(observations) < MIN_FRAMES_FOR_ANCHORING:
        return GridAnchor.UNKNOWN, 0.0

    origins = np.array([o.box_origin for o in observations], dtype=np.float64)
    displacement = float(np.abs(origins - origins[0]).max())

    if displacement < MIN_DISPLACEMENT_PX:
        return GridAnchor.UNKNOWN, 0.0

    phases = np.array([o.phase for o in observations], dtype=np.float64)

    # Screen-anchored: the phase is constant in frame coordinates.
    screen_spread = _circular_spread(phases, (block_w, block_h))

    # Object-anchored: the phase moves with the box, so phase + origin is what stays constant.
    object_phases = (phases + origins) % np.array([block_w, block_h], dtype=np.float64)
    object_spread = _circular_spread(object_phases, (block_w, block_h))

    if screen_spread < object_spread:
        winner, loser = GridAnchor.SCREEN, object_spread
        margin = loser - screen_spread
    else:
        winner, loser = GridAnchor.OBJECT, screen_spread
        margin = loser - object_spread

    # Confidence scales with how much better the winning hypothesis explains the data, normalised by
    # the largest spread possible for this block geometry.
    scale = max(block_w, block_h) / 4.0
    confidence = float(np.clip(margin / scale, 0.0, 1.0))

    if confidence < 0.15:
        return GridAnchor.UNKNOWN, confidence

    return winner, confidence


def _circular_spread(phases: np.ndarray, block: tuple[int, int]) -> float:
    """Mean circular deviation of a phase sequence, in pixels.

    Circular because phase wraps: 0 and ``B - 1`` are adjacent, and a plain standard deviation would
    call a perfectly stable grid unstable whenever it happened to sit near the wrap point.
    """
    spreads = []
    for axis, period in enumerate(block):
        if period <= 1:
            spreads.append(0.0)
            continue

        angles = 2.0 * np.pi * phases[:, axis] / period
        resultant = np.hypot(np.cos(angles).mean(), np.sin(angles).mean())
        spreads.append(float(np.sqrt(max(0.0, -2.0 * np.log(max(resultant, 1e-9)))) * period / (2 * np.pi)))

    return float(np.mean(spreads))
