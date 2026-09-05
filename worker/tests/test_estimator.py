"""Profile estimation. prd.md §5.4.2, §5.4.4."""

from __future__ import annotations

import numpy as np
import pytest

from demosaic_worker.analyze.estimator import (
    AnchorObservation,
    estimate_anchor,
    estimate_geometry,
)
from demosaic_worker.analyze.profile import DegradationType, GridAnchor, MosaicProfile
from demosaic_worker.restore.ibp import block_average


def _detailed(height: int = 128, width: int = 128, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:height, 0:width]
    base = 110 + 60 * np.sin(xs / 5.0) * np.cos(ys / 7.0) + 30 * np.sin(xs / 1.9)
    return np.clip(base + rng.normal(0, 6, (height, width)), 0, 255)


# --- geometry ------------------------------------------------------------------------------------


@pytest.mark.parametrize("block", [4, 6, 8, 12, 16])
def test_the_block_size_is_recovered(block: int) -> None:
    """prd.md §5.4.2 — within +/-1 px is the acceptance criterion."""
    spec = MosaicProfile(block_width=block, block_height=block)
    patch = block_average(_detailed(), spec, (0, 0))

    profile, contrast = estimate_geometry(patch)

    assert profile.kind is DegradationType.PIXELATION
    assert abs(profile.block_width - block) <= 1, f"contrast={contrast:.2f}"
    assert abs(profile.block_height - block) <= 1


def test_non_square_blocks_are_recovered() -> None:
    spec = MosaicProfile(block_width=16, block_height=4)
    patch = block_average(_detailed(), spec, (0, 0))

    profile, _ = estimate_geometry(patch)

    assert abs(profile.block_width - 16) <= 1
    assert abs(profile.block_height - 4) <= 1


@pytest.mark.parametrize("phase", [0, 2, 5])
def test_the_grid_phase_is_recovered(phase: int) -> None:
    spec = MosaicProfile(block_width=8, block_height=8)
    patch = block_average(_detailed(), spec, (phase, phase))

    profile, _ = estimate_geometry(patch)

    # Phase wraps, so compare circularly.
    error = min(
        abs(profile.grid_offset_x - phase),
        8 - abs(profile.grid_offset_x - phase),
    )
    assert error <= 1


def test_clean_content_is_not_called_pixelation() -> None:
    """A detector that saw a grid in ordinary texture would restore footage that was fine."""
    profile, contrast = estimate_geometry(_detailed())

    assert profile.kind is not DegradationType.PIXELATION, f"contrast={contrast:.2f}"


def test_blurred_content_is_not_called_pixelation() -> None:
    patch = _detailed()
    for _ in range(6):
        padded = np.pad(patch, 1, mode="reflect")
        patch = (
            padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] + patch
        ) / 5.0

    profile, _ = estimate_geometry(patch)

    assert profile.kind is DegradationType.GAUSSIAN_BLUR


def test_a_non_2d_patch_is_rejected() -> None:
    with pytest.raises(ValueError):
        estimate_geometry(np.zeros((8, 8, 3)))


# --- anchoring (§5.4.4) ---------------------------------------------------------------------------


def test_a_screen_anchored_grid_is_recognised() -> None:
    """Phase constant in frame coordinates while the box moves."""
    observations = [
        AnchorObservation(box_origin=(x, 0), phase=(3, 5)) for x in range(0, 40, 5)
    ]

    anchor, confidence = estimate_anchor(observations, (8, 8))

    assert anchor is GridAnchor.SCREEN
    assert confidence > 0.15


def test_an_object_anchored_grid_is_recognised() -> None:
    """Phase constant relative to the box origin — the case that measured worse than not restoring."""
    observations = [
        AnchorObservation(box_origin=(x, 0), phase=((3 - x) % 8, 5)) for x in range(0, 40, 5)
    ]

    anchor, confidence = estimate_anchor(observations, (8, 8))

    assert anchor is GridAnchor.OBJECT
    assert confidence > 0.15


def test_a_static_track_reports_unknown_rather_than_guessing() -> None:
    """prd.md §5.4.4 — both hypotheses predict a constant phase, so the evidence cannot separate them.

    Reporting SCREEN here would let the router spend a multi-frame budget on content with no phase
    diversity, which measured below single-frame.
    """
    observations = [AnchorObservation(box_origin=(10, 10), phase=(3, 5)) for _ in range(10)]

    anchor, confidence = estimate_anchor(observations, (8, 8))

    assert anchor is GridAnchor.UNKNOWN
    assert confidence == 0.0


def test_too_few_frames_reports_unknown() -> None:
    observations = [AnchorObservation(box_origin=(x, 0), phase=(3, 5)) for x in range(2)]

    anchor, _ = estimate_anchor(observations, (8, 8))

    assert anchor is GridAnchor.UNKNOWN


def test_a_barely_moving_track_reports_unknown() -> None:
    """Enough frames, not enough displacement. Both hypotheses still predict the same phase."""
    observations = [
        AnchorObservation(box_origin=(x, 0), phase=(3, 5)) for x in (0, 1, 2, 2, 1, 0)
    ]

    anchor, _ = estimate_anchor(observations, (8, 8))

    assert anchor is GridAnchor.UNKNOWN, "2 px of movement cannot separate the hypotheses"


def test_noisy_phase_estimates_lower_the_confidence() -> None:
    clean = [AnchorObservation(box_origin=(x, 0), phase=(3, 5)) for x in range(0, 40, 5)]
    rng = np.random.default_rng(1)
    noisy = [
        AnchorObservation(box_origin=(x, 0), phase=(int(rng.integers(0, 8)), int(rng.integers(0, 8))))
        for x in range(0, 40, 5)
    ]

    _, clean_confidence = estimate_anchor(clean, (8, 8))
    _, noisy_confidence = estimate_anchor(noisy, (8, 8))

    assert clean_confidence > noisy_confidence


def _best_period_reference(comb: np.ndarray) -> tuple[int, int, float]:
    """The loop the vectorised search replaced, kept as the oracle. D-43."""
    from demosaic_worker.analyze.estimator import MAX_BLOCK, MIN_BLOCK

    length = len(comb)
    if length < MIN_BLOCK * 2:
        return 0, 0, 1.0
    if float(comb.mean()) <= 1e-9:
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


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("length", [17, 31, 64, 97, 200])
def test_the_vectorised_period_search_matches_the_loop_exactly(seed: int, length: int) -> None:
    """Same period, same offset, same contrast - including which offset wins a tie."""
    from demosaic_worker.analyze.estimator import _best_period

    rng = np.random.default_rng(seed)
    comb = rng.random(length) * 10.0
    # Plant a real grid on half the seeds so the search has something to find.
    if seed % 2:
        period = int(rng.integers(4, 17))
        comb[int(rng.integers(0, period))::period] += 40.0

    reference = _best_period_reference(comb)
    fast = _best_period(comb)

    assert fast[:2] == reference[:2]
    assert fast[2] == pytest.approx(reference[2], rel=1e-9)


def test_the_vectorised_period_search_on_a_flat_comb() -> None:
    from demosaic_worker.analyze.estimator import _best_period

    assert _best_period(np.zeros(64)) == (0, 0, 1.0)
    assert _best_period(np.ones(5)) == (0, 0, 1.0)
