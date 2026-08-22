"""Mask-aware compositing. prd.md §5.11."""

from __future__ import annotations

import numpy as np
import pytest

from demosaic_worker.post.blend import (
    TemporalAlpha,
    blend_region,
    composite,
    dilate,
    dilation_for,
    feather,
    from_linear,
    to_linear,
)


def _frame(height: int = 64, width: int = 64, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:height, 0:width]
    return np.clip(120 + 50 * np.sin(xs / 6.0) + rng.normal(0, 5, (height, width)), 0, 255)


def _square_mask(height: int = 64, width: int = 64, size: int = 20) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    top = left = (height - size) // 2
    mask[top : top + size, left : left + size] = True
    return mask


# --- dilation -------------------------------------------------------------------------------------


def test_dilation_grows_with_block_size() -> None:
    """A pixelated region's damage extends to its enclosing block, past where the mask stops."""
    assert dilation_for(4) == 3
    assert dilation_for(8) == 4
    assert dilation_for(24) == 8
    assert dilation_for(4) < dilation_for(24)


def test_dilate_grows_the_mask_by_the_radius() -> None:
    mask = np.zeros((16, 16), dtype=bool)
    mask[8, 8] = True

    grown = dilate(mask, 2)

    assert grown[6:11, 6:11].all()
    assert not grown[5, 5]


def test_a_zero_radius_leaves_the_mask_alone() -> None:
    mask = _square_mask()

    assert np.array_equal(dilate(mask, 0), mask)


# --- feathering ------------------------------------------------------------------------------------


def test_feathering_produces_a_transition_rather_than_a_step() -> None:
    mask = _square_mask()
    alpha = feather(mask, _frame(), width=3)

    assert alpha.min() == pytest.approx(0.0, abs=1e-6)
    assert alpha.max() == pytest.approx(1.0, abs=1e-6)

    intermediate = ((alpha > 0.05) & (alpha < 0.95)).sum()
    assert intermediate > 0, "a hard step would leave no intermediate values"


def test_the_feather_width_is_clamped_to_the_documented_range() -> None:
    mask = _square_mask()
    frame = _frame()

    narrow = feather(mask, frame, width=-5)
    wide = feather(mask, frame, width=100)

    assert np.array_equal(narrow, feather(mask, frame, width=1))
    assert np.array_equal(wide, feather(mask, frame, width=9))


def test_a_wider_feather_spreads_the_transition() -> None:
    mask = _square_mask()
    flat = np.full((64, 64), 128.0)   # no edges to snap to, so width dominates

    narrow = ((feather(mask, flat, 1) > 0.05) & (feather(mask, flat, 1) < 0.95)).sum()
    wide = ((feather(mask, flat, 9) > 0.05) & (feather(mask, flat, 9) < 0.95)).sum()

    assert wide > narrow


# --- linear light ------------------------------------------------------------------------------------


def test_the_gamma_round_trip_is_lossless() -> None:
    values = np.linspace(0, 255, 256)

    assert np.allclose(from_linear(to_linear(values)), values, atol=1e-6)


def test_blending_in_linear_light_does_not_darken_the_midpoint() -> None:
    """prd.md §5.11 — blending gamma-encoded values produces a result darker than either.

    That darkening is what puts a visible ring around every restored region.
    """
    black = np.zeros((4, 4))
    white = np.full((4, 4), 255.0)
    half = np.full((4, 4), 0.5)

    linear_blend = composite(black, white, half)
    naive_blend = black * 0.5 + white * 0.5

    assert linear_blend[0, 0] > naive_blend[0, 0]
    assert linear_blend[0, 0] == pytest.approx(from_linear(np.array([0.5]))[0], abs=1e-6)


def test_alpha_zero_and_one_are_exact() -> None:
    original = _frame()
    restored = _frame(seed=9)

    assert np.allclose(composite(original, restored, np.zeros_like(original)), original, atol=1e-6)
    assert np.allclose(composite(original, restored, np.ones_like(original)), restored, atol=1e-6)


def test_mismatched_shapes_are_rejected() -> None:
    with pytest.raises(ValueError):
        composite(np.zeros((8, 8)), np.zeros((8, 8)), np.zeros((4, 4)))


# --- temporal alpha ------------------------------------------------------------------------------


def test_the_first_frame_passes_through_unsmoothed() -> None:
    temporal = TemporalAlpha(smoothing=0.4)
    alpha = feather(_square_mask(), _frame())

    assert np.array_equal(temporal.smooth(1, alpha), alpha)


def test_a_jittering_mask_edge_is_steadied() -> None:
    """prd.md §5.11 — an edge that breathes shimmers, which is §5.10 flicker via the blender."""
    temporal = TemporalAlpha(smoothing=0.3)
    frame = _frame()

    steady = feather(_square_mask(size=20), frame)
    jittered = feather(_square_mask(size=22), frame)

    temporal.smooth(1, steady)
    smoothed = temporal.smooth(1, jittered)

    raw_change = float(np.abs(jittered - steady).mean())
    smoothed_change = float(np.abs(smoothed - steady).mean())

    assert smoothed_change < raw_change


def test_alpha_state_is_per_track() -> None:
    temporal = TemporalAlpha(smoothing=0.5)
    a = feather(_square_mask(size=10), _frame())
    b = feather(_square_mask(size=30), _frame())

    temporal.smooth(1, a)
    temporal.smooth(2, b)

    # Track 2 must not have been pulled towards track 1.
    assert np.array_equal(temporal.smooth(2, b), b)


def test_forgetting_a_track_resets_it() -> None:
    temporal = TemporalAlpha(smoothing=0.5)
    alpha = feather(_square_mask(), _frame())

    temporal.smooth(1, alpha)
    temporal.forget(1)

    assert np.array_equal(temporal.smooth(1, alpha), alpha)


# --- the whole sequence --------------------------------------------------------------------------


def test_blend_region_leaves_the_far_field_untouched() -> None:
    """Outside the dilated, feathered mask the output must be the source, exactly."""
    frame = _frame()
    restored = np.full_like(frame, 255.0)
    mask = _square_mask(size=16)

    out = blend_region(frame, restored, mask, block_size=8, feather_px=3)

    assert np.allclose(out[:5, :5], frame[:5, :5], atol=1e-6)


def test_blend_region_actually_changes_the_region() -> None:
    frame = _frame()
    restored = np.full_like(frame, 255.0)
    mask = _square_mask(size=20)

    out = blend_region(frame, restored, mask, block_size=8)

    assert out[32, 32] > frame[32, 32] + 10
