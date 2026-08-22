"""Global motion estimation. prd.md §5.6, §1.4.1."""

from __future__ import annotations

import numpy as np
import pytest

from demosaic_worker.analyze.motion import (
    MotionBand,
    classify,
    estimate_translation,
    summarize,
)


def _textured(height: int = 128, width: int = 160, seed: int = 7) -> np.ndarray:
    """Random texture. Phase correlation needs detail; a flat field has no peak to find."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width)).astype(np.float64)


def _shift(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    return np.roll(np.roll(image, dy, axis=0), dx, axis=1)


@pytest.mark.parametrize(("dx", "dy"), [(0, 0), (3, 0), (0, 5), (-4, 2), (7, -6)])
def test_a_known_shift_is_recovered(dx: int, dy: int) -> None:
    base = _textured()
    moved = _shift(base, dx, dy)

    # estimate_translation(previous, current) reports how `current` moved relative to `previous`.
    est_dx, est_dy = estimate_translation(base, moved)

    assert est_dx == pytest.approx(-dx, abs=0.5)
    assert est_dy == pytest.approx(-dy, abs=0.5)


def test_a_static_pair_reports_no_motion() -> None:
    base = _textured()
    dx, dy = estimate_translation(base, base.copy())

    assert abs(dx) < 0.5
    assert abs(dy) < 0.5


def test_mismatched_shapes_are_rejected() -> None:
    with pytest.raises(ValueError):
        estimate_translation(_textured(64, 64), _textured(32, 32))


def test_a_colour_array_is_rejected() -> None:
    with pytest.raises(ValueError):
        estimate_translation(np.zeros((8, 8, 3)), np.zeros((8, 8, 3)))


@pytest.mark.parametrize(
    ("magnitude", "expected"),
    [
        (0.0, MotionBand.STATIC),
        (0.2, MotionBand.STATIC),
        (0.5, MotionBand.SLOW),
        (3.0, MotionBand.MEDIUM),
        (6.0, MotionBand.MEDIUM),
        (12.0, MotionBand.FAST),
    ],
)
def test_bands_match_the_prd_thresholds(magnitude: float, expected: MotionBand) -> None:
    assert classify(magnitude) is expected


def test_a_summary_uses_the_median_so_one_whip_pan_does_not_reclassify_a_clip() -> None:
    """A single fast jump must not drag a steady clip into the FAST band.

    The steady pan here is 1 px/frame, which prd.md §5.6 puts in MEDIUM (its low band is
    *below* 1 px/frame). The point of the test is the outlier: it drags the mean to several times
    the median while leaving the median, and therefore the band, alone.
    """
    base = _textured()

    frames = [base]
    for _ in range(8):
        frames.append(_shift(frames[-1], 1, 0))   # steady pan
    frames.append(_shift(frames[-1], 40, 0))      # one cut-like jump

    summary = summarize(frames)

    assert summary.band is MotionBand.MEDIUM
    assert summary.mean_pixels_per_frame > 4 * summary.median_pixels_per_frame, (
        "the outlier should drag the mean"
    )
    assert summary.median_pixels_per_frame < 2.0, "but not the median"
    assert summary.max_pixels_per_frame > 10
    assert summary.frames == len(frames)


def test_a_single_frame_has_no_motion() -> None:
    summary = summarize([_textured()])

    assert summary.frames == 1
    assert summary.band is MotionBand.STATIC


def test_content_shift_reports_the_direction_the_content_moved() -> None:
    """The sign that, if got wrong, makes multi-frame score *worse* than single-frame."""
    from demosaic_worker.analyze.motion import content_shift

    base = _textured()
    moved = _shift(base, 4, -3)   # content moves +4 in x, -3 in y

    dx, dy = content_shift(base, moved)

    assert dx == pytest.approx(4, abs=0.5)
    assert dy == pytest.approx(-3, abs=0.5)


def test_cumulative_shifts_are_relative_to_the_target_and_zero_at_it() -> None:
    from demosaic_worker.analyze.motion import cumulative_content_shifts

    base = _textured()
    frames = [base]
    for _ in range(4):
        frames.append(_shift(frames[-1], 2, 0))

    shifts = cumulative_content_shifts(frames, target_index=2)

    assert shifts[2] == (0.0, 0.0)
    assert shifts[3][0] == pytest.approx(2, abs=0.6)
    assert shifts[4][0] == pytest.approx(4, abs=0.6)
    assert shifts[1][0] == pytest.approx(-2, abs=0.6)
    assert shifts[0][0] == pytest.approx(-4, abs=0.6)


def test_an_out_of_range_target_is_rejected() -> None:
    from demosaic_worker.analyze.motion import cumulative_content_shifts

    with pytest.raises(IndexError):
        cumulative_content_shifts([_textured(), _textured(seed=9)], target_index=5)
