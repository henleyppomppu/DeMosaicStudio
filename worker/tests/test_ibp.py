"""Iterative back-projection. prd.md §16 Phase 0, §1.4.1.

The decisive test is the last one: with a screen-anchored grid and real motion, more frames must
recover more detail. If that fails on synthetic data where the motion is exactly known, it will
certainly fail on video, and the gate's premise is dead before it is measured.
"""

from __future__ import annotations

import numpy as np
import pytest

from demosaic_worker.analyze.profile import GridAnchor, MosaicProfile
from demosaic_worker.metrics import psnr, shift_bilinear
from demosaic_worker.restore.ibp import Observation, block_average, reconstruct, upsample_baseline


def _detailed(height: int = 96, width: int = 96, seed: int = 21) -> np.ndarray:
    """Content with detail at every scale, which is what block averaging destroys."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:height, 0:width]

    smooth = 90 + 60 * np.sin(xs / 9.0) * np.cos(ys / 11.0)
    fine = 40 * np.sin(xs / 1.7) * np.sin(ys / 2.3)
    grain = rng.normal(0, 6, size=(height, width))

    return np.clip(smooth + fine + grain, 0, 255)


def _observe(truth: np.ndarray, spec: MosaicProfile, phase: tuple[int, int], dx: float, dy: float) -> Observation:
    moved = truth if dx == 0.0 and dy == 0.0 else shift_bilinear(truth, dx, dy)
    return Observation(block_average(moved, spec, phase), dx, dy)


def test_block_average_is_constant_within_a_block() -> None:
    spec = MosaicProfile(block_width=8, block_height=8)
    out = block_average(_detailed(), spec, (0, 0))

    assert np.allclose(out[8:16, 8:16], out[8, 8])


def test_the_target_frame_must_be_first_and_stationary() -> None:
    spec = MosaicProfile(block_width=8, block_height=8)
    truth = _detailed()

    with pytest.raises(ValueError):
        reconstruct([_observe(truth, spec, (0, 0), 3.0, 0.0)], spec, (0, 0))

    with pytest.raises(ValueError):
        reconstruct([], spec, (0, 0))


def test_a_single_frame_reconstruction_does_not_fabricate_detail() -> None:
    """With one observation there is no new information, so IBP converges to the block means.

    This is the honest single-frame result: whatever a learned model would add here comes from its
    prior, not from the data (prd.md §1.3).
    """
    spec = MosaicProfile(block_width=8, block_height=8)
    truth = _detailed()
    observation = _observe(truth, spec, (0, 0), 0.0, 0.0)

    result = reconstruct([observation], spec, (0, 0), iterations=30)

    assert psnr(observation.observed, result.image) > 40.0


def test_more_frames_with_motion_recover_more_detail() -> None:
    """The Phase 0 premise, in miniature and under ideal conditions.

    Screen-anchored grid, exactly known integer motion, no codec, no noise beyond mild grain. If
    multi-frame cannot win here, it cannot win anywhere.
    """
    spec = MosaicProfile(block_width=8, block_height=8)
    phase = (0, 0)
    truth = _detailed()

    shifts = [(0.0, 0.0), (2.0, 0.0), (4.0, 3.0), (6.0, 5.0), (1.0, 6.0)]
    observations = [_observe(truth, spec, phase, dx, dy) for dx, dy in shifts]

    passthrough = psnr(truth, upsample_baseline(observations[0].observed))
    single = psnr(truth, reconstruct(observations[:1], spec, phase, iterations=30).image)
    multi = psnr(truth, reconstruct(observations, spec, phase, iterations=30).image)

    assert multi > single + 1.0, f"multi={multi:.2f} single={single:.2f} passthrough={passthrough:.2f}"


def test_frames_without_motion_add_nothing() -> None:
    """The object-anchored case in miniature (prd.md §1.4.1).

    Five identical observations carry the information of one. A pipeline that spent a multi-frame
    budget here would be doing five times the work for nothing.
    """
    spec = MosaicProfile(block_width=8, block_height=8)
    phase = (0, 0)
    truth = _detailed()

    stationary = [_observe(truth, spec, phase, 0.0, 0.0) for _ in range(5)]
    stationary = [stationary[0], *[Observation(o.observed, 0.0, 0.0) for o in stationary[1:]]]

    single = psnr(truth, reconstruct(stationary[:1], spec, phase, iterations=30).image)
    many = psnr(truth, reconstruct(stationary, spec, phase, iterations=30).image)

    assert abs(many - single) < 0.25, f"many={many:.3f} single={single:.3f}"


def test_reconstruction_is_deterministic() -> None:
    spec = MosaicProfile(block_width=8, block_height=8)
    truth = _detailed()
    observations = [_observe(truth, spec, (0, 0), dx, 0.0) for dx in (0.0, 3.0, 5.0)]

    first = reconstruct(observations, spec, (0, 0), iterations=20).image
    second = reconstruct(observations, spec, (0, 0), iterations=20).image

    assert np.array_equal(first, second)


def _block_average_reference(image, spec, phase):
    """The literal definition, kept as an oracle for the vectorised implementation."""
    phase_x, phase_y = phase
    height, width = image.shape
    out = np.empty_like(image, dtype=np.float64)

    y = -phase_y
    while y < height:
        y0, y1 = max(y, 0), min(y + spec.block_height, height)
        x = -phase_x
        while x < width:
            x0, x1 = max(x, 0), min(x + spec.block_width, width)
            if y1 > y0 and x1 > x0:
                out[y0:y1, x0:x1] = image[y0:y1, x0:x1].mean()
            x += spec.block_width
        y += spec.block_height

    return out


@pytest.mark.parametrize("block_w", [3, 8, 12])
@pytest.mark.parametrize("block_h", [4, 8])
@pytest.mark.parametrize("phase_x", [0, 1, 5])
@pytest.mark.parametrize("phase_y", [0, 3])
def test_the_vectorised_operator_matches_the_literal_definition(block_w, block_h, phase_x, phase_y):
    """Partial blocks at the edges are where a fast path usually goes quietly wrong."""
    spec = MosaicProfile(block_width=block_w, block_height=block_h)
    image = _detailed(53, 67)   # deliberately not a multiple of any block size

    fast = block_average(image, spec, (phase_x % block_w, phase_y % block_h))
    slow = _block_average_reference(image, spec, (phase_x % block_w, phase_y % block_h))

    assert np.allclose(fast, slow)


# ------------------------------------------------------------------------------------------
# The observation is given, not computed. prd.md section 5.7.
# ------------------------------------------------------------------------------------------


def test_a_single_observation_is_an_exact_no_op() -> None:
    """One observation carries no information beyond itself, and the solver must respect that.

    The residual has zero mean inside every block by construction, so its block average - the
    back-projection - is identically zero. This holds whatever grid the solver is told to use,
    which is what makes it a useful invariant: it is true even when the estimate is wrong.
    """
    rng = np.random.default_rng(7)
    truth = rng.uniform(0, 255, size=(96, 96))
    spec = MosaicProfile(block_width=8, block_height=8, anchor=GridAnchor.SCREEN)
    observed = block_average(truth, spec, (0, 0))

    for block in (4, 8, 12):
        for phase in ((0, 0), (3, 5)):
            wrong = MosaicProfile(block_width=block, block_height=block, anchor=GridAnchor.SCREEN)
            result = reconstruct([Observation(observed, 0.0, 0.0)], wrong, phase, iterations=20)

            assert np.allclose(result.image, np.clip(observed, 0, 255)), (
                f"single-frame restoration changed the picture with block={block} phase={phase}"
            )


def test_re_averaging_an_observation_destroys_picture_the_solver_never_touches() -> None:
    """Why the pipeline passes the crop, not ``block_average(crop)``.

    The forward operator turns the clean surroundings inside the ROI rectangle into block averages.
    Single-frame restoration is a no-op, so that damage goes straight through to the blend - and
    with an estimated phase it also re-quantises the mosaic onto a shifted grid. Measured on one
    clip, undoing this took the pipeline's own share of the untouched-region loss from 1.63 dB to
    0.26 dB and the halo damage from -4.21 dB to -0.52 dB.
    """
    rng = np.random.default_rng(11)
    scene = rng.uniform(0, 255, size=(64, 64))
    spec = MosaicProfile(block_width=8, block_height=8, anchor=GridAnchor.SCREEN)

    # Only the left half is mosaicked; the right half is picture the pipeline must not touch.
    crop = scene.copy()
    crop[:, :32] = block_average(scene, spec, (0, 0))[:, :32]

    as_given = reconstruct([Observation(crop, 0.0, 0.0)], spec, (0, 0), iterations=20).image
    re_averaged = reconstruct(
        [Observation(block_average(crop, spec, (0, 0)), 0.0, 0.0)], spec, (0, 0), iterations=20
    ).image

    assert np.allclose(as_given, np.clip(crop, 0, 255)), "the crop as given must survive untouched"

    untouched_error = np.abs(re_averaged[:, 32:] - crop[:, 32:]).mean()
    assert untouched_error > 1.0, (
        "re-averaging should visibly damage the unmosaicked half; if it no longer does, this test "
        "has stopped measuring the thing it was written for"
    )
