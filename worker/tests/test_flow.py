"""Dense alignment. prd.md §5.7, §5.9.4."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from demosaic_worker.restore.flow import (  # noqa: E402
    MIN_FLOW_SIZE,
    DenseAligner,
    warp_by_flow,
)


def _textured(height: int = 128, width: int = 128, seed: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:height, 0:width]
    base = 110 + 70 * np.sin(xs / 7.0) * np.cos(ys / 9.0)
    return np.clip(base + rng.normal(0, 8, (height, width)), 0, 255).astype(np.float64)


# --- warping ------------------------------------------------------------------------------------


def test_a_zero_flow_is_the_identity() -> None:
    image = _textured()
    flow = np.zeros((*image.shape, 2), dtype=np.float32)

    assert np.allclose(warp_by_flow(image, flow), image)


def test_a_constant_flow_translates() -> None:
    image = np.zeros((16, 16))
    image[8, 8] = 100.0

    flow = np.zeros((16, 16, 2), dtype=np.float32)
    flow[..., 0] = 3.0     # sample 3 px to the right

    warped = warp_by_flow(image, flow)

    assert warped[8, 5] == pytest.approx(100.0)


def test_sub_pixel_flow_interpolates() -> None:
    image = np.zeros((16, 16))
    image[8, 8] = 100.0

    flow = np.zeros((16, 16, 2), dtype=np.float32)
    flow[..., 0] = 0.5

    warped = warp_by_flow(image, flow)

    assert warped[8, 7] == pytest.approx(50.0)
    assert warped[8, 8] == pytest.approx(50.0)


def test_a_multi_channel_image_warps() -> None:
    """Needed so a flow field can itself be warped — that is the consistency check."""
    field = np.zeros((8, 8, 2))
    field[..., 0] = 5.0

    warped = warp_by_flow(field, np.zeros((8, 8, 2), dtype=np.float32))

    assert warped.shape == (8, 8, 2)
    assert np.allclose(warped[..., 0], 5.0)


def test_malformed_flow_is_rejected() -> None:
    with pytest.raises(ValueError):
        warp_by_flow(_textured(), np.zeros((128, 128, 3)))
    with pytest.raises(ValueError):
        warp_by_flow(_textured(), np.zeros((64, 64, 2)))


# --- alignment ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def aligner() -> DenseAligner:
    return DenseAligner()


def test_identical_frames_produce_near_zero_flow(aligner: DenseAligner) -> None:
    image = _textured()
    alignment = aligner.align(image, image.copy())

    assert np.abs(alignment.target_to_neighbour).max() < 1.0
    assert alignment.usable_fraction > 0.9


def test_a_known_translation_is_recovered(aligner: DenseAligner) -> None:
    image = _textured()
    shifted = np.roll(image, 4, axis=1)      # content moves 4 px right

    alignment = aligner.align(image, shifted)

    # Interior only: the wrapped column at the edge is not a real correspondence.
    interior = alignment.target_to_neighbour[20:-20, 20:-20, 0]

    assert np.median(interior) == pytest.approx(4.0, abs=1.0)


def test_warping_the_target_by_the_flow_lands_on_the_neighbour(aligner: DenseAligner) -> None:
    """The property the reconstruction actually depends on."""
    image = _textured()
    shifted = np.roll(image, 3, axis=0)

    alignment = aligner.align(image, shifted)
    simulated = warp_by_flow(image, alignment.neighbour_to_target)

    interior = slice(20, -20)
    before = np.mean(np.abs(image[interior, interior] - shifted[interior, interior]))
    after = np.mean(np.abs(simulated[interior, interior] - shifted[interior, interior]))

    assert after < before * 0.5, f"alignment did not help: {after:.2f} vs {before:.2f}"


def test_confidence_is_low_where_content_appears_from_nowhere(aligner: DenseAligner) -> None:
    """prd.md §5.9.4 — the whole point of per-pixel confidence.

    A patch that exists in the neighbour and not in the target has no correspondence, so the
    forward-backward round trip cannot close there.
    """
    image = _textured()
    occluded = image.copy()
    occluded[40:90, 40:90] = 255.0

    alignment = aligner.align(image, occluded)

    inside = alignment.confidence[45:85, 45:85].mean()
    outside = np.concatenate(
        [alignment.confidence[:35, :].ravel(), alignment.confidence[95:, :].ravel()]
    ).mean()

    assert inside < outside, f"occluded region scored {inside:.3f} vs background {outside:.3f}"


def test_confidence_is_bounded(aligner: DenseAligner) -> None:
    alignment = aligner.align(_textured(), _textured(seed=9))

    assert alignment.confidence.min() >= 0.0
    assert alignment.confidence.max() <= 1.0


@pytest.mark.parametrize("size", [24, 48, 96, 127, 128, 200])
def test_small_regions_are_aligned_rather_than_rejected(aligner: DenseAligner, size: int) -> None:
    """RAFT needs at least 128 px on each axis; small ROIs are this pipeline's common case.

    Before the aligner padded up, every alignment on a small region raised "feature maps are too
    small" and the router fell back to single-frame everywhere without anything looking broken.
    """
    image = _textured(size, size)
    alignment = aligner.align(image, image.copy())

    assert alignment.target_to_neighbour.shape == (size, size, 2)
    assert alignment.confidence.shape == (size, size)
    assert alignment.usable_fraction > 0.5


def test_a_non_square_small_region_is_aligned(aligner: DenseAligner) -> None:
    image = _textured(40, 150)
    alignment = aligner.align(image, image.copy())

    assert alignment.target_to_neighbour.shape == (40, 150, 2)


def test_a_known_shift_is_recovered_in_a_small_region(aligner: DenseAligner) -> None:
    """Padding must not destroy the measurement it exists to enable."""
    image = _textured(64, 64)
    shifted = np.roll(image, 3, axis=1)

    alignment = aligner.align(image, shifted)
    interior = alignment.target_to_neighbour[16:-16, 16:-16, 0]

    assert np.median(interior) == pytest.approx(3.0, abs=1.5)


# ------------------------------------------------------------------------------------------
# The flow scale and the optional backward pass. D-32.
# ------------------------------------------------------------------------------------------


def test_the_flow_is_estimated_below_full_resolution() -> None:
    """Not a speed knob - a quality one, and a large one.

    Full resolution measured worst on all three clips: +2.81 dB against +5.39 at 0.75 on the
    screen-anchored one. The mosaic is a screen-fixed high-frequency texture competing with the
    content for RAFT's attention, and downscaling suppresses it.
    """
    from demosaic_worker.restore.flow import DEFAULT_FLOW_SCALE

    assert 0.0 < DEFAULT_FLOW_SCALE < 1.0


def test_a_small_region_is_not_downscaled() -> None:
    """Below RAFT's minimum the crop is padded up, so shrinking it first replaces content with
    padding. Measured: a 24 px region at 0.75 drops from a usable fraction of 0.9 to 0.31."""
    aligner = DenseAligner()

    assert aligner._effective_scale((24, 24)) == 1.0
    assert aligner._effective_scale((MIN_FLOW_SIZE, MIN_FLOW_SIZE)) == 1.0
    assert aligner._effective_scale((1000, 1000)) == aligner.flow_scale


def test_the_scale_never_takes_a_side_below_the_minimum() -> None:
    aligner = DenseAligner(flow_scale=0.1)

    for shortest in (129, 160, 200, 400, 2000):
        scale = aligner._effective_scale((shortest, shortest * 2))
        assert shortest * scale >= MIN_FLOW_SIZE - 1e-9, shortest


def test_skipping_the_backward_pass_leaves_the_field_absent_rather_than_approximated() -> None:
    """A plausible stand-in would be used without anyone noticing it was not the real thing."""
    aligner = DenseAligner()
    first, second = _drifting_pair()

    both = aligner.align(first, second)
    forward_only = aligner.align(first, second, backward=False)

    assert both.neighbour_to_target is not None
    assert forward_only.neighbour_to_target is None
    assert forward_only.target_to_neighbour.shape == both.target_to_neighbour.shape
    assert forward_only.confidence.shape == both.confidence.shape


def test_the_photometric_confidence_still_separates_a_good_alignment_from_a_hopeless_one() -> None:
    aligner = DenseAligner()
    first, second = _drifting_pair()

    aligned = aligner.align(first, second, backward=False)
    rng = np.random.default_rng(3)
    noise = rng.uniform(0, 255, size=first.shape)
    hopeless = aligner.align(first, noise, backward=False)

    assert aligned.usable_fraction > hopeless.usable_fraction + 0.2, (
        f"{aligned.usable_fraction:.2f} against {hopeless.usable_fraction:.2f}"
    )


def _drifting_pair() -> tuple[np.ndarray, np.ndarray]:
    """A textured frame and the same content shifted, big enough not to be padded up."""
    first = _textured(height=192, width=192, seed=9)
    flow = np.zeros((192, 192, 2), dtype=np.float32)
    flow[..., 0] = -3.0
    return first, warp_by_flow(first, flow)
