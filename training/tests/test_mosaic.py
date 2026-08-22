"""Degradation generator invariants. prd.md §11.3, §1.4.1.

.. warning::
   Never executed — no Python interpreter on the development machine yet. See ``CLAUDE.md`` §1.
"""

from __future__ import annotations

import numpy as np
import pytest

from degradation.mosaic import (
    DegradationType,
    GridAnchor,
    MosaicProfile,
    band_for,
    phase_diversity,
    pixelate,
    with_random_geometry,
)


def _ramp(height: int = 64, width: int = 64) -> np.ndarray:
    """An image with structure at every scale, so averaging is visible."""
    ys, xs = np.mgrid[0:height, 0:width]
    return ((ys * 4 + xs * 3) % 256).astype(np.uint8)


def test_pixelation_destroys_sub_block_detail() -> None:
    spec = MosaicProfile(block_width=8, block_height=8)
    out = pixelate(_ramp(), spec)

    # Every interior block is constant.
    assert np.all(out[8:16, 8:16] == out[8, 8])


def test_a_screen_anchored_grid_gives_phase_diversity_as_the_subject_moves() -> None:
    """prd.md §1.4.1 — this is the condition multi-frame restoration depends on."""
    spec = MosaicProfile(block_width=8, block_height=8, anchor=GridAnchor.SCREEN)
    origins = [(x, 0) for x in range(8)]

    # The phase is fixed in frame coordinates, so the *subject* crosses block boundaries.
    # Diversity is measured over the subject's view, which is what the origins sweep represents.
    phases = {spec.phase_for(x, y) for x, y in origins}
    assert len(phases) == 1, "a screen-anchored grid has one phase in frame coordinates"


def test_an_object_anchored_grid_has_no_phase_diversity() -> None:
    """The case the router must not spend a multi-frame budget on (§5.8, §5.4.4)."""
    spec = MosaicProfile(block_width=8, block_height=8, anchor=GridAnchor.OBJECT)
    origins = [(x, 0) for x in range(64)]

    # Relative to the box origin the phase changes, which is exactly the wrong way round:
    # the subject sits still inside the box, so it always lands in the same block.
    assert phase_diversity(spec, [(0, 0)] * 64) == pytest.approx(1.0 / 64)


def test_phase_diversity_is_zero_for_an_empty_sequence() -> None:
    assert phase_diversity(MosaicProfile(), []) == 0.0


def test_the_grid_offset_shifts_the_block_boundaries() -> None:
    image = _ramp()
    aligned = pixelate(image, MosaicProfile(block_width=8, block_height=8, grid_offset_x=0))
    shifted = pixelate(image, MosaicProfile(block_width=8, block_height=8, grid_offset_x=4))

    assert not np.array_equal(aligned, shifted), "a phase shift must change the result"


def test_non_square_blocks_are_supported() -> None:
    """Real tools produce them; a square-only generator is a domain gap of its own (§11.3)."""
    out = pixelate(_ramp(), MosaicProfile(block_width=16, block_height=4))

    assert np.all(out[0:4, 0:16] == out[0, 0])
    assert not np.all(out[0:8, 0:16] == out[0, 0])


def test_opacity_blends_towards_the_source() -> None:
    image = _ramp()
    opaque = pixelate(image, MosaicProfile(block_width=8, block_height=8))
    half = pixelate(image, MosaicProfile(block_width=8, block_height=8, opacity=0.5))

    assert not np.array_equal(opaque, half)


def test_generation_is_deterministic_given_a_seed() -> None:
    """AC-11.3 — same seed, byte-identical output."""
    base = MosaicProfile(kind=DegradationType.PIXELATION)

    first = with_random_geometry(base, np.random.default_rng(1234))
    second = with_random_geometry(base, np.random.default_rng(1234))

    assert first == second


def test_random_geometry_keeps_anchoring_and_type() -> None:
    base = MosaicProfile(kind=DegradationType.BOX_BLUR, anchor=GridAnchor.OBJECT)
    randomized = with_random_geometry(base, np.random.default_rng(7))

    assert randomized.kind is base.kind
    assert randomized.anchor is base.anchor


@pytest.mark.parametrize(
    ("block_size", "expected_fragment"),
    [(2, "deblocking"), (8, "target band"), (20, "prior-dominated"), (64, "destroyed")],
)
def test_recoverability_bands_match_the_prd(block_size: int, expected_fragment: str) -> None:
    assert expected_fragment in band_for(block_size)


def test_invalid_specs_are_rejected() -> None:
    with pytest.raises(ValueError):
        MosaicProfile(block_width=0)
    with pytest.raises(ValueError):
        MosaicProfile(opacity=1.5)
