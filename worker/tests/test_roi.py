"""ROI stabilisation. prd.md §5.5."""

from __future__ import annotations

import numpy as np
import pytest

from demosaic_worker.roi import ALIGNMENT, Roi, build_roi, padding_for


def _frame(height: int = 200, width: int = 300) -> np.ndarray:
    ys, xs = np.mgrid[0:height, 0:width]
    return (ys * 3 + xs * 5).astype(np.float64) % 256


# --- padding ------------------------------------------------------------------------------------


def test_padding_respects_the_absolute_minimum() -> None:
    """A 20 px region still needs real context around it."""
    assert padding_for((0, 0, 20, 20), block_size=2) == 16


def test_padding_scales_with_the_region() -> None:
    small = padding_for((0, 0, 100, 100), block_size=4)
    large = padding_for((0, 0, 600, 600), block_size=4)

    assert large > small


def test_padding_gives_at_least_two_blocks() -> None:
    """Phase estimation needs two blocks of surroundings; with fewer there is no periodicity."""
    assert padding_for((0, 0, 40, 40), block_size=24) >= 48


# --- geometry -----------------------------------------------------------------------------------


def test_an_interior_region_is_padded_on_every_side() -> None:
    roi = build_roi((100, 80, 160, 140), (200, 300), block_size=8)

    left, top, right, bottom = roi.bounds
    assert left < 100 and top < 80
    assert right > 160 and bottom > 140


def test_the_crop_is_aligned() -> None:
    """prd.md §5.5.3 — the network needs a multiple of 16 on both axes."""
    for box in [(10, 10, 43, 57), (0, 0, 31, 31), (100, 80, 161, 143)]:
        roi = build_roi(box, (200, 300), block_size=8)
        height, width = roi.aligned_size

        assert height % ALIGNMENT == 0, box
        assert width % ALIGNMENT == 0, box


def test_the_crop_matches_the_declared_aligned_size() -> None:
    frame = _frame()
    roi = build_roi((100, 80, 163, 141), frame.shape, block_size=8)

    assert roi.crop(frame).shape == roi.aligned_size


@pytest.mark.parametrize(
    "box",
    [
        (0, 0, 40, 40),          # top-left corner
        (260, 160, 300, 200),    # bottom-right corner
        (0, 80, 40, 140),        # left edge
        (140, 0, 200, 40),       # top edge
    ],
)
def test_a_region_at_the_frame_edge_still_produces_a_full_crop(box) -> None:
    """prd.md §5.5.2 — the clamp takes pixels away and reflect padding makes them up."""
    frame = _frame()
    roi = build_roi(box, frame.shape, block_size=8)
    patch = roi.crop(frame)

    assert patch.shape == roi.aligned_size
    assert np.isfinite(patch).all()


def test_edge_padding_reflects_rather_than_zeroing() -> None:
    """A hard black border is content as far as the model is concerned."""
    frame = np.full((100, 100), 200.0)
    roi = build_roi((0, 0, 30, 30), frame.shape, block_size=8)

    patch = roi.crop(frame)

    assert patch.min() == pytest.approx(200.0), "zero padding would have introduced 0"


# --- round trip ---------------------------------------------------------------------------------


def test_cropping_and_pasting_back_is_lossless() -> None:
    frame = _frame()
    roi = build_roi((100, 80, 160, 140), frame.shape, block_size=8)

    restored = roi.paste(frame, roi.crop(frame))

    assert np.array_equal(restored, frame)


def test_the_alignment_padding_never_reaches_the_output() -> None:
    """prd.md §5.5.3 — the crop is trimmed back to its unaligned bounds before compositing."""
    frame = _frame()
    roi = build_roi((100, 80, 163, 141), frame.shape, block_size=8)

    patch = roi.crop(frame)
    marked = patch.copy()
    marked[:, :] = patch  # keep the interior
    pad_left, pad_top, pad_right, pad_bottom = roi.reflect
    if pad_top:
        marked[:pad_top, :] = -999.0
    if pad_left:
        marked[:, :pad_left] = -999.0
    if pad_bottom:
        marked[-pad_bottom:, :] = -999.0
    if pad_right:
        marked[:, -pad_right:] = -999.0

    out = roi.paste(frame, marked)

    assert out.min() >= 0.0, "reflect padding leaked into the frame"


def test_the_inner_box_locates_the_region_inside_the_crop() -> None:
    frame = _frame()
    box = (100, 80, 160, 140)
    roi = build_roi(box, frame.shape, block_size=8)

    patch = roi.crop(frame)
    inner_left, inner_top, inner_right, inner_bottom = roi.inner

    assert patch[inner_top:inner_bottom, inner_left:inner_right].shape == (
        box[3] - box[1],
        box[2] - box[0],
    )
    assert np.array_equal(
        patch[inner_top:inner_bottom, inner_left:inner_right],
        frame[box[1] : box[3], box[0] : box[2]],
    )


def test_a_colour_frame_crops_and_pastes() -> None:
    frame = np.stack([_frame()] * 3, axis=2)
    roi = build_roi((100, 80, 160, 140), frame.shape[:2], block_size=8)

    patch = roi.crop(frame)
    assert patch.shape[:2] == roi.aligned_size
    assert patch.shape[2] == 3
    assert np.array_equal(roi.paste(frame, patch), frame)


def test_the_crop_is_much_smaller_than_the_frame() -> None:
    """The reason this module exists: full-frame restoration wastes an order of magnitude."""
    frame = _frame(800, 1920)
    roi = build_roi((700, 300, 1000, 500), frame.shape, block_size=10)

    crop_pixels = roi.aligned_size[0] * roi.aligned_size[1]
    frame_pixels = frame.size

    assert crop_pixels < frame_pixels / 5
