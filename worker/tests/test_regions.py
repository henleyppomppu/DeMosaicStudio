"""Mask to regions. prd.md §5.2.1, §5.2.3, §5.2.4."""

from __future__ import annotations

import numpy as np
import pytest

from demosaic_worker.detect.regions import Region, extract_regions, iou


def _blob(height: int, width: int, top: int, left: int, h: int, w: int, value: float = 0.9) -> np.ndarray:
    field = np.zeros((height, width), dtype=np.float64)
    field[top : top + h, left : left + w] = value
    return field


def test_a_single_blob_becomes_one_region() -> None:
    regions, clamped = extract_regions(_blob(64, 64, 10, 10, 20, 20), min_area=16)

    assert len(regions) == 1
    assert not clamped
    assert regions[0].area == 400
    assert regions[0].box == (10, 10, 30, 30)
    assert regions[0].confidence == pytest.approx(0.9)


def test_separate_blobs_become_separate_regions() -> None:
    field = _blob(64, 64, 4, 4, 12, 12) + _blob(64, 64, 40, 40, 12, 12)
    regions, _ = extract_regions(field, min_area=16)

    assert len(regions) == 2


def test_regions_below_the_minimum_area_are_dropped() -> None:
    field = _blob(64, 64, 4, 4, 20, 20) + _blob(64, 64, 50, 50, 3, 3)
    regions, _ = extract_regions(field, min_area=256)

    assert len(regions) == 1
    assert regions[0].area == 400


def test_the_mask_is_the_authoritative_shape() -> None:
    """prd.md §5.2.1 — §5.11 blends on the mask, so a region without one is a schema violation."""
    field = np.zeros((32, 32))
    ys, xs = np.mgrid[0:32, 0:32]
    field[((ys - 16) ** 2 + (xs - 16) ** 2) <= 64] = 0.8

    regions, _ = extract_regions(field, min_area=16)

    assert len(regions) == 1
    region = regions[0]
    assert region.mask.shape == field.shape
    assert region.area < region.width * region.height, "a disc does not fill its box"
    assert region.fill_ratio < 0.9


def test_a_close_fills_single_pixel_holes() -> None:
    """Without it, every hole in a thresholded probability map becomes its own region."""
    field = _blob(64, 64, 10, 10, 20, 20)
    field[15, 15] = 0.0
    field[16, 18] = 0.0

    regions, _ = extract_regions(field, min_area=16)

    assert len(regions) == 1
    assert regions[0].area == 400, "the holes should have been closed"


def test_the_region_count_is_clamped_and_reported() -> None:
    """prd.md §5.2.4 — silently keeping the top N makes a busy frame look like a quiet one."""
    field = np.zeros((128, 128))
    for index in range(20):
        y, x = divmod(index, 5)
        field[y * 24 + 2 : y * 24 + 14, x * 24 + 2 : x * 24 + 14] = 0.5 + index * 0.02

    regions, clamped = extract_regions(field, min_area=16, max_regions=16)

    assert len(regions) == 16
    assert clamped


def test_regions_come_back_most_confident_first() -> None:
    field = _blob(64, 64, 2, 2, 12, 12, value=0.6) + _blob(64, 64, 40, 40, 12, 12, value=0.95)
    regions, _ = extract_regions(field, min_area=16)

    assert [round(r.confidence, 2) for r in regions] == [0.95, 0.6]


def test_a_non_2d_probability_map_is_rejected() -> None:
    with pytest.raises(ValueError):
        extract_regions(np.zeros((8, 8, 3)))


def test_iou_is_one_for_identical_boxes() -> None:
    mask = np.ones((4, 4), dtype=bool)
    a = Region(mask, (0, 0, 10, 10), 100, 0.9)

    assert iou(a, a) == pytest.approx(1.0)


def test_iou_is_zero_for_disjoint_boxes() -> None:
    mask = np.ones((4, 4), dtype=bool)
    a = Region(mask, (0, 0, 10, 10), 100, 0.9)
    b = Region(mask, (20, 20, 30, 30), 100, 0.9)

    assert iou(a, b) == 0.0


def test_iou_is_between_zero_and_one_for_partial_overlap() -> None:
    mask = np.ones((4, 4), dtype=bool)
    a = Region(mask, (0, 0, 10, 10), 100, 0.9)
    b = Region(mask, (5, 5, 15, 15), 100, 0.9)

    assert iou(a, b) == pytest.approx(25 / 175)


def test_run_labelling_matches_the_breadth_first_search() -> None:
    """Both are four-connected. Labels may be numbered differently; components must not."""
    from demosaic_worker.detect.regions import _label, _label_bfs

    rng = np.random.default_rng(9)
    mask = rng.random((60, 80)) > 0.55
    mask[10:20, 10:20] = True
    mask[30:32, :] = False

    fast, fast_count = _label(mask)
    slow, slow_count = _label_bfs(mask)

    assert fast_count == slow_count
    # Same partition: every pair of pixels is in the same component under one iff under the other.
    pairs_fast = {tuple(sorted(set(fast[slow == k]))) for k in range(1, slow_count + 1)}
    assert all(len(p) == 1 for p in pairs_fast), "a BFS component was split by the run labeller"
    pairs_slow = {tuple(sorted(set(slow[fast == k]))) for k in range(1, fast_count + 1)}
    assert all(len(p) == 1 for p in pairs_slow), "a run-labeller component was split by the BFS"


def test_close_matches_the_sliced_close_including_the_frame_edge() -> None:
    """The erosion pads with True, so a region touching the edge keeps its edge. Both forms."""
    from demosaic_worker.detect.regions import _close, _close_slices

    rng = np.random.default_rng(4)
    mask = rng.random((40, 50)) > 0.6
    mask[0:5, 0:5] = True        # touches the corner
    mask[20, 20] = False         # a single-pixel hole inside a solid block
    mask[18:23, 18:23] |= True
    mask[20, 20] = False

    np.testing.assert_array_equal(_close(mask), _close_slices(mask))


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="the torch path is the CUDA path"
)
def test_the_gpu_close_matches_the_sliced_close_exactly() -> None:
    from demosaic_worker.detect.regions import _close_slices, _close_torch

    rng = np.random.default_rng(5)
    mask = rng.random((64, 80)) > 0.6
    mask[0:4, 0:4] = True
    mask[30, 40] = False

    fast = _close_torch(mask)
    assert fast is not None
    np.testing.assert_array_equal(fast, _close_slices(mask))


@pytest.mark.parametrize("seed", range(6))
def test_run_labelling_on_awkward_shapes(seed: int) -> None:
    """U shapes, diagonal-only touches (which 4-connectivity must NOT join), single pixels, edges."""
    from demosaic_worker.detect.regions import _find_objects, _label, _label_bfs

    rng = np.random.default_rng(seed)
    mask = rng.random((37, 53)) > 0.7
    mask[0, :] = rng.random(53) > 0.5          # top edge
    mask[:, -1] = rng.random(37) > 0.5         # right edge
    mask[10, 10] = True; mask[11, 11] = True   # diagonal touch only
    mask[10, 11] = False; mask[11, 10] = False

    fast, n_fast = _label(mask)
    slow, n_slow = _label_bfs(mask)
    assert n_fast == n_slow
    # Same partition, both directions.
    for k in range(1, n_slow + 1):
        assert len(set(fast[slow == k])) == 1
    for k in range(1, n_fast + 1):
        assert len(set(slow[fast == k])) == 1

    # Boxes agree with a brute-force nonzero per label.
    for k, (rows, cols) in enumerate(_find_objects(fast, n_fast), start=1):
        ys, xs = np.nonzero(fast == k)
        assert (rows.start, rows.stop, cols.start, cols.stop) == (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
