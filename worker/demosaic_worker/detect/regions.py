"""Mask to regions. prd.md §5.2.1, §5.2.3.

The segmentation mask is the primary representation — §5.11 blends on it, so it is what restoration
boundaries are made of. Bounding boxes exist for tracking, cropping and scheduling only, which is
why a region here always carries its mask and a box derived from it, never a box alone.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Region:
    """One connected mosaic region in one frame."""

    #: Boolean mask at frame resolution. The authoritative shape (§5.2.1).
    mask: np.ndarray

    #: Inclusive-exclusive bounds, ``(left, top, right, bottom)``.
    box: tuple[int, int, int, int]

    #: Pixels in the mask, not in the box.
    area: int

    #: Mean detector probability inside the region.
    confidence: float

    @property
    def width(self) -> int:
        """Bounding-box width."""
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        """Bounding-box height."""
        return self.box[3] - self.box[1]

    @property
    def centre(self) -> tuple[float, float]:
        """Bounding-box centre, used by the tracker's Kalman state (§5.3.2)."""
        left, top, right, bottom = self.box
        return (left + right) / 2.0, (top + bottom) / 2.0

    @property
    def fill_ratio(self) -> float:
        """Mask area over box area. A low value means the box is a poor proxy for the mask."""
        box_area = self.width * self.height
        return self.area / box_area if box_area else 0.0


def _label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Four-connected labelling, in numpy, by runs.

    Rows are cut into horizontal runs of True (vectorised), runs are the nodes of a union-find,
    and two runs in adjacent rows are joined when their columns overlap. Pixels are never visited
    one at a time; the only Python loop is over runs, of which a 1080p mask has hundreds to a few
    thousand rather than two million pixels. Measured: the per-pixel breadth-first search this
    replaces cost 397 ms a frame; this is a few milliseconds. ``_label_bfs`` remains as the
    specification and the oracle in ``test_regions.py``.

    **Not scipy.** ``scipy.ndimage.label`` was tried first and is faster still, but loading its
    C extension deadlocked the worker on the very first frame whenever the process had the stdin
    reader thread alive (D-37) - main thread inside ``create_module`` for a scipy DLL, reader
    blocked on stdin, forever. In-process scripts never showed it; every subprocess did, under
    the bench's minimal PATH and the C# tests' full one alike. A dependency that hangs the shipped
    engine while passing every in-process test is not one to keep for a few milliseconds.
    """
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    if not mask.any():
        return labels, 0

    # --- runs: one per maximal horizontal stretch of True ---
    padded = np.zeros((height, width + 2), dtype=bool)
    padded[:, 1:-1] = mask
    edges = np.diff(padded.astype(np.int8), axis=1)          # +1 at run start, -1 after run end
    start_rows, start_cols = np.nonzero(edges == 1)
    _, end_cols = np.nonzero(edges == -1)                     # same order as starts, row by row
    run_count = len(start_rows)

    parent = np.arange(run_count, dtype=np.int64)

    def find(i: int) -> int:
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:                               # path compression
            parent[i], i = root, parent[i]
        return root

    # --- union runs on adjacent rows whose column spans touch (4-connectivity: overlap >= 1) ---
    row_starts = np.searchsorted(start_rows, np.arange(height + 1))
    for row in range(1, height):
        above = range(row_starts[row - 1], row_starts[row])
        here = range(row_starts[row], row_starts[row + 1])
        if not above or not here:
            continue
        a = row_starts[row - 1]
        for j in here:
            s, e = start_cols[j], end_cols[j]
            # advance the pointer in the row above past runs that end before this one starts
            while a < row_starts[row] and end_cols[a] <= s:
                a += 1
            k = a
            while k < row_starts[row] and start_cols[k] < e:
                ra, rb = find(k), find(j)
                if ra != rb:
                    parent[rb] = ra
                k += 1

    # --- number the roots 1..count and paint each run with its root's number ---
    roots = np.array([find(i) for i in range(run_count)], dtype=np.int64)
    unique_roots, numbered = np.unique(roots, return_inverse=True)
    numbered = numbered.astype(np.int32) + 1
    for i in range(run_count):
        labels[start_rows[i], start_cols[i]:end_cols[i]] = numbered[i]

    return labels, int(len(unique_roots))


def _find_objects(labels: np.ndarray, count: int) -> list[tuple[slice, slice]]:
    """The bounding box of every label, as slices, in one pass over the frame."""
    ys, xs = np.nonzero(labels)
    ids = labels[ys, xs] - 1
    top = np.full(count, np.iinfo(np.int64).max, dtype=np.int64)
    left = np.full(count, np.iinfo(np.int64).max, dtype=np.int64)
    bottom = np.zeros(count, dtype=np.int64)
    right = np.zeros(count, dtype=np.int64)
    np.minimum.at(top, ids, ys)
    np.minimum.at(left, ids, xs)
    np.maximum.at(bottom, ids, ys)
    np.maximum.at(right, ids, xs)
    return [
        (slice(int(top[i]), int(bottom[i]) + 1), slice(int(left[i]), int(right[i]) + 1))
        for i in range(count)
    ]


def _label_bfs(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """The dependency-free fallback. Same result, two orders of magnitude slower on a full frame."""
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    count = 0

    for start_y in range(height):
        for start_x in range(width):
            if not mask[start_y, start_x] or labels[start_y, start_x]:
                continue

            count += 1
            queue = deque([(start_y, start_x)])
            labels[start_y, start_x] = count

            while queue:
                y, x = queue.popleft()
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not labels[ny, nx]:
                        labels[ny, nx] = count
                        queue.append((ny, nx))

    return labels, count


def _close(mask: np.ndarray) -> np.ndarray:
    """A 3x3 morphological close. prd.md §5.2.2.

    Fills the single-pixel holes a thresholded probability map leaves inside an otherwise solid
    region — without it those holes become their own regions and the count explodes.

    Dilation pads with False and erosion pads with True, so a region touching the frame edge is
    not eroded from outside the picture. The GPU form reproduces exactly that through the
    padding rules of ``max_pool2d``; the shifted-slice form below is the fallback and the
    specification.
    """
    closed = _close_torch(mask)
    return closed if closed is not None else _close_slices(mask)


def _close_torch(mask: np.ndarray) -> np.ndarray | None:
    """The close as two max-pools on the GPU. ``None`` without CUDA.

    ``max_pool2d`` pads with minus infinity, which is exactly the two paddings the close needs:
    the dilation's pad reads as False, and the erosion - written as ``1 - maxpool(1 - x)`` -
    sees its pad as False in the inverted image, which is True in the original. The sliced
    numpy close costs about 30 ms a frame at 1080p; this is under a millisecond.
    """
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None

    x = torch.as_tensor(np.ascontiguousarray(mask, dtype=np.float32), device="cuda")[None, None]
    dilated = functional.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    eroded = 1.0 - functional.max_pool2d(1.0 - dilated, kernel_size=3, stride=1, padding=1)
    return (eroded[0, 0] > 0.5).cpu().numpy()


def _close_slices(mask: np.ndarray) -> np.ndarray:
    """Dependency-free close. Eighteen full-frame passes; the GPU form is two pools."""
    padded = np.pad(mask, 1, constant_values=False)

    dilated = np.zeros_like(mask)
    for dy in range(3):
        for dx in range(3):
            dilated |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]

    padded = np.pad(dilated, 1, constant_values=True)
    eroded = np.ones_like(mask)
    for dy in range(3):
        for dx in range(3):
            eroded &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]

    return eroded


def extract_regions(
    probability: np.ndarray,
    *,
    threshold: float = 0.5,
    min_area: int = 256,
    max_regions: int = 16,
) -> tuple[list[Region], bool]:
    """Turns a detector probability map into regions.

    Returns ``(regions, clamped)``. ``clamped`` is true when ``max_regions`` dropped something, in
    which case the caller emits W3101 — silently keeping the top N would make a busy frame look
    like a quiet one (§5.2.4).
    """
    if probability.ndim != 2:
        raise ValueError(f"expected a 2-D probability map, got {probability.shape}")

    binary = _close(probability >= threshold)
    labels, count = _label(binary)

    # Every component's area in one pass, and only the ones that survive `min_area` get their
    # full-frame mask built. The loop this replaces did `labels == index` and `mask.sum()` per
    # component - two passes over the whole frame for every speckle the threshold let through,
    # most of which were then discarded for being too small. Profiled at 92 ms a frame after the
    # labelling itself had already been fixed.
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    kept = [index for index in range(1, count + 1) if areas[index] >= min_area]

    # Boxes for every component in one pass, and the per-component work confined to the box: a
    # full-frame `labels == index` and a boolean index into `probability` were two more 2 MP
    # passes per kept region.
    slices = _find_objects(labels, count)

    regions: list[Region] = []
    for index in kept:
        rows, cols = slices[index - 1]
        top, bottom, left, right = rows.start, rows.stop, cols.start, cols.stop
        inside = labels[top:bottom, left:right] == index

        mask = np.zeros(labels.shape, dtype=bool)
        mask[top:bottom, left:right] = inside

        regions.append(
            Region(
                mask=mask,
                box=(int(left), int(top), int(right), int(bottom)),
                area=int(areas[index]),
                confidence=float(probability[top:bottom, left:right][inside].mean()),
            )
        )

    regions.sort(key=lambda r: r.confidence, reverse=True)

    clamped = len(regions) > max_regions
    return regions[:max_regions], clamped


def iou(a: Region, b: Region) -> float:
    """Box IoU, used for track association (§5.3.1).

    Boxes rather than masks here on purpose: association only needs to know that two detections are
    the same object, and box IoU is cheap and stable under the mask jitter a per-frame segmenter
    produces.
    """
    ax0, ay0, ax1, ay1 = a.box
    bx0, by0, bx1, by1 = b.box

    left, top = max(ax0, bx0), max(ay0, by0)
    right, bottom = min(ax1, bx1), min(ay1, by1)

    if right <= left or bottom <= top:
        return 0.0

    intersection = (right - left) * (bottom - top)
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - intersection

    return intersection / union if union else 0.0
