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
    """Four-connected labelling, breadth-first.

    Written out rather than pulled from scipy: it is twenty lines, it removes a dependency from the
    shipped engine, and the shapes involved are small because they come from a thresholded mask.
    """
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
    """
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

    regions: list[Region] = []
    for index in range(1, count + 1):
        mask = labels == index
        area = int(mask.sum())
        if area < min_area:
            continue

        ys, xs = np.nonzero(mask)
        box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

        regions.append(
            Region(
                mask=mask,
                box=box,
                area=area,
                confidence=float(probability[mask].mean()),
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
