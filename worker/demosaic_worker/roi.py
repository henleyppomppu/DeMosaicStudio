"""Temporal ROI stabilisation. prd.md §5.5.

Restoration works on a padded crop around each region, not on the whole frame. That is not an
optimisation, it is the specification — and skipping it is expensive in a way that is easy to miss:
running dense flow and iterative back-projection at 1920x800 for a region covering 8% of the picture
spends more than ten times the compute and VRAM the job needs, on pixels that will be discarded.

Three rules, each of which has a failure it prevents:

* **Adaptive padding** (§5.5.1). A 20 px region and a 600 px region need different absolute context,
  and a large mosaic block needs at least two blocks of surroundings before its phase can be
  estimated at all.
* **Reflect at the frame edge, never zero** (§5.5.2). Zero-padding injects a hard black border that
  the model reads as content, and the restoration then tries to reconstruct towards it.
* **Alignment to a multiple of 16** (§5.5.3), taken from real neighbouring pixels where they exist.
  The crop is restored to its unaligned bounds before compositing, so the padding never reaches the
  output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Absolute floor on padding, whatever the region size. prd.md §5.5.1.
MIN_PADDING_PX = 16

#: Padding as a fraction of the region's short side. Default 0.15, range 0.10-0.20.
DEFAULT_PADDING_RATIO = 0.15

#: Tensor alignment required by the network. prd.md §5.5.3.
ALIGNMENT = 16


def padding_for(box: tuple[int, int, int, int], block_size: int, ratio: float = DEFAULT_PADDING_RATIO) -> int:
    """Padding in pixels for one region. prd.md §5.5.1.

    ``max(minimum, short_side * ratio, block * 2)`` — the block term is what makes phase estimation
    possible: with less than two blocks of surroundings there is no periodicity to measure.
    """
    left, top, right, bottom = box
    short_side = min(right - left, bottom - top)

    return int(max(MIN_PADDING_PX, short_side * ratio, block_size * 2))


@dataclass(frozen=True, slots=True)
class Roi:
    """A padded crop and everything needed to put it back."""

    #: Padded bounds inside the frame, clamped to it.
    bounds: tuple[int, int, int, int]

    #: The region's own bounds, relative to the crop.
    inner: tuple[int, int, int, int]

    #: Reflect padding added on each side to reach the aligned size: (left, top, right, bottom).
    reflect: tuple[int, int, int, int]

    @property
    def width(self) -> int:
        """Padded width inside the frame."""
        return self.bounds[2] - self.bounds[0]

    @property
    def height(self) -> int:
        """Padded height inside the frame."""
        return self.bounds[3] - self.bounds[1]

    @property
    def aligned_size(self) -> tuple[int, int]:
        """The crop size after reflect padding, a multiple of :data:`ALIGNMENT`."""
        left, top, right, bottom = self.reflect
        return self.height + top + bottom, self.width + left + right

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """Extracts the aligned crop, reflecting where the frame ran out."""
        left, top, right, bottom = self.bounds
        patch = frame[top:bottom, left:right]

        pad_left, pad_top, pad_right, pad_bottom = self.reflect
        if not any(self.reflect):
            return patch

        widths = [(pad_top, pad_bottom), (pad_left, pad_right)]
        if patch.ndim == 3:
            widths.append((0, 0))

        # Reflect rather than zero: a hard black border is content as far as the model is concerned.
        return np.pad(patch, widths, mode="reflect")

    def paste(self, frame: np.ndarray, restored: np.ndarray) -> np.ndarray:
        """Writes an aligned crop back, dropping the alignment padding first."""
        pad_left, pad_top, pad_right, pad_bottom = self.reflect

        trimmed = restored[
            pad_top : restored.shape[0] - pad_bottom if pad_bottom else restored.shape[0],
            pad_left : restored.shape[1] - pad_right if pad_right else restored.shape[1],
        ]

        left, top, right, bottom = self.bounds
        out = frame.copy()
        out[top:bottom, left:right] = trimmed
        return out


def build_roi(
    box: tuple[int, int, int, int],
    frame_shape: tuple[int, int],
    *,
    block_size: int = 8,
    ratio: float = DEFAULT_PADDING_RATIO,
    alignment: int = ALIGNMENT,
) -> Roi:
    """Builds a padded, aligned ROI around ``box``.

    The padded bounds are clamped to the frame; whatever the clamp took away is made up with reflect
    padding, so the crop always reaches the aligned size without inventing content.
    """
    height, width = frame_shape
    left, top, right, bottom = box

    pad = padding_for(box, block_size, ratio)

    padded_left = left - pad
    padded_top = top - pad
    padded_right = right + pad
    padded_bottom = bottom + pad

    clamped_left = max(0, padded_left)
    clamped_top = max(0, padded_top)
    clamped_right = min(width, padded_right)
    clamped_bottom = min(height, padded_bottom)

    # What the frame edge took away, plus what alignment still needs.
    reflect_left = clamped_left - padded_left
    reflect_top = clamped_top - padded_top
    reflect_right = padded_right - clamped_right
    reflect_bottom = padded_bottom - clamped_bottom

    crop_height = (clamped_bottom - clamped_top) + reflect_top + reflect_bottom
    crop_width = (clamped_right - clamped_left) + reflect_left + reflect_right

    reflect_bottom += (-crop_height) % alignment
    reflect_right += (-crop_width) % alignment

    return Roi(
        bounds=(clamped_left, clamped_top, clamped_right, clamped_bottom),
        inner=(
            left - clamped_left + reflect_left,
            top - clamped_top + reflect_top,
            right - clamped_left + reflect_left,
            bottom - clamped_top + reflect_top,
        ),
        reflect=(reflect_left, reflect_top, reflect_right, reflect_bottom),
    )
