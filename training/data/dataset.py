"""Training pairs for the mosaic detector. prd.md §11.1, §11.3.

The positive class is *manufactured*, which is the whole reason this project can train a detector
without collecting a single mosaicked video: apply the degradation to a known region of a clean
frame and the mask is exact by construction.

Three choices here matter more than the rest.

**Negatives are in-distribution.** A fraction of samples carry no mosaic at all, and some carry
manufactured hard negatives — genuine low-bitrate blocking, resampling softness, heavy grain. A
detector trained only on positives learns "output a mask", not "find a mosaic" (§5.2.5, §11.4).

**Every sample is recompressed.** Codec quantisation is what a real detector sees, and the Phase 0
gate measured how much it changes (§11.3). JPEG stands in for H.264 during training because it is
per-crop and fast; it is a real DCT-quantisation artifact, not a simulation of one, but it is *not*
the same artifact — see the limitation note in `docs/phase1-detector-report.md`.

**Splits are by clip, never by frame** (§11.6). Two crops from the same shot on opposite sides of a
split would inflate every number reported.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
from PIL import Image

from degradation.mosaic import DegradationType, GridAnchor, MosaicSpec, pixelate


@dataclass(frozen=True, slots=True)
class SampleSpec:
    """What one training sample was built from. Recorded so a failure can be reproduced."""

    clip: str
    frame: int
    has_mosaic: bool
    block: int
    kind: str
    jpeg_quality: int


class ClipFrames:
    """Decoded luma frames for one clip, cached in memory.

    The corpus is small enough to hold entirely: 24 clips x 96 frames at 1920x800 luma is about
    3.5 GB as float, so frames are kept as uint8 and converted per crop.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.name = path.name
        self.frames: list[np.ndarray] = []

        with av.open(str(path)) as container:
            for frame in container.decode(container.streams.video[0]):
                self.frames.append(frame.to_ndarray(format="gray"))

    def __len__(self) -> int:
        return len(self.frames)


def load_split(
    manifest_path: Path,
    corpus_dir: Path,
    *,
    val_fraction: float = 0.25,
) -> tuple[list[ClipFrames], list[ClipFrames], dict[str, list[str]]]:
    """Splits the corpus by clip, stratified by motion band.

    Stratifying matters: a split that put every fast-motion clip in training would report a
    validation number that says nothing about fast motion, and motion is the axis the whole
    multi-frame question turns on (§1.4.1).
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    by_band: dict[str, list[str]] = {}
    for clip in manifest["clips"]:
        by_band.setdefault(clip["motion_band"], []).append(clip["name"])

    train_names: list[str] = []
    val_names: list[str] = []

    for band in sorted(by_band):
        names = sorted(by_band[band])
        take = max(1, round(len(names) * val_fraction))
        val_names.extend(names[:take])
        train_names.extend(names[take:])

    train = [ClipFrames(corpus_dir / n) for n in train_names]
    val = [ClipFrames(corpus_dir / n) for n in val_names]

    return train, val, {"train": train_names, "val": val_names}


def _random_region(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """An elliptical or rectangular region mask.

    Not always a rectangle: §5.11 blends on the mask, so a detector that only ever saw axis-aligned
    boxes would produce boundaries the blender cannot use.
    """
    mask = np.zeros((height, width), dtype=bool)

    region_h = int(rng.integers(height // 6, height // 2))
    region_w = int(rng.integers(width // 6, width // 2))
    top = int(rng.integers(0, height - region_h))
    left = int(rng.integers(0, width - region_w))

    if rng.random() < 0.5:
        mask[top : top + region_h, left : left + region_w] = True
        return mask

    ys, xs = np.mgrid[0:height, 0:width]
    cy, cx = top + region_h / 2, left + region_w / 2
    ry, rx = region_h / 2, region_w / 2
    mask[((ys - cy) / ry) ** 2 + ((xs - cx) / rx) ** 2 <= 1.0] = True

    return mask


def _jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    buffer = io.BytesIO()
    Image.fromarray(image.astype(np.uint8), mode="L").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("L"), dtype=np.uint8)


def _hard_negative(crop: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Manufactures a clean-but-confusing crop. prd.md §11.4.

    These are *real* artifacts rather than imitations of them: aggressive JPEG produces genuine
    encoder blocking, and downscale-then-upscale produces genuine resampling softness. Both look
    enough like a mosaic to a naive detector to be worth training against, and neither is one.
    """
    choice = rng.integers(0, 3)

    if choice == 0:
        # Severe blocking from a real quantiser.
        return _jpeg_roundtrip(crop, int(rng.integers(3, 12)))

    if choice == 1:
        # Upscaled low resolution: soft, but with no grid.
        factor = int(rng.integers(3, 9))
        small = Image.fromarray(crop, mode="L").resize(
            (max(1, crop.shape[1] // factor), max(1, crop.shape[0] // factor)),
            Image.Resampling.BILINEAR,
        )
        return np.asarray(
            small.resize((crop.shape[1], crop.shape[0]), Image.Resampling.BILINEAR), dtype=np.uint8
        )

    # Heavy grain.
    noisy = crop.astype(np.float64) + rng.normal(0, rng.uniform(12, 30), crop.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def make_sample(
    clips: list[ClipFrames],
    rng: np.random.Generator,
    *,
    size: int = 256,
    positive_rate: float = 0.7,
    hard_negative_rate: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, SampleSpec]:
    """Builds one (image, mask) pair.

    Returns the image as ``uint8`` and the mask as ``float32`` in ``{0, 1}``.
    """
    clip = clips[int(rng.integers(0, len(clips)))]
    frame_index = int(rng.integers(0, len(clip)))
    frame = clip.frames[frame_index]

    height, width = frame.shape
    top = int(rng.integers(0, height - size))
    left = int(rng.integers(0, width - size))
    crop = frame[top : top + size, left : left + size].copy()

    mask = np.zeros((size, size), dtype=np.float32)
    block = 0
    kind = "none"

    if rng.random() < positive_rate:
        region = _random_region(size, size, rng)

        block = int(rng.integers(4, 25))
        block_h = block if rng.random() < 0.6 else int(rng.integers(4, 25))
        degradation = DegradationType.PIXELATION if rng.random() < 0.75 else DegradationType.BOX_BLUR

        spec = MosaicSpec(
            kind=degradation,
            block_width=block,
            block_height=block_h,
            grid_offset_x=int(rng.integers(0, block)),
            grid_offset_y=int(rng.integers(0, block_h)),
            anchor=GridAnchor.SCREEN,
            opacity=float(rng.uniform(0.85, 1.0)),
        )

        degraded = pixelate(crop, spec)
        crop = np.where(region, degraded, crop).astype(np.uint8)
        mask[region] = 1.0
        kind = degradation.value

    elif rng.random() < hard_negative_rate:
        crop = _hard_negative(crop, rng)
        kind = "hard_negative"

    quality = int(rng.integers(55, 96))
    crop = _jpeg_roundtrip(crop, quality)

    return (
        crop,
        mask,
        SampleSpec(clip.name, frame_index, bool(mask.any()), block, kind, quality),
    )


def make_batch(
    clips: list[ClipFrames],
    rng: np.random.Generator,
    batch_size: int,
    **kwargs: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Builds a batch as ``(N, 1, H, W)`` float32 images in [0, 1] and matching masks."""
    images = []
    masks = []

    for _ in range(batch_size):
        image, mask, _ = make_sample(clips, rng, **kwargs)  # type: ignore[arg-type]
        images.append(image.astype(np.float32) / 255.0)
        masks.append(mask)

    return (
        np.stack(images)[:, None, :, :],
        np.stack(masks)[:, None, :, :],
    )
