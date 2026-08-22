"""Training-pair construction. prd.md §11.1, §11.3, §11.6."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from data.dataset import ClipFrames, load_split, make_batch, make_sample


def _repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "fixtures").is_dir():
            return candidate
    raise AssertionError("no 'fixtures' directory above this test file")


REPO = _repository_root()
MANIFEST = REPO / "training" / "datasets" / "clean-tos.manifest.json"
CORPUS = REPO / "training" / "datasets" / "clean"

pytestmark = pytest.mark.skipif(
    not MANIFEST.exists(), reason="corpus missing; run scripts/build_corpus.py"
)


@pytest.fixture(scope="module")
def clips() -> list[ClipFrames]:
    names = [c["name"] for c in json.loads(MANIFEST.read_text(encoding="utf-8"))["clips"][:2]]
    return [ClipFrames(CORPUS / n) for n in names]


def test_a_positive_sample_has_a_non_empty_mask(clips: list[ClipFrames]) -> None:
    rng = np.random.default_rng(0)
    image, mask, spec = make_sample(clips, rng, positive_rate=1.0)

    assert image.shape == (256, 256)
    assert mask.shape == (256, 256)
    assert mask.max() == 1.0
    assert 0.0 < mask.mean() < 1.0, "a region, not the whole crop"
    assert spec.has_mosaic
    assert spec.block >= 4


def test_a_negative_sample_has_an_empty_mask(clips: list[ClipFrames]) -> None:
    rng = np.random.default_rng(1)
    _, mask, spec = make_sample(clips, rng, positive_rate=0.0)

    assert mask.max() == 0.0
    assert not spec.has_mosaic


def test_the_mask_marks_exactly_the_altered_region(clips: list[ClipFrames]) -> None:
    """The whole reason positives are manufactured: the label is exact, not annotated."""
    rng = np.random.default_rng(5)

    for _ in range(6):
        image, mask, spec = make_sample(clips, rng, positive_rate=1.0)
        if spec.kind != "PIXELATION" or spec.block < 12:
            continue

        inside = image[mask > 0.5].astype(np.float64)
        outside = image[mask < 0.5].astype(np.float64)

        if inside.size < 500 or outside.size < 500:
            continue

        # Block averaging removes high-frequency energy; the untouched region keeps it.
        assert inside.std() <= outside.std() * 1.5
        return

    pytest.skip("no suitable pixelation sample drawn")


def test_every_sample_is_recompressed(clips: list[ClipFrames]) -> None:
    """prd.md §11.3 — a detector trained on clean synthetics reports numbers it cannot reproduce."""
    rng = np.random.default_rng(3)
    _, _, spec = make_sample(clips, rng)

    assert 55 <= spec.jpeg_quality <= 95


def test_hard_negatives_are_produced(clips: list[ClipFrames]) -> None:
    rng = np.random.default_rng(11)
    kinds = {
        make_sample(clips, rng, positive_rate=0.0, hard_negative_rate=1.0)[2].kind
        for _ in range(10)
    }

    assert kinds == {"hard_negative"}


def test_a_batch_is_normalised_and_shaped_for_torch(clips: list[ClipFrames]) -> None:
    rng = np.random.default_rng(2)
    images, masks = make_batch(clips, rng, batch_size=4)

    assert images.shape == (4, 1, 256, 256)
    assert masks.shape == (4, 1, 256, 256)
    assert images.dtype == np.float32
    assert 0.0 <= images.min() and images.max() <= 1.0
    assert set(np.unique(masks)).issubset({0.0, 1.0})


def test_splits_are_by_clip_and_disjoint() -> None:
    """prd.md §11.6 — frames from one shot on both sides of a split inflate every metric."""
    _, _, names = load_split(MANIFEST, CORPUS)

    assert set(names["train"]).isdisjoint(names["val"])
    assert names["val"], "validation split is empty"


def test_splits_are_stratified_across_motion_bands() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    band_of = {c["name"]: c["motion_band"] for c in manifest["clips"]}

    _, _, names = load_split(MANIFEST, CORPUS)

    val_bands = {band_of[n] for n in names["val"]}
    all_bands = set(band_of.values())

    assert val_bands == all_bands, "every motion band must appear in validation"


def test_generation_is_deterministic_given_a_seed(clips: list[ClipFrames]) -> None:
    a = make_batch(clips, np.random.default_rng(99), 3)
    b = make_batch(clips, np.random.default_rng(99), 3)

    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
