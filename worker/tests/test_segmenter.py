"""Model loading and inference. prd.md §5.2.2, §14.1, §14.3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from demosaic_worker.detect.segmenter import Segmenter, load_model_info  # noqa: E402
from demosaic_worker.detect.unet import MosaicUNet  # noqa: E402
from demosaic_worker.errors import WorkerError  # noqa: E402


def _write_model(directory: Path, *, width: int = 8, corrupt_hash: bool = False) -> Path:
    directory.mkdir(parents=True, exist_ok=True)

    model = MosaicUNet(width=width)
    weights = directory / "model.pt"
    torch.save({"state_dict": model.state_dict(), "width": width}, weights)

    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if corrupt_hash:
        digest = "0" * 64

    (directory / "metadata.json").write_text(
        json.dumps({"id": "test-unet", "version": "0.0.1", "sha256": digest, "trainedOn": "nothing"}),
        encoding="utf-8",
    )
    return directory


# --- provenance (§14.1) ---------------------------------------------------------------------------


def test_a_model_directory_reports_its_identity(tmp_path: Path) -> None:
    info = load_model_info(_write_model(tmp_path / "m"))

    assert info.model_id == "test-unet"
    assert info.version == "0.0.1"
    assert len(info.sha256) == 64


def test_a_hash_mismatch_is_refused(tmp_path: Path) -> None:
    """prd.md §14.1 R-14.1a — running weights the metadata does not describe makes every downstream
    number untraceable, so this is a failure rather than a warning."""
    with pytest.raises(WorkerError) as caught:
        load_model_info(_write_model(tmp_path / "m", corrupt_hash=True))

    assert caught.value.code.code == "E3001"


def test_an_incomplete_directory_is_refused(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(WorkerError) as caught:
        load_model_info(empty)

    assert caught.value.code.code == "E3001"


def test_a_missing_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(WorkerError):
        load_model_info(tmp_path / "nope")


# --- inference -------------------------------------------------------------------------------------


def test_the_probability_map_matches_the_input_size(tmp_path: Path) -> None:
    segmenter = Segmenter(_write_model(tmp_path / "m"), device="cpu")

    for shape in [(64, 64), (100, 140), (37, 53)]:
        probability = segmenter.probability(np.zeros(shape, dtype=np.float64))
        assert probability.shape == shape, shape


def test_probabilities_are_bounded(tmp_path: Path) -> None:
    segmenter = Segmenter(_write_model(tmp_path / "m"), device="cpu")
    rng = np.random.default_rng(0)

    probability = segmenter.probability(rng.integers(0, 256, (96, 96)).astype(np.float64))

    assert probability.min() >= 0.0
    assert probability.max() <= 1.0


def test_a_frame_larger_than_a_tile_is_tiled(tmp_path: Path) -> None:
    """The seams are averaged: a hard tile boundary puts a straight edge into the mask, and §5.11
    blends on the mask, so that edge would reach the picture."""
    segmenter = Segmenter(_write_model(tmp_path / "m"), device="cpu")
    rng = np.random.default_rng(1)

    frame = rng.integers(0, 256, (700, 900)).astype(np.float64)
    probability = segmenter.probability(frame)

    assert probability.shape == frame.shape
    assert np.isfinite(probability).all()


def test_a_non_2d_frame_is_rejected(tmp_path: Path) -> None:
    segmenter = Segmenter(_write_model(tmp_path / "m"), device="cpu")

    with pytest.raises(ValueError):
        segmenter.probability(np.zeros((32, 32, 3)))


def test_inference_is_deterministic(tmp_path: Path) -> None:
    segmenter = Segmenter(_write_model(tmp_path / "m"), device="cpu")
    frame = np.random.default_rng(2).integers(0, 256, (64, 64)).astype(np.float64)

    assert np.array_equal(segmenter.probability(frame), segmenter.probability(frame))
