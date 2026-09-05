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


def test_a_1080p_frame_goes_through_in_one_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fifteen tiles cost 209 ms a frame of which the network was 19.6 ms; one pass is 67 ms.

    The network is fully convolutional, so the single pass is the same computation minus the seams.
    This pins the *count*: a regression back to tiling would keep every other test green.
    """
    segmenter = Segmenter(_write_model(tmp_path / "m"), device="cpu")
    calls: list[tuple[int, int]] = []
    real = segmenter._infer
    monkeypatch.setattr(segmenter, "_infer", lambda luma: calls.append(luma.shape) or real(luma))

    frame = np.random.default_rng(1).integers(0, 256, (1080, 1920)).astype(np.float64)
    probability = segmenter.probability(frame)

    assert calls == [(1080, 1920)]
    assert probability.shape == frame.shape
    assert np.isfinite(probability).all()


def test_a_frame_beyond_the_single_pass_bound_is_tiled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """4K is four times the pixels and would need four times the memory. The seams are averaged: a
    hard tile boundary puts a straight edge into the mask, and section 5.11 blends on the mask."""
    from demosaic_worker.detect import segmenter as module

    # Lower the bound rather than allocate a 4K frame on a CPU test: the rule is the same.
    monkeypatch.setattr(module, "SINGLE_PASS_MAX_PIXELS", 500 * 500)
    segmenter = Segmenter(_write_model(tmp_path / "m"), device="cpu")
    calls: list[tuple[int, int]] = []
    real = segmenter._infer
    monkeypatch.setattr(segmenter, "_infer", lambda luma: calls.append(luma.shape) or real(luma))

    frame = np.random.default_rng(1).integers(0, 256, (700, 900)).astype(np.float64)
    probability = segmenter.probability(frame)

    assert len(calls) > 1
    assert all(h <= module.TILE and w <= module.TILE for h, w in calls)
    assert probability.shape == frame.shape
    assert np.isfinite(probability).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 is the CUDA path")
def test_half_precision_on_the_gpu_reads_the_same_map_as_fp32(tmp_path: Path) -> None:
    """The detector is a sigmoid head; the last bits of fp32 are not where its answer lives."""
    frame = np.random.default_rng(3).integers(0, 256, (256, 320)).astype(np.float64)
    # Written once: `_write_model` initialises a fresh random network each call, and comparing two
    # different networks measured 0.53 where the precision difference is 0.001.
    model = _write_model(tmp_path / "m")

    on_cpu = Segmenter(model, device="cpu").probability(frame)
    on_gpu = Segmenter(model, device="cuda")
    assert on_gpu.dtype == torch.float16

    assert np.abs(on_gpu.probability(frame) - on_cpu).max() < 0.02


def test_a_non_2d_frame_is_rejected(tmp_path: Path) -> None:
    segmenter = Segmenter(_write_model(tmp_path / "m"), device="cpu")

    with pytest.raises(ValueError):
        segmenter.probability(np.zeros((32, 32, 3)))


def test_inference_is_deterministic(tmp_path: Path) -> None:
    segmenter = Segmenter(_write_model(tmp_path / "m"), device="cpu")
    frame = np.random.default_rng(2).integers(0, 256, (64, 64)).astype(np.float64)

    assert np.array_equal(segmenter.probability(frame), segmenter.probability(frame))
