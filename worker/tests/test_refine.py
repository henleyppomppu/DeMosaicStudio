"""The diffusion refiner's plumbing. D-44.

The pipeline itself is not under test - its weights are the user's and are not in the repository.
What is: how settings are read, how the region's chroma is unpacked from a yuv420p plane stack,
that only the *luma change* comes back (so a colour-matrix bias cannot shift the region's
brightness), and that a refiner that cannot load degrades the job with one W6101 rather than
failing it or warning per frame.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from demosaic_worker.errors import E4001, WorkerError
from demosaic_worker.restore.refine import (
    DiffusionRefiner,
    RefineSettings,
    RefineStats,
    _luma_of_rgb,
    chroma_for,
    rgb_from_yuv,
)

# --------------------------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------------------------


def test_absent_refine_means_off_with_the_shipped_defaults() -> None:
    s = RefineSettings.from_settings({"restoration": {"preset": "Fast"}})
    assert s == RefineSettings()
    assert not s.enabled and s.strength == 0.2 and s.steps == 8 and s.seed == 7


def test_settings_are_read_and_clamped() -> None:
    s = RefineSettings.from_settings({"restoration": {"refine": {
        "enabled": True, "strength": 1.7, "model": "sd15", "lora": "", "embeddings": ["a", "", "b"],
        "steps": 0, "seed": 3,
    }}})
    assert s.enabled and s.strength == 1.0 and s.model == "sd15"
    assert s.lora is None                       # empty string is "none"
    assert s.embeddings == ("a", "b")           # empties dropped
    assert s.steps == 1                         # floor


# --------------------------------------------------------------------------------------------
# chroma unpacking
# --------------------------------------------------------------------------------------------


def test_chroma_is_unpacked_from_pyavs_plane_stack() -> None:
    """PyAV's yuv420p ndarray is Y (H rows) then U then V packed row-major at half size."""
    av = pytest.importorskip("av")

    height, width = 32, 48
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    frame = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(format="yuv420p")
    planes = frame.to_ndarray(format="yuv420p")

    u_plane = np.frombuffer(bytes(frame.planes[1]), dtype=np.uint8).reshape(height // 2, -1)[:, : width // 2]
    v_plane = np.frombuffer(bytes(frame.planes[2]), dtype=np.uint8).reshape(height // 2, -1)[:, : width // 2]

    u, v = chroma_for(planes, height, (8, 4, 40, 28))
    assert u.shape == v.shape == (24, 32)
    # Nearest-upsampled: every luma pixel reads its 2x2 block's chroma.
    np.testing.assert_array_equal(u, np.repeat(np.repeat(u_plane, 2, 0), 2, 1)[4:28, 8:40])
    np.testing.assert_array_equal(v, np.repeat(np.repeat(v_plane, 2, 0), 2, 1)[4:28, 8:40])


def test_chroma_for_a_region_past_the_frame_edge_matches_the_padded_luma() -> None:
    """Roi.crop_bounds reach past the picture by the reflect padding; the chroma must too."""
    height, width = 32, 48
    planes = np.zeros((height + height // 2, width), dtype=np.uint8)
    planes[height:].ravel()[: (height // 2) * (width // 2)] = np.arange(384, dtype=np.uint8)

    u, v = chroma_for(planes, height, (40, -4, 60, 20))     # right of the frame and above it
    assert u.shape == v.shape == (24, 20)
    # In-frame part is the real chroma; the padded part mirrors it rather than being zeros.
    assert u[4:, :8].any() and u[:4, :].any() and u[:, 8:].any()


def test_rgb_from_yuv_puts_grey_at_grey() -> None:
    luma = np.full((4, 4), 128.0)
    neutral = np.full((4, 4), 128.0)
    rgb = rgb_from_yuv(luma, neutral, neutral)
    assert np.allclose(rgb[..., 0], rgb[..., 1]) and np.allclose(rgb[..., 1], rgb[..., 2])
    assert 125 < rgb[0, 0, 0] < 135


# --------------------------------------------------------------------------------------------
# luma delta through a fake pipeline
# --------------------------------------------------------------------------------------------


class _FakeImage:
    """Enough of PIL.Image for refine_luma: size, crop, resize, and ndarray conversion."""

    def __init__(self, array: np.ndarray) -> None:
        self.array = array
        self.height, self.width = array.shape[:2]
        self.size = (self.width, self.height)

    def resize(self, size: tuple[int, int], resample: Any = None) -> "_FakeImage":
        w, h = size
        ys = (np.arange(h) * self.height // h)
        xs = (np.arange(w) * self.width // w)
        return _FakeImage(self.array[ys][:, xs])

    def crop(self, box: tuple[int, int, int, int]) -> "_FakeImage":
        l, t, r, b = box
        return _FakeImage(self.array[t:b, l:r])

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        return self.array.astype(dtype) if dtype is not None else self.array


class _FakePipe:
    """Adds a constant to every channel: the only luma change should be that constant."""

    device = "cpu"

    def __init__(self, shift: float) -> None:
        self.shift = shift
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        image = kwargs["image"]
        out = np.clip(np.asarray(image, dtype=np.float64) + self.shift, 0, 255).astype(np.uint8)

        class _Result:
            images = [_FakeImage(out)]

        return _Result()


def test_only_the_luma_change_comes_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """A colour matrix that differed from PyAV's would shift brightness on every frame; taking
    the *difference* under one matrix cancels that. A pipe that adds 10 must yield luma + 10."""
    import PIL.Image

    refiner = DiffusionRefiner(Path("/nowhere"), RefineSettings(enabled=True, strength=0.2, model="x"))
    pipe = _FakePipe(shift=10.0)
    refiner._pipe = pipe
    refiner._torch = type("T", (), {"Generator": lambda self, d: type("G", (), {"manual_seed": lambda s, n: s})()})()
    monkeypatch.setattr(PIL.Image, "fromarray", lambda a: _FakeImage(np.asarray(a)))
    monkeypatch.setattr(PIL.Image, "new", lambda mode, size: _FakeImage(np.zeros((size[1], size[0], 3), np.uint8)))
    _FakeImage.paste = lambda self, im, box: self.array.__setitem__(  # type: ignore[attr-defined]
        (slice(box[1], box[1] + im.height), slice(box[0], box[0] + im.width)), im.array)

    luma = np.full((40, 60), 100.0)
    neutral = np.full((40, 60), 128.0)
    out = refiner.refine_luma(luma, neutral, neutral)

    assert out.shape == luma.shape
    assert np.allclose(out, 110.0, atol=1.5)
    call = pipe.calls[0]
    assert call["prompt"] == ""                       # C-4 by construction
    assert call["strength"] == 0.2 and call["num_inference_steps"] == 8
    # Worked at the network's scale: the 60-wide crop went in upscaled toward 512.
    assert call["image"].width >= 480


# --------------------------------------------------------------------------------------------
# failure shape
# --------------------------------------------------------------------------------------------


def test_a_missing_model_is_e4001_and_marks_the_refiner_unavailable(tmp_path: Path) -> None:
    pytest.importorskip("diffusers")
    refiner = DiffusionRefiner(tmp_path, RefineSettings(enabled=True, model="does-not-exist"))
    assert refiner.available
    with pytest.raises(WorkerError) as failure:
        refiner.refine_luma(np.zeros((8, 8)), np.zeros((8, 8)), np.zeros((8, 8)))
    assert failure.value.code is E4001
    assert not refiner.available


def test_an_unavailable_refiner_warns_once_and_the_job_completes(tmp_path: Path) -> None:
    """Through the real job: refine enabled, model missing -> one W6101, output still written."""
    from demosaic_worker.jobs import JobContext, JobRunner
    from demosaic_worker.messages import Emitter

    from test_jobs import SOURCE, AlwaysFires, _runner, _settings

    settings = _settings()
    settings["restoration"]["refine"] = {"enabled": True, "strength": 0.2, "model": "nope", "steps": 4}
    buffer = io.StringIO()
    context = JobContext(job_id="r", source_path=str(SOURCE), output_path=str(tmp_path / "o.mp4"), settings=settings)
    summary = _runner(AlwaysFires()).run(context, Emitter(stream=buffer))

    warnings = [json.loads(l) for l in buffer.getvalue().splitlines()
                if l.strip() and json.loads(l)["type"] == "log" and json.loads(l).get("code") == "W6101"]
    assert len(warnings) == 1, "one warning for the job, not one per frame"
    assert "refiner" in warnings[0]["message"]
    assert summary["regionsRefined"] == 0
    assert (tmp_path / "o.mp4").exists()
