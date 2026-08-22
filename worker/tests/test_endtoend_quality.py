"""Does the pipeline still restore? prd.md section 12.3, section 5.1.8.

Every quality number this project has produced came from a script somebody ran by hand. That is how
it went weeks measuring everything in a regime where restoration is impossible, and how a defect
that cost 1.5 dB - the forward operator applied to something that was already an observation - sat
in the restoration path with a full green suite.

So the claim is a test now. It builds a small screen-anchored clip, runs the real pipeline over it,
and asserts the output is closer to the original than the input was.

**It is a regression bar, not a benchmark.** The bars sit well below what is currently measured
(+2.8 dB, 100% of frames) because a test that fails on a good day is a test that gets deleted. The
number to watch is in `docs/forward-model-report.md`; this one is here to notice it collapsing.

The detector's weights are gitignored - they are part of the machine, not the repository - so this
skips where they are absent, and says so rather than passing quietly.
"""

from __future__ import annotations

import io
from pathlib import Path

import av
import numpy as np
import pytest

from demosaic_worker.analyze.profile import GridAnchor, MosaicProfile
from demosaic_worker.jobs import JobContext, JobRunner
from demosaic_worker.messages import Emitter
from demosaic_worker.metrics import psnr
from demosaic_worker.restore.ibp import block_average


def _repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "fixtures").is_dir():
            return candidate
    raise AssertionError("no 'fixtures' directory above this test file")


REPO = _repository_root()
CORPUS = REPO / "training" / "datasets" / "clean" / "tos_002.mp4"
MODEL = REPO / "models" / "detector" / "det-unet-0.2.0" / "model.pt"

WIDTH, HEIGHT = 320, 240
FRAMES = 40
PAN = 4
SPEC = MosaicProfile(block_width=10, block_height=10, anchor=GridAnchor.SCREEN)


def _region() -> np.ndarray:
    """A fixed ellipse in screen coordinates: it does not move, the content moves past it."""
    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH]
    return (((ys - HEIGHT // 2) / 60) ** 2 + ((xs - WIDTH // 2) / 80) ** 2) <= 1.0


def _write(path: Path, pictures: list[np.ndarray]) -> None:
    with av.open(str(path), mode="w") as out:
        encoder = out.add_stream("libx264", rate=24)
        encoder.width, encoder.height, encoder.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        encoder.options = {"crf": "16", "preset": "medium"}
        for picture in pictures:
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(picture), format="rgb24")
            for packet in encoder.encode(frame):
                out.mux(packet)
        for packet in encoder.encode():
            out.mux(packet)


def _luma(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as container:
        return [
            frame.to_ndarray(format="gray").astype(np.float64)
            for frame in container.decode(container.streams.video[0])
        ]


@pytest.fixture(scope="module")
def restored(tmp_path_factory: pytest.TempPathFactory) -> tuple[list, list, list, dict]:
    """Builds the clip, runs the pipeline, and returns (clean, input, output, summary)."""
    if not CORPUS.exists():
        pytest.skip(f"corpus clip missing: {CORPUS}")
    if not MODEL.exists():
        pytest.skip(f"detector weights missing (gitignored): {MODEL}")

    directory = tmp_path_factory.mktemp("endtoend")
    mask = _region()

    with av.open(str(CORPUS)) as container:
        source = [
            frame.to_ndarray(format="rgb24")
            for _, frame in zip(range(FRAMES), container.decode(container.streams.video[0]))
        ]

    clean_pictures, degraded_pictures = [], []
    for index, frame in enumerate(source):
        offset = min(index * PAN, 120)
        picture = frame[200:200 + HEIGHT, 300 + offset:300 + offset + WIDTH].copy()
        clean_pictures.append(picture)

        blocks = np.stack(
            [block_average(picture[:, :, c].astype(np.float64), SPEC, (0, 0)) for c in range(3)],
            axis=2,
        )
        degraded_pictures.append(
            np.where(mask[:, :, None], blocks, picture).astype(np.uint8)
        )

    clean_path, input_path = directory / "clean.mp4", directory / "input.mp4"
    output_path = directory / "out.mp4"
    _write(clean_path, clean_pictures)
    _write(input_path, degraded_pictures)

    context = JobContext(
        job_id="quality",
        source_path=str(input_path),
        output_path=str(output_path),
        settings={
            "detection": {"confidence": 0.45, "maskThreshold": 0.5, "minRegionArea": 512},
            "restoration": {"preset": "Balanced", "temporalWindow": "auto"},
            "encode": {"codec": "H264", "constantQuality": 14, "preset": "veryfast"},
            "modelVersion": "0.2.0",
        },
    )
    summary = JobRunner().run(context, Emitter(stream=io.StringIO()))

    return _luma(clean_path), _luma(input_path), _luma(output_path), summary


def _gains(restored) -> list[float]:
    clean, degraded, output, _ = restored
    mask = _region()
    count = min(len(clean), len(degraded), len(output))
    return [
        psnr(clean[i][mask], output[i][mask]) - psnr(clean[i][mask], degraded[i][mask])
        for i in range(count)
    ]


def test_the_pipeline_restores_the_mosaicked_region(restored) -> None:
    """T-QUALITY-ENDTOEND-01. Currently +2.8 dB; the bar is a third of that."""
    gains = _gains(restored)

    assert float(np.mean(gains)) > 1.0, (
        f"the pipeline gained {np.mean(gains):+.2f} dB inside the region; it used to gain +2.8"
    )


def test_most_frames_improve(restored) -> None:
    """A mean can be carried by a few frames. Currently 100%; the bar is 70%."""
    gains = _gains(restored)
    improved = float(np.mean([gain > 0 for gain in gains]))

    assert improved > 0.70, f"only {improved:.0%} of frames improved"


def test_it_leaves_the_rest_of_the_picture_alone(restored) -> None:
    """Section 5.1.8: restoration touches a fraction of the frame and must not cost the rest.

    Scored against the *input* rather than the original, so the encoder is the only thing being
    allowed for - what this measures is what the pipeline did on top of it.
    """
    clean, degraded, output, _ = restored
    mask = _region()
    outside = ~mask

    scores = [
        psnr(degraded[i][outside], output[i][outside])
        for i in range(min(len(degraded), len(output)))
    ]

    assert float(np.mean(scores)) > 38.0, (
        f"outside the region the output is {np.mean(scores):.1f} dB from its input"
    )


def test_the_timeline_survives(restored) -> None:
    """Section 5.1.7, through the whole pipeline rather than the media layer alone."""
    _, degraded, output, summary = restored

    assert len(output) == len(degraded)
    assert summary["framesSeen"] == len(degraded)
    assert summary["frameCountPreserved"]


def test_the_output_is_reported_as_synthetic(restored) -> None:
    """Section 1.3: where information was destroyed the result is an estimate, and the summary says so."""
    *_, summary = restored

    assert summary["framesRestored"] > 0
    assert summary["synthetic"] is True
    assert summary["passthrough"] is False
