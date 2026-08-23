"""The job runner. prd.md FR-1.8, section 8.3, section 5.2.5c.

There were no tests here at all, which is how ``passthrough: true`` came to be reported for files
the pipeline had just fully re-encoded, and how ``analyze`` came to run the entire restoration and
throw the pixels away. Both were visible in the summary and invisible to the suite: nothing
compared what the summary claimed against what landed on disk.

So these tests assert on **bytes and cost**, not on the runner's own account of itself:

* a job that restores nothing produces a video stream byte-identical to its source;
* ``analyze`` writes no file and does not restore;
* the staging file never survives, on success or on failure.

The detector is stubbed through ``JobRunner._segmenter``, which is a lazily-created attribute and
therefore the seam. A real model would make these tests measure the model rather than the runner.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest

from demosaic_worker.errors import WorkerError
from demosaic_worker.jobs import JobContext, JobRunner
from demosaic_worker.messages import Emitter

MEDIA = Path(__file__).resolve().parent.parent.parent / "fixtures" / "media"
SOURCE = MEDIA / "cfr_30fps.mp4"


def _emitter() -> Emitter:
    """An emitter whose messages go to a buffer instead of the protocol stream."""
    return Emitter(stream=io.StringIO())


class NeverFires:
    """A detector that sees no mosaic anywhere. The R-1.8a condition."""

    def probability(self, luma: np.ndarray) -> np.ndarray:
        return np.zeros_like(luma, dtype=np.float64)


class AlwaysFires:
    """A detector that marks a fixed block, so a job has something to restore."""

    def probability(self, luma: np.ndarray) -> np.ndarray:
        probability = np.zeros_like(luma, dtype=np.float64)
        height, width = luma.shape
        probability[height // 4 : height // 4 + 80, width // 4 : width // 4 + 80] = 1.0
        return probability


def _runner(detector: Any) -> JobRunner:
    runner = JobRunner()
    runner._segmenter = detector
    return runner


def _settings(**overrides: Any) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "detection": {"confidence": 0.45, "maskThreshold": 0.5, "minRegionArea": 1024},
        "restoration": {"preset": "Fast", "temporalWindow": "auto"},
        "encode": {"codec": "H264", "constantQuality": 28, "preset": "ultrafast"},
    }
    settings.update(overrides)
    return settings


def _video_stream_digest(path: Path) -> str:
    """Hashes the compressed video packets, which is what "byte-identical" has to mean here.

    Comparing whole files would compare containers: muxers write their own headers, and a remux
    legitimately produces a different file for an identical video stream.
    """
    digest = hashlib.sha256()
    with av.open(str(path)) as container:
        for packet in container.demux(container.streams.video[0]):
            if packet.dts is None:
                continue
            digest.update(bytes(packet))
    return digest.hexdigest()


# --------------------------------------------------------------------------------------------
# R-1.8a - zero detections means a stream copy, not a re-encode
# --------------------------------------------------------------------------------------------


def test_a_job_that_restores_nothing_leaves_the_video_stream_byte_identical(tmp_path: Path) -> None:
    """T-IO-PASSTHROUGH-COPY-01.

    This is the assertion the old code could never have passed: it reported passthrough=true while
    re-encoding every frame. A re-encode at the transparent operating point still costs about
    2.9 dB across the whole frame, so "we decided not to touch it" has to mean the bytes.
    """
    destination = tmp_path / "out.mp4"

    context = JobContext(
        job_id="j1", source_path=str(SOURCE), output_path=str(destination),
        settings=_settings(),
    )
    summary = _runner(NeverFires()).run(context, _emitter())

    assert summary["passthrough"] is True
    assert summary["framesRestored"] == 0
    assert summary["regionsDetected"] == 0
    assert _video_stream_digest(destination) == _video_stream_digest(SOURCE)


def test_a_job_that_restores_something_does_not_claim_passthrough(tmp_path: Path) -> None:
    destination = tmp_path / "out.mp4"

    context = JobContext(
        job_id="j2", source_path=str(SOURCE), output_path=str(destination),
        settings=_settings(),
    )
    summary = _runner(AlwaysFires()).run(context, _emitter())

    assert summary["regionsDetected"] > 0
    assert summary["passthrough"] is False
    assert _video_stream_digest(destination) != _video_stream_digest(SOURCE)


def test_the_staging_file_never_survives(tmp_path: Path) -> None:
    """Both branches remove it: the copy branch discards it, the encode branch renames it."""
    for name, detector in (("copy", NeverFires()), ("encode", AlwaysFires())):
        destination = tmp_path / f"{name}.mp4"
        context = JobContext(
            job_id=name, source_path=str(SOURCE), output_path=str(destination),
            settings=_settings(),
        )
        _runner(detector).run(context, _emitter())

        assert destination.exists()
        assert not destination.with_name(f"{destination.stem}.part{destination.suffix}").exists()


def test_a_failure_mid_encode_leaves_no_staging_file(tmp_path: Path) -> None:
    """A truncated file at the destination would look finished. One in staging looks like nothing."""

    class Explodes:
        def probability(self, luma: np.ndarray) -> np.ndarray:
            raise MemoryError("out of memory")

    destination = tmp_path / "out.mp4"
    context = JobContext(
        job_id="j3", source_path=str(SOURCE), output_path=str(destination),
        settings=_settings(),
    )

    # A detector that raises is caught per frame and passed through, so the job still completes;
    # what matters is that no staging file is left behind either way.
    _runner(Explodes()).run(context, _emitter())

    assert not destination.with_name(f"{destination.stem}.part{destination.suffix}").exists()


def test_process_without_an_output_path_is_refused(tmp_path: Path) -> None:
    context = JobContext(job_id="j4", source_path=str(SOURCE), output_path="", settings=_settings())

    with pytest.raises(WorkerError) as error:
        _runner(NeverFires()).run(context, _emitter())

    assert "outputPath" in str(error.value)


# --------------------------------------------------------------------------------------------
# section 8.3 - analyze is detection and tracking only
# --------------------------------------------------------------------------------------------


def test_analyze_writes_no_file(tmp_path: Path) -> None:
    """It used to write a video next to the source when no output path was given.

    A preview that leaves a file behind is not a preview, and `run_job.py --dry-run` advertised
    that it wrote nothing while this was happening.
    """
    before = set(tmp_path.iterdir())

    context = JobContext(
        job_id="a1", source_path=str(SOURCE), output_path=str(tmp_path / "unused.mp4"),
        settings=_settings(), analyze_only=True,
    )
    summary = _runner(AlwaysFires()).run(context, _emitter())

    assert set(tmp_path.iterdir()) == before
    assert not list(SOURCE.parent.glob("*.analysis.mp4"))
    assert summary["framesSeen"] > 0


def test_analyze_detects_but_does_not_restore() -> None:
    """The protocol says "detection and tracking only". Restoration is what made it cost more than
    the job it previews."""
    context = JobContext(
        job_id="a2", source_path=str(SOURCE), output_path="", settings=_settings(),
        analyze_only=True,
    )
    summary = _runner(AlwaysFires()).run(context, _emitter())

    assert summary["regionsDetected"] > 0
    assert summary["framesWithRegions"] > 0
    assert summary["framesRestored"] == 0
    assert summary["synthetic"] is False


def test_analyze_is_cheaper_than_processing(tmp_path: Path) -> None:
    """Guards the property, not a wall-clock number.

    The old implementation ran the identical transform for both, so the two summaries were
    indistinguishable in the one way that mattered: whether restoration ran.
    """
    analyzed = JobContext(
        job_id="a3", source_path=str(SOURCE), output_path="", settings=_settings(),
        analyze_only=True,
    )
    analysis = _runner(AlwaysFires()).run(analyzed, _emitter())

    processed = JobContext(
        job_id="p3", source_path=str(SOURCE), output_path=str(tmp_path / "out.mp4"),
        settings=_settings(),
    )
    process = _runner(AlwaysFires()).run(processed, _emitter())

    assert analysis["framesRestored"] == 0
    assert process["framesRestored"] > 0
    assert analysis["regionsDetected"] == process["regionsDetected"]


@pytest.mark.parametrize("sample_every", [1, 2, 5])
def test_analyze_honours_sample_every(sample_every: int) -> None:
    """`sampleEvery` is in the protocol table and was accepted by nobody."""
    context = JobContext(
        job_id="a4", source_path=str(SOURCE), output_path="", settings=_settings(),
        analyze_only=True, sample_every=sample_every,
    )
    summary = _runner(AlwaysFires()).run(context, _emitter())

    seen = summary["framesSeen"]
    assert summary["framesExamined"] == pytest.approx(
        -(-seen // sample_every), abs=1
    ), f"examined {summary['framesExamined']} of {seen} at every {sample_every}"


def test_a_passthrough_that_cannot_stream_copy_says_so_instead_of_claiming_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The branch that could quietly restore the original defect.

    tools/ffmpeg is gitignored, so a fresh checkout genuinely has no way to remux. The job must
    still produce a usable file - but it must not report a pass-through it did not perform, which
    is precisely what the old `regions_detected == 0` summary did.
    """
    from demosaic_worker.media import passthrough

    monkeypatch.setattr(passthrough, "find_ffmpeg", lambda: None)

    destination = tmp_path / "out.mp4"
    context = JobContext(
        job_id="j5", source_path=str(SOURCE), output_path=str(destination),
        settings=_settings(),
    )

    buffer = io.StringIO()
    summary = _runner(NeverFires()).run(context, Emitter(stream=buffer))

    assert summary["framesRestored"] == 0
    assert summary["passthrough"] is False, "claimed a stream copy it could not perform"
    assert destination.exists(), "the job still has to produce a file"
    assert "W5102" in buffer.getvalue(), "the fallback has to be reported, not silent"


def test_analyze_reports_every_frame_it_decoded_not_just_the_sampled_ones() -> None:
    """framesSeen is a fact about the file; framesExamined is a fact about the sampling."""
    context = JobContext(
        job_id="a5", source_path=str(SOURCE), output_path="", settings=_settings(),
        analyze_only=True, sample_every=4,
    )
    sampled = _runner(AlwaysFires()).run(context, _emitter())

    every = JobContext(
        job_id="a6", source_path=str(SOURCE), output_path="", settings=_settings(),
        analyze_only=True,
    )
    full = _runner(AlwaysFires()).run(every, _emitter())

    assert sampled["framesSeen"] == full["framesSeen"]
    assert sampled["framesExamined"] < full["framesExamined"]


# --------------------------------------------------------------------------------------------
# section 5.8.1 - the confidence gate. It has always existed and has never been exercised.
# --------------------------------------------------------------------------------------------


def test_the_confidence_gate_withholds_and_leaves_the_source_pixels(tmp_path: Path) -> None:
    """A withheld region must come back as the original picture, not as a weaker restoration.

    `minRestorationConfidence` defaults to 0.0, which disables it, so nothing in the suite had ever
    run with it on. `docs/gate-calibration.json` measures what it is worth; this measures that it
    does what it says.
    """
    ungated = JobContext(
        job_id="g1", source_path=str(SOURCE), output_path=str(tmp_path / "ungated.mp4"),
        settings=_settings(),
    )
    open_summary = _runner(AlwaysFires()).run(ungated, _emitter())

    settings = _settings()
    settings["restoration"] = dict(settings["restoration"])
    settings["restoration"]["minRestorationConfidence"] = 1.01  # above any reachable confidence

    gated = JobContext(
        job_id="g2", source_path=str(SOURCE), output_path=str(tmp_path / "gated.mp4"),
        settings=settings,
    )
    shut_summary = _runner(AlwaysFires()).run(gated, _emitter())

    assert open_summary["framesRestored"] > 0, "the ungated arm has to restore something"
    assert shut_summary["framesRestored"] == 0, "a gate above every confidence must withhold all"
    assert shut_summary["regionsDetected"] == open_summary["regionsDetected"], (
        "gating must not change what was detected, only what was written"
    )
    assert shut_summary["regionsGated"] > 0, "withheld regions have to be reported (R-8.1d)"


def test_a_fully_gated_job_is_a_pass_through(tmp_path: Path) -> None:
    """The two features have to compose: withholding everything means writing nothing, and writing
    nothing means R-1.8a applies."""
    settings = _settings()
    settings["restoration"] = dict(settings["restoration"])
    settings["restoration"]["minRestorationConfidence"] = 1.01

    destination = tmp_path / "gated.mp4"
    context = JobContext(
        job_id="g3", source_path=str(SOURCE), output_path=str(destination), settings=settings,
    )
    summary = _runner(AlwaysFires()).run(context, _emitter())

    assert summary["framesRestored"] == 0
    assert summary["passthrough"] is True
    assert _video_stream_digest(destination) == _video_stream_digest(SOURCE)


# --------------------------------------------------------------------------------------------
# Progress belongs to the decode, not to the restoration
# --------------------------------------------------------------------------------------------


def test_a_job_with_nothing_to_restore_still_reports_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The defect: progress was emitted only on the path that restored a frame.

    Every early return - no region, detector failure, cancelled, analysis - skipped it, and those
    are the ordinary cases. A video the detector never fires on reported 0% from start to finish
    while the decoder worked through the whole file, which is what the window showed.

    The rate limiter is lifted because it is not what is under test: at four messages a second a
    short fixture can legitimately emit only the two forced ones, and the defect would survive.
    """
    from demosaic_worker import messages as messages_module

    monkeypatch.setattr(messages_module, "MAX_PROGRESS_PER_SECOND", 10_000)

    buffer = io.StringIO()
    context = JobContext(
        job_id="progress-1",
        source_path=str(SOURCE),
        output_path=str(tmp_path / "out.mp4"),
        settings=_settings(),
    )
    _runner(NeverFires()).run(context, Emitter(stream=buffer))

    reports = [
        json.loads(line)
        for line in buffer.getvalue().splitlines()
        if line.strip() and json.loads(line)["type"] == "progress"
    ]
    moving = [r for r in reports if r["stage"] == "restoring" and 0.0 < r["fraction"] < 1.0]

    assert moving, "no progress between the forced endpoints on a job that restored nothing"
    assert [r["fraction"] for r in moving] == sorted(r["fraction"] for r in moving)
