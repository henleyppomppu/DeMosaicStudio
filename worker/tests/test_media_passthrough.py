"""Phase 0 task 0.4 — decode -> passthrough -> encode. prd.md §5.1.7, §5.1.5, §5.8.2.

These are the first tests in the repository that touch real media. They exist to pin down the one
rule that silently destroys A/V sync when it is only *assumed*: the output timeline is the source
timeline, frame for frame, timestamp for timestamp.
"""

from __future__ import annotations

import hashlib
import subprocess
from fractions import Fraction
from pathlib import Path

import av
import pytest

from demosaic_worker.errors import WorkerError
from demosaic_worker.media import is_variable_frame_rate, rescale_pts, run_passthrough


def _repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "fixtures").is_dir():
            return candidate
    raise AssertionError("no 'fixtures' directory above this test file")


REPO = _repository_root()
MEDIA = REPO / "fixtures" / "media"
FFPROBE = REPO / "tools" / "ffmpeg" / "bin" / "ffprobe.exe"

pytestmark = pytest.mark.skipif(
    not MEDIA.is_dir() or not any(MEDIA.glob("*.mp4")),
    reason="media fixtures missing; run scripts/make_fixtures.py",
)


def _presentation_pts(path: Path) -> tuple[list[int], Fraction]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        pts = [f.pts for f in container.decode(stream) if f.pts is not None]
        return sorted(pts), Fraction(stream.time_base)


def _audio_stream_count(path: Path) -> int:
    with av.open(str(path)) as container:
        return len(container.streams.audio)


def _audio_stream_digest(path: Path, index: int) -> str:
    """SHA-256 over one audio stream's coded packets, timestamps included.

    The container is opened fresh for each stream rather than seeking between them.
    ``container.seek(0)`` is not a rewind: it seeks to the nearest keyframe and drops the codec
    pre-roll, so an AAC stream digested after a seek is missing its priming packet and compares
    unequal to the identical stream digested from a fresh open. That cost an hour once.
    """
    hasher = hashlib.sha256()

    with av.open(str(path)) as container:
        stream = container.streams.audio[index]
        for packet in container.demux(stream):
            if packet.dts is None:
                continue
            hasher.update(str(packet.pts).encode("ascii"))
            hasher.update(bytes(packet))

    return hasher.hexdigest()


# --- T-IO-PTS-CFR-01 ------------------------------------------------------------------------------


def test_a_cfr_source_keeps_its_frame_count_and_timestamps(tmp_path: Path) -> None:
    """prd.md §5.1.7 — output_frames == input_frames, and each PTS survives."""
    source = MEDIA / "cfr_30fps.mp4"
    destination = tmp_path / "out.mp4"

    result = run_passthrough(source, destination, preset="ultrafast", crf=28)
    timeline = result.timeline()

    assert timeline.frame_count_preserved, timeline.describe()
    assert timeline.is_faithful(), timeline.describe()
    assert not result.is_variable_frame_rate

    written_pts, _ = _presentation_pts(destination)
    assert written_pts == result.source_pts


# --- T-IO-PTS-VFR-01 ------------------------------------------------------------------------------


def test_a_vfr_source_keeps_its_uneven_timeline(tmp_path: Path) -> None:
    """A VFR source must come out VFR, with its per-frame durations intact.

    This is the case a frame-index pipeline silently converts to CFR, and the only visible symptom
    is drift that grows across a long file.
    """
    source = MEDIA / "vfr.mp4"
    destination = tmp_path / "out.mp4"

    result = run_passthrough(source, destination, preset="ultrafast", crf=28)
    timeline = result.timeline()

    assert result.is_variable_frame_rate, "the fixture is supposed to be VFR"
    assert timeline.frame_count_preserved, timeline.describe()
    assert timeline.is_faithful(), timeline.describe()

    written_pts, _ = _presentation_pts(destination)
    assert written_pts == result.source_pts, "the uneven spacing was not preserved"

    source_deltas = [b - a for a, b in zip(result.source_pts, result.source_pts[1:], strict=False)]
    written_deltas = [b - a for a, b in zip(written_pts, written_pts[1:], strict=False)]
    assert source_deltas == written_deltas
    assert len(set(written_deltas)) > 1, "output collapsed to a constant frame rate"


# --- T-IO-AUDIO-COPY-01 ---------------------------------------------------------------------------


def test_every_audio_track_is_copied_bit_identically(tmp_path: Path) -> None:
    """prd.md §5.1.5 — audio is stream-copied, never transcoded, never resampled."""
    source = MEDIA / "multi_audio.mkv"
    destination = tmp_path / "out.mkv"

    result = run_passthrough(source, destination, preset="ultrafast", crf=28)

    assert result.timeline().frame_count_preserved

    assert _audio_stream_count(source) == 3, "the fixture is supposed to carry three audio tracks"
    assert _audio_stream_count(destination) == 3, "an audio track was dropped"

    for index in range(3):
        assert _audio_stream_digest(destination, index) == _audio_stream_digest(source, index), (
            f"audio stream {index} was altered"
        )


# --- T-DEGRADE-CHAIN / §5.8.2 ---------------------------------------------------------------------


def test_a_transform_returning_none_passes_the_frame_through(tmp_path: Path) -> None:
    """prd.md §5.8.2 — a frame is either fully transformed or untouched. No third state."""
    source = MEDIA / "cfr_30fps.mp4"
    destination = tmp_path / "out.mp4"

    seen: list[int] = []

    def transform(frame: object, index: int) -> None:
        seen.append(index)
        return None

    result = run_passthrough(source, destination, transform=transform, preset="ultrafast", crf=28)

    assert result.frames_transformed == 0
    assert result.frames_passed_through == result.frame_count
    assert seen == list(range(result.frame_count))
    assert result.timeline().frame_count_preserved


def test_a_transform_that_replaces_frames_still_preserves_the_timeline(tmp_path: Path) -> None:
    """Whatever the restoration stage does to pixels, it must not touch the clock."""
    source = MEDIA / "cfr_30fps.mp4"
    destination = tmp_path / "out.mp4"

    def darken(frame: av.VideoFrame, index: int) -> av.VideoFrame:
        array = frame.to_ndarray(format="rgb24")
        replacement = av.VideoFrame.from_ndarray(array // 2, format="rgb24")
        replacement.time_base = frame.time_base
        return replacement

    result = run_passthrough(source, destination, transform=darken, preset="ultrafast", crf=28)

    assert result.frames_passed_through == 0
    assert result.frames_transformed == result.frame_count
    assert result.timeline().is_faithful(), result.timeline().describe()


# --- error paths ----------------------------------------------------------------------------------


def test_a_missing_source_reports_e1001(tmp_path: Path) -> None:
    with pytest.raises(WorkerError) as caught:
        run_passthrough(tmp_path / "nope.mp4", tmp_path / "out.mp4")

    assert caught.value.code.code == "E1001"
    assert caught.value.recoverable is False


def test_a_corrupt_source_reports_a_numbered_code(tmp_path: Path) -> None:
    """prd.md §10.1 — never a free-text-only failure."""
    source = MEDIA / "corrupt_truncated.mp4"
    if not source.exists():
        pytest.skip("corrupt fixture missing")

    try:
        run_passthrough(source, tmp_path / "out.mp4", preset="ultrafast", crf=28)
    except WorkerError as error:
        assert error.code.code.startswith("E")
    except av.FFmpegError:  # pragma: no cover - documents the gap below
        pytest.fail(
            "a raw FFmpegError escaped run_passthrough; every media failure must carry a "
            "numbered code from prd.md §10.2"
        )


# --- timing arithmetic ----------------------------------------------------------------------------


def test_rescaling_matches_ffmpeg_rounding() -> None:
    """Half away from zero, as ``av_rescale_q`` does."""
    # Identity: same time base in and out.
    assert rescale_pts(512, Fraction(1, 15360), Fraction(1, 15360)) == 512
    assert rescale_pts(0, Fraction(1, 1000), Fraction(1, 90000)) == 0

    # 1 ms expressed in a 90 kHz clock.
    assert rescale_pts(1, Fraction(1, 1000), Fraction(1, 90000)) == 90

    # A finer target base multiplies ticks; a coarser one divides them.
    assert rescale_pts(1, Fraction(1, 2), Fraction(1, 4)) == 2
    assert rescale_pts(1, Fraction(1, 3), Fraction(1, 2)) == 1  # 2/3 of a tick rounds to 1

    # Exact halves round away from zero, not to even.
    assert rescale_pts(1, Fraction(1, 2), Fraction(1, 3)) == 2  # 1.5 -> 2
    assert rescale_pts(-1, Fraction(1, 2), Fraction(1, 3)) == -2


def test_variable_frame_rate_detection_tolerates_one_tick_of_jitter() -> None:
    """A millisecond time base rounds 30 fps to alternating 33/34 ms; that is still CFR."""
    assert not is_variable_frame_rate([0, 33, 67, 100, 133])
    assert is_variable_frame_rate([0, 512, 1536, 2048, 3584])


@pytest.mark.skipif(not FFPROBE.exists(), reason="ffprobe missing")
def test_the_output_is_readable_by_an_independent_demuxer(tmp_path: Path) -> None:
    """PyAV wrote it; make something else agree that it is a valid file."""
    destination = tmp_path / "out.mp4"
    run_passthrough(MEDIA / "cfr_30fps.mp4", destination, preset="ultrafast", crf=28)

    probe = subprocess.run(
        [
            str(FFPROBE), "-hide_banner", "-loglevel", "error",
            "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames",
            "-of", "default=nokey=1:noprint_wrappers=1",
            str(destination),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    source_pts, _ = _presentation_pts(MEDIA / "cfr_30fps.mp4")
    assert int(probe.stdout.strip()) == len(source_pts)
