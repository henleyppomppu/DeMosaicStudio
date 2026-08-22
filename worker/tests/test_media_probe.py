"""Media probing. prd.md §8.3, §5.16.2."""

from __future__ import annotations

from pathlib import Path

import pytest

from demosaic_worker.errors import WorkerError
from demosaic_worker.media import probe


def _repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "fixtures").is_dir():
            return candidate
    raise AssertionError("no 'fixtures' directory above this test file")


MEDIA = _repository_root() / "fixtures" / "media"

pytestmark = pytest.mark.skipif(
    not MEDIA.is_dir() or not any(MEDIA.glob("*.mp4")),
    reason="media fixtures missing; run scripts/make_fixtures.py",
)


def test_a_cfr_source_reports_its_geometry_and_codec() -> None:
    info = probe(MEDIA / "cfr_30fps.mp4")

    assert (info.width, info.height) == (160, 120)
    assert info.video_codec == "h264"
    assert info.nominal_fps == 30
    assert info.has_audio
    assert len(info.audio_streams) == 1
    assert info.size_bytes > 0


def test_vfr_is_measured_not_taken_from_the_declared_rate() -> None:
    """prd.md §5.1.7 — a VFR stream frequently declares a nominal rate.

    Believing the declaration is how an output ends up silently retimed, so the two fixtures must
    disagree on ``is_vfr`` even though both declare a frame rate.
    """
    cfr = probe(MEDIA / "cfr_30fps.mp4")
    vfr = probe(MEDIA / "vfr.mp4")

    assert cfr.is_vfr is False
    assert vfr.is_vfr is True
    assert vfr.nominal_fps is not None, "the VFR fixture still declares a rate"


def test_every_audio_track_is_reported() -> None:
    info = probe(MEDIA / "multi_audio.mkv")

    assert len(info.audio_streams) == 3
    assert all(s.codec == "aac" for s in info.audio_streams)


def test_a_missing_file_reports_e1001(tmp_path: Path) -> None:
    with pytest.raises(WorkerError) as caught:
        probe(tmp_path / "nope.mp4")

    assert caught.value.code.code == "E1001"


def test_a_corrupt_file_reports_a_numbered_code() -> None:
    source = MEDIA / "corrupt_truncated.mp4"
    if not source.exists():
        pytest.skip("corrupt fixture missing")

    with pytest.raises(WorkerError) as caught:
        probe(source)

    assert caught.value.code.code in {"E1004", "E2003"}
