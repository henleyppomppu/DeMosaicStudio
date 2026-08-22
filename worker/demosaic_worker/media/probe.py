"""Media inspection. prd.md §8.3 (probeResult.media).

Facts only. Nothing here decides anything; the host and the router do that from what this reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import av

from ..errors import E1001, E1005, E2003, WorkerError
from .timing import is_variable_frame_rate


@dataclass(frozen=True, slots=True)
class AudioStreamInfo:
    """One audio stream. prd.md §8.3."""

    index: int
    codec: str
    sample_rate: int | None
    channels: int | None
    language: str | None


@dataclass(frozen=True, slots=True)
class MediaInfo:
    """What the host needs to display and to plan a job. prd.md §5.16.2, §8.3."""

    path: str
    container: str
    size_bytes: int
    duration_seconds: float | None
    width: int
    height: int
    nominal_fps: Fraction | None
    is_vfr: bool
    video_codec: str
    pixel_format: str | None
    rotation: int
    audio_streams: list[AudioStreamInfo] = field(default_factory=list)
    subtitle_streams: list[int] = field(default_factory=list)

    @property
    def has_audio(self) -> bool:
        """Whether anything needs stream-copying alongside the video (§5.1.5)."""
        return bool(self.audio_streams)


def probe(path: Path, *, sample_frames: int = 120) -> MediaInfo:
    """Reads media facts.

    ``is_vfr`` is **measured** from decoded timestamps rather than taken from the container's
    declared frame rate. A VFR stream frequently declares a nominal rate, and believing it is how
    an output ends up silently retimed (§5.1.7).
    """
    if not path.exists():
        raise WorkerError(E1001, f"source not found: {path}", path=str(path))

    try:
        container = av.open(str(path))
    except av.FFmpegError as exc:
        raise WorkerError(E2003, f"cannot open source: {exc}", path=str(path)) from exc

    with container:
        if not container.streams.video:
            raise WorkerError(E1005, "source has no video stream", path=str(path))

        video = container.streams.video[0]

        pts: list[int] = []
        for frame in container.decode(video):
            if frame.pts is not None:
                pts.append(frame.pts)
            if len(pts) >= sample_frames:
                break

        rotation = 0
        try:
            rotation = int(video.side_data.get("DISPLAYMATRIX", 0) or 0)
        except (AttributeError, TypeError, ValueError):
            rotation = 0

        return MediaInfo(
            path=str(path),
            container=container.format.name,
            size_bytes=path.stat().st_size,
            duration_seconds=(float(container.duration) / 1_000_000) if container.duration else None,
            width=video.codec_context.width,
            height=video.codec_context.height,
            nominal_fps=Fraction(video.average_rate) if video.average_rate else None,
            is_vfr=is_variable_frame_rate(sorted(pts)),
            video_codec=video.codec_context.name,
            pixel_format=str(video.codec_context.pix_fmt) if video.codec_context.pix_fmt else None,
            rotation=rotation,
            audio_streams=[
                AudioStreamInfo(
                    index=s.index,
                    codec=s.codec_context.name,
                    sample_rate=s.codec_context.sample_rate,
                    channels=getattr(s.codec_context, "channels", None),
                    language=s.metadata.get("language"),
                )
                for s in container.streams.audio
            ],
            subtitle_streams=[s.index for s in container.streams.subtitles],
        )
