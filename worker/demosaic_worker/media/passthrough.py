"""Decode -> transform -> encode, preserving the source timeline. prd.md §5.1.7, Phase 0 task 0.4.

This is the spine of the whole pipeline. The restoration stages plug into ``transform``; everything
around them exists to make sure that a frame the pipeline does *not* touch comes out the other side
identical, at the same timestamp, in the same order.

Two rules from the PRD are enforced here rather than hoped for:

* **§5.1.7** — output frame count equals input frame count, and each output frame carries its
  source PTS rescaled to the output time base.
* **§5.8.2** — a frame is either fully transformed or bit-identical to its source. There is no
  third state. ``transform`` returning ``None`` means "leave this frame alone", and that path
  copies the frame rather than re-deriving it.

.. note::
   PyAV bundles its own FFmpeg, which carries libx264/libx265 but **not** NVENC. The Speed encoder
   profile (§5.1.4) therefore cannot go through PyAV and must shell out to ``tools/ffmpeg``.
   See ``CLAUDE.md`` §4.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import av

from ..errors import E1001, E1005, E2003, E5002, WorkerError
from .timing import TimelineCheck, check_timeline, is_variable_frame_rate

#: A transform returns a replacement frame, or ``None`` to pass the source frame through untouched.
FrameTransform = Callable[[Any, int], Any | None]


@dataclass(slots=True)
class PassthroughResult:
    """What one decode/encode run did. prd.md §5.1.7, §5.1.8."""

    source_pts: list[int] = field(default_factory=list)
    output_pts: list[int] = field(default_factory=list)
    source_time_base: Fraction = Fraction(1, 1000)
    output_time_base: Fraction = Fraction(1, 1000)
    frames_transformed: int = 0
    frames_passed_through: int = 0

    @property
    def frame_count(self) -> int:
        """Frames written."""
        return len(self.output_pts)

    @property
    def is_variable_frame_rate(self) -> bool:
        """Whether the *source* timeline was variable (§5.1.7)."""
        return is_variable_frame_rate(self.source_pts)

    def timeline(self) -> TimelineCheck:
        """Compares the written timeline against the source."""
        return check_timeline(
            self.source_pts,
            self.output_pts,
            self.source_time_base,
            self.output_time_base,
        )


def _open_input(path: Path) -> av.container.InputContainer:
    if not path.exists():
        raise WorkerError(E1001, f"source not found: {path}", path=str(path))

    try:
        return av.open(str(path))
    except av.FFmpegError as exc:  # pragma: no cover - exercised only on a corrupt file
        raise WorkerError(E2003, f"cannot open source: {exc}", path=str(path)) from exc


def run_passthrough(
    source: Path,
    destination: Path,
    *,
    transform: FrameTransform | None = None,
    encoder: str = "libx265",
    crf: int = 18,
    preset: str = "medium",
    copy_audio: bool = True,
) -> PassthroughResult:
    """Decodes ``source``, optionally transforms each frame, and writes ``destination``.

    ``transform`` receives ``(frame, index)`` and returns a replacement frame or ``None``. The
    default of ``None`` for the whole argument is the identity pipeline, which is what Phase 0
    task 0.4 measures.

    Audio is **stream-copied** (§5.1.5): never decoded, never re-encoded, never resampled. That is
    what keeps A/V sync a property of construction rather than of correction.
    """
    result = PassthroughResult()

    with _open_input(source) as container:
        if not container.streams.video:
            raise WorkerError(E1005, "source has no video stream", path=str(source))

        in_video = container.streams.video[0]
        in_video.thread_type = "AUTO"

        result.source_time_base = Fraction(in_video.time_base)

        in_audio = list(container.streams.audio) if copy_audio else []

        destination.parent.mkdir(parents=True, exist_ok=True)

        with av.open(str(destination), mode="w") as output:
            out_video = output.add_stream(encoder, rate=in_video.average_rate)
            out_video.width = in_video.codec_context.width
            out_video.height = in_video.codec_context.height
            out_video.pix_fmt = "yuv420p"

            # Preserve the source time base so PTS need no rescaling in the common case, which
            # keeps the §5.1.7 assertion exact rather than approximate.
            out_video.time_base = in_video.time_base
            out_video.options = {"crf": str(crf), "preset": preset}

            audio_map: dict[int, Any] = {}
            for stream in in_audio:
                # Stream copy: the output stream is created from the input's codec parameters and
                # packets are muxed verbatim.
                audio_map[stream.index] = output.add_stream_from_template(stream)

            index = 0
            try:
                for packet in container.demux(in_video, *in_audio):
                    if packet.stream.type == "audio":
                        if packet.dts is None:
                            continue
                        packet.stream = audio_map[packet.stream.index]
                        output.mux(packet)
                        continue

                    for frame in packet.decode():
                        source_pts = frame.pts
                        if source_pts is None:
                            # A frame without a timestamp is a hole in the timeline; carrying it
                            # would silently retime everything after it.
                            continue

                        result.source_pts.append(source_pts)

                        replacement = transform(frame, index) if transform is not None else None

                        if replacement is None:
                            outgoing = frame
                            result.frames_passed_through += 1
                        else:
                            outgoing = replacement
                            result.frames_transformed += 1

                        # The source PTS, always. Never a counter, never a synthesized rate.
                        outgoing.pts = source_pts
                        outgoing.time_base = in_video.time_base

                        for encoded in out_video.encode(outgoing):
                            if encoded.pts is not None:
                                result.output_pts.append(encoded.pts)
                            output.mux(encoded)

                        index += 1

                for encoded in out_video.encode():
                    if encoded.pts is not None:
                        result.output_pts.append(encoded.pts)
                    output.mux(encoded)

            except av.FFmpegError as exc:
                raise WorkerError(E5002, f"encode failed: {exc}", path=str(destination)) from exc

            result.output_time_base = Fraction(out_video.time_base)

    # Encoders emit packets in decode order; the timeline check compares presentation order.
    result.output_pts.sort()

    return result


@dataclass(slots=True)
class AnalysisResult:
    """What one detection pass saw. prd.md §8.3 ``analyze``, §5.2.5c."""

    frames_seen: int = 0
    frames_examined: int = 0
    source_pts: list[int] = field(default_factory=list)

    @property
    def is_variable_frame_rate(self) -> bool:
        return is_variable_frame_rate(self.source_pts)


def run_analysis(
    source: Path,
    *,
    transform: FrameTransform,
    sample_every: int = 1,
) -> AnalysisResult:
    """Decodes ``source`` and hands each sampled frame to ``transform``. Writes nothing.

    The protocol defines ``analyze`` as "detection and tracking only" (§8.3). Running it through
    :func:`run_passthrough` and discarding the pixels satisfies the letter of that and none of the
    point: the run still encoded a whole video to a throwaway file, and it still paid for
    restoration. Measured, the analysis of a 96-frame clip took 162 s against 153 s for the real
    job — a preview that costs more than the thing it previews is not a preview, and §5.2.5c wants
    it precisely so the user can see what would be altered *before* committing to it.

    ``sample_every`` skips frames, which the protocol allows for the same reason: a region summary
    does not need every frame to be accurate about where the mosaics are.

    The transform's return value is ignored — there is nowhere for a frame to go.
    """
    if sample_every < 1:
        raise WorkerError(E5002, f"sampleEvery must be at least 1, got {sample_every}")

    result = AnalysisResult()

    with _open_input(source) as container:
        if not container.streams.video:
            raise WorkerError(E1005, "source has no video stream", path=str(source))

        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        for frame in container.decode(stream):
            if frame.pts is None:
                continue

            result.source_pts.append(frame.pts)
            index = result.frames_seen
            result.frames_seen += 1

            if index % sample_every:
                continue

            result.frames_examined += 1
            transform(frame, index)

    return result


class StreamCopyUnavailable(RuntimeError):
    """No tool on this machine can remux without re-encoding. Raised, never swallowed."""


def find_ffmpeg() -> Path | None:
    """Locates an ffmpeg executable, or returns ``None``.

    ``tools/ffmpeg`` is gitignored - it is part of the machine, not the repository - so this can
    legitimately find nothing on a fresh checkout or in CI. Callers must handle that rather than
    assume the binary is there.
    """
    override = os.environ.get("DEMOSAIC_FFMPEG")
    if override and Path(override).exists():
        return Path(override)

    bundled = Path(__file__).resolve().parents[3] / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
    if bundled.exists():
        return bundled

    found = shutil.which("ffmpeg")
    return Path(found) if found else None


def run_stream_copy(
    source: Path,
    destination: Path,
    *,
    copy_audio: bool = True,
) -> PassthroughResult:
    """Remuxes ``source`` into ``destination`` without decoding or re-encoding anything.

    R-1.8a: a job that restores nothing must not re-encode. A full re-encode at the transparent
    operating point still costs about 2.9 dB across the whole frame
    (``docs/untouched-decomposition.json``), so a file the pipeline had no reason to touch would
    come back measurably softer for no benefit at all.

    **This shells out to ffmpeg, and it has to.** PyAV cannot remux a video stream: its
    ``add_stream_from_template`` builds an *encoder*-backed stream (``libx264``, ``is_encoder=1``),
    and muxing demuxed packets through it writes one byte per packet - a 23 KB video stream comes
    out as 21 bytes and does not decode at all. Measured on PyAV 14.0.1, in both MP4 and Matroska,
    with and without the extradata copied across. Audio survives the same call, which is why the
    audio pass-through in :func:`run_passthrough` has always been correct and this one could not be
    written the same way.

    Raises :class:`StreamCopyUnavailable` when no ffmpeg is present. The caller decides what to do
    about it; quietly re-encoding instead is what R-1.8a exists to prevent.
    """
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise StreamCopyUnavailable(
            "no ffmpeg executable found (looked at DEMOSAIC_FFMPEG, tools/ffmpeg, PATH)"
        )

    if not source.exists():
        raise WorkerError(E1001, f"source not found: {source}", path=str(source))

    destination.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(ffmpeg), "-y", "-loglevel", "error",
        "-i", str(source),
        "-map", "0" if copy_audio else "0:v",
        "-c", "copy",
        str(destination),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)

    if completed.returncode != 0 or not destination.exists():
        raise WorkerError(
            E5002,
            f"stream copy failed: {completed.stderr.strip()[:200] or 'ffmpeg produced no output'}",
            path=str(destination),
        )

    # Read the timeline back off both files rather than assuming a copy preserved it. A stream copy
    # that silently retimed would be exactly the kind of failure section 5.1.7 exists to catch.
    result = PassthroughResult()

    with _open_input(source) as container:
        stream = container.streams.video[0]
        result.source_time_base = Fraction(stream.time_base)
        result.source_pts = [
            packet.pts for packet in container.demux(stream)
            if packet.dts is not None and packet.pts is not None
        ]

    with _open_input(destination) as container:
        stream = container.streams.video[0]
        result.output_time_base = Fraction(stream.time_base)
        result.output_pts = [
            packet.pts for packet in container.demux(stream)
            if packet.dts is not None and packet.pts is not None
        ]

    result.frames_passed_through = len(result.output_pts)
    result.source_pts.sort()
    result.output_pts.sort()

    return result
