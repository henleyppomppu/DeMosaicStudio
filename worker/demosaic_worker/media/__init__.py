"""Media I/O. prd.md §5.1."""

from __future__ import annotations

from .passthrough import (
    AnalysisResult,
    PassthroughResult,
    StreamCopyUnavailable,
    find_ffmpeg,
    run_analysis,
    run_passthrough,
    run_stream_copy,
)
from .probe import AudioStreamInfo, MediaInfo, probe
from .timing import TimelineCheck, check_timeline, is_variable_frame_rate, rescale_pts

__all__ = [
    "AnalysisResult",
    "AudioStreamInfo",
    "MediaInfo",
    "PassthroughResult",
    "StreamCopyUnavailable",
    "TimelineCheck",
    "check_timeline",
    "find_ffmpeg",
    "is_variable_frame_rate",
    "probe",
    "rescale_pts",
    "run_analysis",
    "run_passthrough",
    "run_stream_copy",
]
