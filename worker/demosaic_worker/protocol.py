"""Host <-> worker protocol. prd.md §8.

This module is the *single* definition of ``PROTOCOL_VERSION``. ``demosaic_worker.__init__``
re-exports it rather than restating it, and the host mirrors it in
``DeMosaicStudio.Domain.Protocol.ProtocolVersion`` under a parity test. Never write the version
down a third time: a duplicated constant drifts, and it drifts silently for several revisions
before anyone notices.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

#: The protocol version this worker speaks. prd.md §8.1.
PROTOCOL_VERSION: Final[str] = "1.0"


class HostMessage(str, Enum):
    """Messages the host sends. prd.md §8.2."""

    HELLO = "hello"
    PROBE = "probe"
    ANALYZE = "analyze"
    PROCESS = "process"
    PREVIEW = "preview"
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"
    SHUTDOWN = "shutdown"


class WorkerMessage(str, Enum):
    """Messages the worker sends. prd.md §8.2."""

    READY = "ready"
    PROBE_RESULT = "probeResult"
    PROGRESS = "progress"
    LOG = "log"
    TRACK_UPDATE = "trackUpdate"
    CHECKPOINT = "checkpoint"
    PREVIEW_RESULT = "previewResult"
    RESULT = "result"
    ERROR = "error"


class Stage(str, Enum):
    """Progress stages, forward-only. prd.md §8.4."""

    PROBING = "probing"
    ANALYZING = "analyzing"
    RESTORING = "restoring"
    ENCODING = "encoding"
    MUXING = "muxing"
    FINALIZING = "finalizing"


#: Stage order, used to reject a progress report that moves backwards (§8.4).
STAGE_ORDER: Final[tuple[Stage, ...]] = (
    Stage.PROBING,
    Stage.ANALYZING,
    Stage.RESTORING,
    Stage.ENCODING,
    Stage.MUXING,
    Stage.FINALIZING,
)

#: Progress is advisory and rate-limited, per job. prd.md §8.4.
MAX_PROGRESS_PER_SECOND: Final[int] = 4


@dataclass(frozen=True, slots=True)
class Version:
    """A parsed ``major.minor`` protocol version."""

    major: int
    minor: int

    @classmethod
    def parse(cls, text: str) -> "Version":
        """Parses ``major.minor``, rejecting anything else."""
        parts = text.split(".")
        if len(parts) != 2 or not all(p.isdigit() for p in parts):
            raise ValueError(f"{text!r} is not a protocol version of the form major.minor")
        return cls(int(parts[0]), int(parts[1]))

    def is_compatible_with(self, other: "Version") -> bool:
        """prd.md §8.1: differing *major* versions are refused (E7001); minor differences are fine."""
        return self.major == other.major

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"


#: This worker's version, parsed.
CURRENT: Final[Version] = Version.parse(PROTOCOL_VERSION)
