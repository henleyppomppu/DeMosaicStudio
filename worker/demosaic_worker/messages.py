"""Protocol message construction and parsing. prd.md §8.

Split from :mod:`demosaic_worker.protocol` so that the *types* stay importable without pulling in
the dispatch machinery — the host mirror's parity test reads ``protocol.py`` as a source file, and a
module that only declares things is much harder to break by accident.

The rules that live here rather than in the loop, because they are the ones that get forgotten:

* **stdout is the protocol channel.** Nothing else may write to it. A stray ``print`` corrupts the
  stream and the host reports E7006.
* **Progress is rate-limited and monotonic** (§8.4). The worker never emits a lower fraction than it
  has already emitted and never moves a stage backwards, so a host that merely *drops* bad progress
  is enough to make the display correct.
* **Unknown fields are ignored, never rejected** (§8.1). That is what lets a newer host add fields
  without breaking an older worker.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import IO, Any

from .errors import E7006, ErrorCode, WorkerError
from .protocol import (
    MAX_PROGRESS_PER_SECOND,
    PROTOCOL_VERSION,
    STAGE_ORDER,
    HostMessage,
    Stage,
    WorkerMessage,
)


def new_id() -> str:
    """A message id. Random rather than sequential so two workers' logs never collide."""
    return str(uuid.uuid4())


@dataclass(frozen=True, slots=True)
class Request:
    """One decoded host message."""

    type: HostMessage
    id: str
    job_id: str | None
    payload: dict[str, Any]

    def require(self, key: str) -> Any:
        """Fetches a required field, raising E7006 when it is absent."""
        if key not in self.payload:
            raise WorkerError(E7006, f"{self.type.value} is missing required field {key!r}")
        return self.payload[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Fetches an optional field."""
        return self.payload.get(key, default)


def parse_request(line: str) -> Request:
    """Parses one JSON Lines message from the host.

    Raises :class:`WorkerError` with E7006 for anything malformed. Unknown *fields* are kept in the
    payload rather than rejected; unknown *types* are an error, because acting on a message we do
    not understand is worse than refusing it.
    """
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        raise WorkerError(E7006, f"not valid JSON: {exc}") from exc

    if not isinstance(obj, dict):
        raise WorkerError(E7006, "top-level message must be an object")

    raw_type = obj.get("type")
    try:
        message_type = HostMessage(raw_type)
    except ValueError as exc:
        raise WorkerError(E7006, f"unknown message type {raw_type!r}") from exc

    return Request(
        type=message_type,
        id=str(obj.get("id") or new_id()),
        job_id=obj.get("jobId"),
        payload={k: v for k, v in obj.items() if k not in {"v", "type", "id", "jobId"}},
    )


@dataclass
class Emitter:
    """Writes protocol messages, with the §8.4 progress rules enforced here rather than by callers.

    A caller that emits progress in a loop should not have to remember the rate limit or the
    monotonicity rule; forgetting either produces a display bug that is tedious to trace back from.
    """

    stream: IO[str] = field(default_factory=lambda: sys.stdout)
    _last_progress_at: float = 0.0
    _last_fraction: float = -1.0
    _last_stage_index: int = -1
    _terminated_jobs: set[str] = field(default_factory=set)

    def send(self, message_type: WorkerMessage, job_id: str | None = None, **payload: Any) -> None:
        """Writes one message."""
        envelope: dict[str, Any] = {
            "v": PROTOCOL_VERSION,
            "type": message_type.value,
            "id": new_id(),
            "jobId": job_id,
        }
        envelope.update(payload)

        self.stream.write(json.dumps(envelope, ensure_ascii=False) + "\n")
        self.stream.flush()

    def ready(self, worker_version: str, capabilities: dict[str, Any]) -> None:
        """Handshake reply."""
        self.send(
            WorkerMessage.READY,
            workerVersion=worker_version,
            protocolVersion=PROTOCOL_VERSION,
            capabilities=capabilities,
        )

    def log(self, level: str, message: str, code: ErrorCode | None = None, **context: Any) -> None:
        """A structured log line. Never carries pixel data or a full source path (§2.3 C-6)."""
        self.send(
            WorkerMessage.LOG,
            level=level,
            code=code.code if code else None,
            message=message,
            context=context,
        )

    def warn(self, code: ErrorCode, message: str, **context: Any) -> None:
        """A numbered warning. Warnings never fail a job (§10.1)."""
        self.log("warning", message, code, **context)

    def progress(
        self,
        job_id: str,
        stage: Stage,
        fraction: float,
        *,
        pts: int | None = None,
        fps: float | None = None,
        eta_seconds: float | None = None,
        force: bool = False,
    ) -> bool:
        """Emits progress if the §8.4 rules allow it. Returns whether anything was written.

        Dropped rather than queued: progress is advisory, and a backlog of stale fractions arriving
        late is exactly the failure the ordering rules exist to prevent.
        """
        if job_id in self._terminated_jobs:
            return False

        stage_index = STAGE_ORDER.index(stage)
        if stage_index < self._last_stage_index:
            return False
        if stage_index == self._last_stage_index and fraction < self._last_fraction:
            return False

        now = time.monotonic()
        if not force and (now - self._last_progress_at) < (1.0 / MAX_PROGRESS_PER_SECOND):
            return False

        self._last_progress_at = now
        self._last_fraction = fraction
        self._last_stage_index = stage_index

        self.send(
            WorkerMessage.PROGRESS,
            job_id,
            stage=stage.value,
            fraction=round(fraction, 4),
            pts=pts,
            fps=fps,
            eta=eta_seconds,
        )
        return True

    def result(self, job_id: str, status: str, summary: dict[str, Any], error: Any = None) -> None:
        """The terminal message for a job. Nothing further is emitted for it."""
        self.send(WorkerMessage.RESULT, job_id, status=status, summary=summary, error=error)
        self._terminated_jobs.add(job_id)
        self._last_fraction = -1.0
        self._last_stage_index = -1

    def error(self, job_id: str | None, error: WorkerError) -> None:
        """A failure, always with a numbered code (§10.1)."""
        self.send(
            WorkerMessage.ERROR,
            job_id,
            code=error.code.code,
            recoverable=error.recoverable,
            message=str(error),
            context=error.context,
        )

    def checkpoint(self, job_id: str, last_completed_pts: int, path: str) -> None:
        """Records that a checkpoint was written (§9)."""
        self.send(WorkerMessage.CHECKPOINT, job_id, lastCompletedPts=last_completed_pts, path=path)

    def is_terminated(self, job_id: str) -> bool:
        """Whether a terminal result has already been sent for this job."""
        return job_id in self._terminated_jobs
