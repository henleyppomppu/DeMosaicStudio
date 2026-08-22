"""Worker entry point and dispatch loop. prd.md §8.5.

Reads JSON Lines from stdin, dispatches, writes JSON Lines to stdout. Everything of substance lives
in the handlers; this file's job is the lifecycle, and the lifecycle rules are the ones that are
easy to get subtly wrong:

* **One job at a time** (§8.5.2). A second ``process`` while one is running is refused with E7003
  rather than queued, because a queue here would duplicate the host's own scheduling and the two
  would disagree.
* **Cancel is cooperative** (§8.5.3). The host asks; the worker drains, checkpoints, and emits a
  terminal ``result`` with ``status="cancelled"``. Nothing is killed mid-write.
* **Every failure carries a numbered code** (§10.1). An unexpected exception becomes E9001 rather
  than a traceback on stdout — a traceback on stdout would corrupt the protocol stream on top of
  whatever went wrong.
* **stdout is configured to UTF-8 before the handshake** (§8.1). On this machine the console is
  cp949, and the first non-ASCII path would otherwise raise inside the protocol writer.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass, field
from typing import IO, Any

from . import __version__
from .errors import E7003, E7006, E9001, WorkerError
from .jobs import JobContext, JobRunner
from .messages import Emitter, Request, parse_request
from .protocol import PROTOCOL_VERSION, HostMessage, WorkerMessage
from .stdio import configure_stdio_utf8


@dataclass
class Worker:
    """The dispatch loop and its one piece of state: the job that is currently running."""

    emitter: Emitter
    runner: JobRunner
    current: JobContext | None = None
    _running: bool = True
    _capabilities: dict[str, Any] = field(default_factory=dict)

    def capabilities(self) -> dict[str, Any]:
        """What this build can do, reported at handshake.

        Availability means "we loaded it and it worked", never "a driver reported a device"
        (§8.3) — so this probes rather than guesses, and caches the answer.
        """
        if not self._capabilities:
            self._capabilities = self.runner.capabilities()
        return self._capabilities

    def handle(self, request: Request) -> None:
        """Dispatches one request."""
        match request.type:
            case HostMessage.HELLO:
                self.emitter.ready(__version__, self.capabilities())

            case HostMessage.PROBE:
                media, hardware = self.runner.probe(request.require("sourcePath"))
                self.emitter.send(
                    WorkerMessage.PROBE_RESULT, request.job_id, media=media, hardware=hardware
                )

            case HostMessage.ANALYZE:
                self._start(request, analyze_only=True)

            case HostMessage.PROCESS:
                self._start(request, analyze_only=False)

            case HostMessage.PREVIEW:
                result = self.runner.preview(
                    request.job_id or "",
                    request.require("sourcePath"),
                    int(request.require("pts")),
                    request.get("settings", {}),
                    overlay=bool(request.get("overlay", False)),
                )
                self.emitter.send(WorkerMessage.PREVIEW_RESULT, request.job_id, **result)

            case HostMessage.PAUSE:
                if self.current:
                    self.current.paused = True

            case HostMessage.RESUME:
                if self.current:
                    self.current.paused = False

            case HostMessage.CANCEL:
                if self.current:
                    self.current.cancelled = True
                    self.emitter.log("info", "cancel acknowledged", job=self.current.job_id)

            case HostMessage.SHUTDOWN:
                if self.current:
                    self.current.cancelled = True
                self._running = False

    def _start(self, request: Request, *, analyze_only: bool) -> None:
        # jobId lives in the envelope, not the payload: parse_request lifts it out. Reaching for it
        # with require() looks right and finds nothing.
        job_id = request.job_id
        if not job_id:
            raise WorkerError(E7006, f"{request.type.value} requires a jobId")

        if self.current is not None and not self.current.finished:
            raise WorkerError(E7003, "a job is already running", requested=job_id)

        context = JobContext(
            job_id=job_id,
            source_path=str(request.require("sourcePath")),
            output_path=str(request.get("outputPath") or ""),
            settings=dict(request.get("settings") or {}),
            resume=bool(request.get("resume", False)),
            comparison_pts=[int(p) for p in (request.get("comparisonPts") or [])],
            analyze_only=analyze_only,
            sample_every=max(1, int(request.get("sampleEvery") or 1)),
        )
        self.current = context

        try:
            summary = self.runner.run(context, self.emitter)
            status = "cancelled" if context.cancelled else "completed"
            self.emitter.result(job_id, status, summary)
        except WorkerError as error:
            self.emitter.error(job_id, error)
            self.emitter.result(job_id, "failed", context.summary(), error=error.code.code)
        finally:
            context.finished = True

    def run(self, stream: IO[str] | None = None) -> int:
        """Reads and dispatches until stdin closes or ``shutdown`` arrives."""
        source = stream if stream is not None else sys.stdin

        for line in source:
            line = line.strip()
            if not line:
                continue

            try:
                request = parse_request(line)
            except WorkerError as error:
                self.emitter.error(None, error)
                continue

            try:
                self.handle(request)
            except WorkerError as error:
                self.emitter.error(request.job_id, error)
            except Exception as exc:  # noqa: BLE001 - the loop must survive anything
                # A traceback on stdout would corrupt the protocol stream on top of whatever went
                # wrong, so it goes to stderr and the host sees a numbered code.
                traceback.print_exc(file=sys.stderr)
                self.emitter.error(request.job_id, WorkerError(E9001, str(exc)))

            if not self._running:
                break

        return 0


def main(argv: list[str] | None = None) -> int:
    """Process entry point."""
    configure_stdio_utf8()

    emitter = Emitter()
    worker = Worker(emitter=emitter, runner=JobRunner())

    print(
        f"demosaic_worker {__version__}, protocol {PROTOCOL_VERSION}",
        file=sys.stderr,
        flush=True,
    )

    return worker.run()


if __name__ == "__main__":
    sys.exit(main())
