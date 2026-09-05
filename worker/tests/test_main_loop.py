"""The dispatch loop's lifecycle rules. prd.md section 8.5.

The rule under test is the one the module's own docstring has always claimed and never had:
**cancel is answered while the job is running.** A worker that reads its next line only between
jobs cannot honour a cancel at all - the line waits in the pipe until the work it was meant to
interrupt has finished, and every `context.cancelled` check downstream is unreachable. That is
what shipped, and pressing Stop in the window did nothing.

The tests here therefore drive the loop with a stdin that produces the cancel *after* the job is
under way, which is what a person clicking a button does and what a StringIO holding both lines
cannot reproduce.
"""

from __future__ import annotations

import io
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import pytest

from demosaic_worker.jobs import JobContext
from demosaic_worker.main_loop import Worker
from demosaic_worker.messages import Emitter
from demosaic_worker.protocol import PROTOCOL_VERSION

#: Long enough that a loop which cannot see the cancel fails instead of hanging the suite.
PATIENCE = 10.0


def _line(message_type: str, **payload: Any) -> str:
    return json.dumps({"v": PROTOCOL_VERSION, "type": message_type, "id": "1", **payload})


@dataclass
class SlowRunner:
    """A job that runs until it is cancelled, and reports whether it ever was."""

    started: threading.Event = field(default_factory=threading.Event)
    observed_cancel: bool = False

    def capabilities(self) -> dict[str, Any]:
        return {"cudaAvailable": False, "device": "cpu", "models": []}

    def probe(self, source_path: str) -> tuple[dict, dict]:
        return {}, {}

    def run(self, context: JobContext, emitter: Emitter) -> dict[str, Any]:
        self.started.set()
        deadline = time.monotonic() + PATIENCE
        while not context.cancelled and time.monotonic() < deadline:
            time.sleep(0.005)

        self.observed_cancel = context.cancelled
        return {"framesSeen": 0}


class Stdin:
    """A stdin that releases its later lines only once a gate opens.

    Iterating a StringIO hands the reader every line at once, so a worker that queued the cancel
    behind the job would still appear to work: by the time the job ended the flag would be set.
    This makes the ordering real.
    """

    def __init__(self, before: list[str], gate: threading.Event, after: list[str]) -> None:
        self._before, self._gate, self._after = before, gate, after

    def __iter__(self) -> Iterator[str]:
        yield from (line + "\n" for line in self._before)
        if not self._gate.wait(PATIENCE):
            raise AssertionError("the job never started")
        yield from (line + "\n" for line in self._after)


def _messages(buffer: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in buffer.getvalue().splitlines() if line.strip()]


def test_a_cancel_reaches_the_job_it_is_cancelling_rather_than_waiting_for_it() -> None:
    """The defect: stdin was read only between jobs, so Stop could not stop anything."""
    runner = SlowRunner()
    buffer = io.StringIO()
    worker = Worker(emitter=Emitter(stream=buffer), runner=runner)

    stdin = Stdin(
        [_line("process", jobId="job-1", sourcePath="a.mp4", outputPath="b.mp4")],
        runner.started,
        [_line("cancel", jobId="job-1"), _line("shutdown")],
    )

    finished = threading.Thread(target=worker.run, args=(stdin,), daemon=True)
    finished.start()
    finished.join(timeout=PATIENCE + 5.0)

    assert not finished.is_alive(), "the loop never returned: the cancel was not read"
    assert runner.observed_cancel, "the job ran to completion despite the cancel"

    results = [m for m in _messages(buffer) if m["type"] == "result"]
    assert [m["status"] for m in results] == ["cancelled"]


def test_a_cancel_that_arrives_before_the_job_starts_is_not_lost() -> None:
    """The reader answers immediately, so it can beat the dispatch thread to `current`."""
    runner = SlowRunner()
    worker = Worker(emitter=Emitter(stream=io.StringIO()), runner=runner)

    # The order a race would produce: the cancel is handled while `current` is still None.
    worker.interrupt(_request("cancel", "job-1"))
    worker._start(_request("process", "job-1", sourcePath="a.mp4", outputPath="b.mp4"),
                  analyze_only=False)

    assert runner.observed_cancel


def test_an_unknown_job_id_does_not_cancel_the_running_one() -> None:
    runner = SlowRunner()
    worker = Worker(emitter=Emitter(stream=io.StringIO()), runner=runner)

    worker.interrupt(_request("cancel", "some-other-job"))
    context = JobContext(job_id="job-1", source_path="a.mp4", output_path="b.mp4")
    worker.current = context

    assert not context.cancelled


def test_the_loop_survives_a_line_that_is_not_json() -> None:
    buffer = io.StringIO()
    worker = Worker(emitter=Emitter(stream=buffer), runner=SlowRunner())
    worker.run(io.StringIO("{ not json\n" + _line("hello", hostVersion="1.0") + "\n"))

    types = [m["type"] for m in _messages(buffer)]
    assert "error" in types and "ready" in types


def _request(message_type: str, job_id: str, **payload: Any):
    from demosaic_worker.messages import parse_request

    return parse_request(_line(message_type, jobId=job_id, **payload))


def test_optional_imports_are_warmed_before_the_loop_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    """The deadlock fix: heavy native libraries load on the main thread before the reader thread
    exists and before any job has native threads. Order is the whole point, so it is what is tested."""
    from demosaic_worker import main_loop

    order: list[str] = []
    monkeypatch.setattr(main_loop, "warm_optional_imports", lambda: order.append("warm"))
    monkeypatch.setattr(main_loop, "configure_stdio_utf8", lambda: order.append("stdio"))
    monkeypatch.setattr(main_loop.Worker, "run", lambda self, stream=None: order.append("run") or 0)
    monkeypatch.setattr(main_loop, "JobRunner", lambda: object())

    assert main_loop.main() == 0
    assert order == ["stdio", "warm", "run"]


def test_a_missing_optional_library_is_skipped_not_fatal() -> None:
    from demosaic_worker.main_loop import warm_optional_imports

    loaded = warm_optional_imports(("json", "no_such_module_anywhere", "os"))
    assert loaded == ["json", "os"]
