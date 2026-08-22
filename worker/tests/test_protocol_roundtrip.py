"""A real host/worker conversation, over a real subprocess. AGENTS.md protocol procedure, step 4.

Every other test in this suite imports the worker. This one launches it the way the host will -
stdio, JSON Lines, one message per line - because that boundary has broken twice in ways an
in-process test could not see: `jobId` was lifted into the envelope but read from the payload
(every `process` refused with E7006), and stdout defaulted to cp949 so one non-ASCII log line
killed the stream.

Model weights are gitignored, so a job cannot run on a fresh checkout. The handshake and probe can,
and they are what the fixture covers.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from demosaic_worker.protocol import PROTOCOL_VERSION


def _repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "fixtures").is_dir():
            return candidate
    raise AssertionError("no 'fixtures' directory above this test file")


REPO = _repository_root()
FIXTURE = REPO / "fixtures" / "protocol" / "handshake.json"


def _conversation() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_the_fixture_declares_the_version_this_build_speaks() -> None:
    """The fixture is a document as much as a test input; a stale one is worse than none."""
    assert _conversation()["protocolVersion"] == PROTOCOL_VERSION


def test_a_real_worker_process_answers_the_whole_exchange() -> None:
    conversation = _conversation()
    requests = [step["send"] for step in conversation["exchange"]]

    completed = subprocess.run(
        [sys.executable, "-m", "demosaic_worker.main_loop"],
        input="\n".join(json.dumps(request) for request in requests) + "\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
        cwd=str(REPO),
        env={"PYTHONPATH": str(REPO / "worker"), "SYSTEMROOT": r"C:\Windows",
             "PATH": r"C:\Windows\System32"},
    )

    assert completed.returncode == 0, completed.stderr[-800:]

    replies = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    by_type = {reply["type"]: reply for reply in replies}

    for step in conversation["exchange"]:
        expected = step["expect"]
        if expected is None:
            continue

        reply = by_type.get(expected["type"])
        assert reply is not None, (
            f"no {expected['type']} in reply to {step['send']['type']}; got {sorted(by_type)}"
        )
        for field in expected["fields"]:
            assert field in reply, f"{expected['type']} is missing {field}"

    assert by_type["ready"]["protocolVersion"] == PROTOCOL_VERSION


def test_every_reply_carries_the_envelope() -> None:
    """`v`, `type` and `id` on every message. The host parses before it dispatches."""
    conversation = _conversation()
    requests = [step["send"] for step in conversation["exchange"]]

    completed = subprocess.run(
        [sys.executable, "-m", "demosaic_worker.main_loop"],
        input="\n".join(json.dumps(request) for request in requests) + "\n",
        capture_output=True, text=True, encoding="utf-8", timeout=180, cwd=str(REPO),
        env={"PYTHONPATH": str(REPO / "worker"), "SYSTEMROOT": r"C:\Windows",
             "PATH": r"C:\Windows\System32"},
    )

    replies = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    assert replies, "the worker said nothing at all"

    for reply in replies:
        for field in ("v", "type", "id"):
            assert field in reply, f"{reply.get('type')} has no {field}"
        assert reply["v"] == PROTOCOL_VERSION


@pytest.mark.parametrize("malformed", ['{"v": "1.1"}', "not json at all", "{}"])
def test_a_malformed_message_does_not_take_the_worker_down(malformed: str) -> None:
    """A host bug must not look like a worker crash: it has to come back as a numbered error."""
    completed = subprocess.run(
        [sys.executable, "-m", "demosaic_worker.main_loop"],
        input=malformed + '\n{"v": "1.1", "type": "shutdown", "id": "z"}\n',
        capture_output=True, text=True, encoding="utf-8", timeout=180, cwd=str(REPO),
        env={"PYTHONPATH": str(REPO / "worker"), "SYSTEMROOT": r"C:\Windows",
             "PATH": r"C:\Windows\System32"},
    )

    assert completed.returncode == 0, f"the worker died on a malformed line: {completed.stderr[-400:]}"

    replies = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    assert any(reply["type"] == "error" for reply in replies), (
        f"no error reported for {malformed!r}; got {[r['type'] for r in replies]}"
    )
