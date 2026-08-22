"""stdio encoding guard. prd.md §8.1.

This exists because the development machine's default is cp949, not UTF-8. A test that only ran on
a UTF-8 host would pass while the shipped worker corrupted every non-ASCII path it logged.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

from demosaic_worker.stdio import configure_stdio_utf8, is_utf8


def _repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "fixtures").is_dir():
            return candidate
    raise AssertionError("no 'fixtures' directory above this test file")


def _text_stream(encoding: str) -> io.TextIOWrapper:
    return io.TextIOWrapper(io.BytesIO(), encoding=encoding, newline="\n")


def test_a_cp949_stream_is_switched_to_utf8() -> None:
    """The exact situation on the target machine."""
    stdout = _text_stream("cp949")
    stderr = _text_stream("cp949")

    assert not is_utf8(stdout)

    configure_stdio_utf8(stdout, stderr)

    assert is_utf8(stdout)
    assert is_utf8(stderr)


def test_configuring_twice_is_harmless() -> None:
    stdout = _text_stream("cp949")

    configure_stdio_utf8(stdout, _text_stream("cp949"))
    configure_stdio_utf8(stdout, _text_stream("cp949"))

    assert is_utf8(stdout)


def test_a_stream_that_cannot_be_reconfigured_is_left_alone() -> None:
    """A test double or a raw pipe must not make the worker crash on startup."""

    class Dumb:
        encoding = "cp949"

    configure_stdio_utf8(Dumb(), Dumb())  # must not raise


def test_non_ascii_survives_a_real_subprocess_round_trip() -> None:
    """The end-to-end version: a child process writes Hangul and Japanese to stdout.

    Without ``configure_stdio_utf8`` this raises UnicodeEncodeError on a cp949 host, which is how
    the problem was found in the first place.
    """
    root = _repository_root()
    payload = '{"v":"1.0","type":"log","message":"자막 テスト §5.1"}'

    script = (
        "import sys\n"
        "sys.path.insert(0, r'" + str(root / "worker") + "')\n"
        "from demosaic_worker.stdio import configure_stdio_utf8\n"
        "configure_stdio_utf8()\n"
        "print(" + repr(payload) + ")\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        env={"PATH": "", "SYSTEMROOT": r"C:\Windows"},
    )

    assert result.stdout.decode("utf-8").strip() == payload


def test_stdin_is_utf8_too() -> None:
    """The channel requests arrive on, and the one this module forgot.

    stdout and stderr were configured from the start; stdin was left at whatever the console
    offered. On a Korean-locale machine that is cp949, so a host that correctly wrote a UTF-8 path
    had it mis-decoded and the worker died mid-handshake. A C# round-trip test against a directory
    named in Hangul found it in one run.
    """
    script = (
        "import sys, json\n"
        "sys.path.insert(0, r'{worker}')\n"
        "from demosaic_worker.stdio import configure_stdio_utf8, is_utf8\n"
        "configure_stdio_utf8()\n"
        "line = sys.stdin.readline()\n"
        "print(json.dumps({{'stdin': is_utf8(sys.stdin), 'read': json.loads(line)['path']}}))\n"
    ).format(worker=str(_repository_root() / "worker"))

    path = "D:/영상/클립.mp4"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        input=json.dumps({"path": path}) + "\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr[-500:]
    reply = json.loads(completed.stdout.strip().splitlines()[-1])
    assert reply["stdin"], "stdin was not reconfigured to UTF-8"
    assert reply["read"] == path, f"the path came back as {reply['read']!r}"
