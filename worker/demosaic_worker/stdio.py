"""stdio configuration for the JSON Lines protocol. prd.md §8.1.

The protocol says UTF-8. Python does not agree by default: on this machine
``sys.stdout.encoding`` is **cp949**, because that is the console code page for a Korean Windows
locale. Left alone, the first log line or file path containing a non-ASCII character either
mojibakes or raises ``UnicodeEncodeError`` in the middle of a job — and it does so only on
locale-non-ASCII machines, which is precisely the class of bug that passes CI and breaks on the
user's PC.

So the worker configures its own streams rather than inheriting whatever the console offers.

* stdout is the protocol channel: UTF-8, newline-terminated, line-buffered, and **strict** — a
  character that cannot be encoded is a bug worth failing on, not one worth hiding.
* stderr is the human log channel: UTF-8, but with ``errors="replace"``, because losing a
  diagnostic message to an encoding error while diagnosing something else is a poor trade.
"""

from __future__ import annotations

import sys
from typing import IO, Any

#: The protocol's wire encoding. prd.md §8.1.
PROTOCOL_ENCODING = "utf-8"

#: JSON Lines: one object per line, newline-terminated, no translation.
PROTOCOL_NEWLINE = "\n"


def configure_stdio_utf8(
    stdout: IO[Any] | None = None,
    stderr: IO[Any] | None = None,
) -> None:
    """Forces UTF-8 on the worker's standard streams.

    Call this before writing anything, and before the handshake in particular. Idempotent, and a
    no-op on a stream that cannot be reconfigured (a pipe replaced by a test double, say).
    """
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr

    _reconfigure(out, errors="strict")
    _reconfigure(err, errors="replace")


def _reconfigure(stream: IO[Any], *, errors: str) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return

    reconfigure(
        encoding=PROTOCOL_ENCODING,
        errors=errors,
        newline=PROTOCOL_NEWLINE,
        line_buffering=True,
    )


def is_utf8(stream: IO[Any]) -> bool:
    """True when a stream will emit UTF-8. Used by the guard test and by the handshake."""
    encoding = getattr(stream, "encoding", None)
    if encoding is None:
        return False

    return encoding.lower().replace("-", "").replace("_", "") == "utf8"
