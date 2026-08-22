"""Numbered error codes. prd.md §10.

This is one of two implementations; the other is ``DeMosaicStudio.Domain.Diagnostics.ErrorCodes``.
They are locked together by ``fixtures/parity/error_codes.json`` (§13.4). Adding a code means
updating both sides, the fixture, ``docs/ERROR_CODES.md`` and ``docs/TROUBLESHOOTING.md`` — the
parity test fails otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final


class Severity(str, Enum):
    """Whether an entry fails a job or merely annotates it. prd.md §10.1."""

    ERROR = "Error"
    WARNING = "Warning"


@dataclass(frozen=True, slots=True)
class ErrorCode:
    """One numbered entry from prd.md §10.2."""

    code: str
    meaning: str
    recoverable: bool
    severity: Severity

    @property
    def is_warning(self) -> bool:
        """True when this entry annotates rather than fails."""
        return self.severity is Severity.WARNING


def _error(code: str, meaning: str, *, recoverable: bool) -> ErrorCode:
    return ErrorCode(code, meaning, recoverable, Severity.ERROR)


def _warning(code: str, meaning: str) -> ErrorCode:
    # Warnings are never failures, so 'recoverable' does not apply and is fixed at False (§10.1).
    return ErrorCode(code, meaning, False, Severity.WARNING)


# E1xxx — media / input
E1001 = _error("E1001", "File not found or unreadable", recoverable=False)
E1002 = _error("E1002", "Unsupported container", recoverable=False)
E1003 = _error("E1003", "Unsupported video codec or profile", recoverable=False)
E1004 = _error("E1004", "Corrupt source: demux failure", recoverable=False)
E1005 = _error("E1005", "Source has no video stream", recoverable=False)
E1006 = _error("E1006", "Source metadata inconsistent", recoverable=True)

# E2xxx — decode
E2001 = _error("E2001", "Hardware decoder init failed", recoverable=True)
E2002 = _error("E2002", "Decode error mid-stream, frame unrecoverable", recoverable=True)
E2003 = _error("E2003", "Decode error mid-stream, stream unrecoverable", recoverable=False)
E2004 = _error("E2004", "Timestamp discontinuity beyond tolerance", recoverable=True)

# E3xxx — detection / tracking
E3001 = _error("E3001", "Detector model load failed", recoverable=False)
E3002 = _error("E3002", "Detector inference failure", recoverable=True)
E3003 = _error("E3003", "Detector output shape mismatch", recoverable=False)
E3201 = _error("E3201", "Track state-machine violation", recoverable=False)

# E4xxx — restoration
E4001 = _error("E4001", "Restoration model load failed", recoverable=False)
E4002 = _error("E4002", "Restoration inference failure", recoverable=True)
E4003 = _error("E4003", "Alignment failure for the whole window", recoverable=True)
E4004 = _error("E4004", "ROI smaller than model minimum", recoverable=True)
E4401 = _error("E4401", "GPU OOM, mitigation ladder exhausted", recoverable=False)
E4402 = _error("E4402", "Backend/runtime unsupported for this model", recoverable=False)

# E5xxx — encode / mux
E5001 = _error("E5001", "Encoder init failed", recoverable=True)
E5002 = _error("E5002", "Encode failure mid-stream", recoverable=False)
E5003 = _error("E5003", "Mux failure", recoverable=False)
E5004 = _error("E5004", "Output container cannot carry a source stream", recoverable=True)

# E6xxx — system
E6001 = _error("E6001", "Disk full", recoverable=True)
E6002 = _error("E6002", "Output path not writable", recoverable=True)
E6003 = _error("E6003", "Output file locked by another process", recoverable=True)
E6004 = _error("E6004", "Insufficient system RAM", recoverable=False)
E6005 = _error("E6005", "Required support library missing or unloadable", recoverable=False)

# E7xxx — protocol / process
E7001 = _error("E7001", "Protocol major version mismatch", recoverable=False)
E7002 = _error("E7002", "Worker handshake timeout", recoverable=True)
E7003 = _error("E7003", "Worker busy: a job is already running", recoverable=False)
E7004 = _error("E7004", "Worker did not exit within the cancel grace period", recoverable=True)
E7005 = _error("E7005", "Worker crashed", recoverable=True)
E7006 = _error("E7006", "Malformed protocol message", recoverable=False)

# E9xxx
E9001 = _error("E9001", "Unexpected internal error", recoverable=False)

# Warnings
W1101 = _warning("W1101", "Fell back to software decode")
W3101 = _warning("W3101", "Region count clamped to max_regions_per_frame")
W4101 = _warning("W4101", "OOM ladder step applied")
W4102 = _warning("W4102", "Region left untouched: confidence below minRestorationConfidence")
W4103 = _warning("W4103", "Requested temporalWindow reduced by a safety rule")
W5101 = _warning("W5101", "Stream dropped for container compatibility")
W5102 = _warning("W5102", "Stream copy unavailable; output was re-encoded")
W6101 = _warning("W6101", "Backend substituted")


ALL: Final[tuple[ErrorCode, ...]] = (
    E1001, E1002, E1003, E1004, E1005, E1006,
    E2001, E2002, E2003, E2004,
    E3001, E3002, E3003, E3201,
    E4001, E4002, E4003, E4004, E4401, E4402,
    E5001, E5002, E5003, E5004,
    E6001, E6002, E6003, E6004, E6005,
    E7001, E7002, E7003, E7004, E7005, E7006,
    E9001,
    W1101, W3101, W4101, W4102, W4103, W5101, W5102, W6101,
)

_BY_CODE: Final[dict[str, ErrorCode]] = {c.code: c for c in ALL}


def get(code: str) -> ErrorCode:
    """Looks up an entry, raising when it is not in the table."""
    try:
        return _BY_CODE[code]
    except KeyError as exc:
        raise KeyError(f"{code!r} is not in the error table (prd.md §10.2)") from exc


def try_get(code: str) -> ErrorCode | None:
    """Looks up an entry, returning ``None`` when it is not in the table."""
    return _BY_CODE.get(code)


class WorkerError(Exception):
    """An error carrying a numbered code. A free-text-only failure is a defect (§10.1)."""

    def __init__(self, code: ErrorCode, message: str = "", **context: object) -> None:
        super().__init__(message or code.meaning)
        self.code = code
        self.context = context

    @property
    def recoverable(self) -> bool:
        """Whether the host may auto-retry once (§10.3)."""
        return self.code.recoverable
