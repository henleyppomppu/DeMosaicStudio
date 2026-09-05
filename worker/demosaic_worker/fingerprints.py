"""Per-artifact settings fingerprints. prd.md §9.3.

Mirror of ``DeMosaicStudio.Domain.Jobs.SettingsFingerprint``, locked by
``fixtures/parity/fingerprints.json`` (§13.4). The two must agree byte-for-byte: a resume computed
by one side and checked by the other would otherwise discard work that is perfectly valid.

The canonical form is deliberately not JSON. A newline-joined, ordinally sorted list of
``key=value`` lines is trivial to reproduce identically in two languages; a JSON serializer's
key order, spacing and float formatting are not.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from enum import Enum
from typing import Any, Final


class Scope(str, Enum):
    """Which artifact a fingerprint governs. prd.md §9.3."""

    DETECTION = "detection"
    RESTORATION = "restoration"
    ENCODE = "encode"


#: Keys that make up each fingerprint, and how to render their values.
#:
#: Anything absent here is excluded on purpose: performance knobs (the OOM ladder changes precision
#: mid-run) and diagnostic knobs (comparison points change no output pixel). See §9.3.
_FIELDS: Final[dict[Scope, tuple[tuple[str, str], ...]]] = {
    Scope.DETECTION: (
        ("confidence", "num"),
        # How often the detector runs. In the fingerprint because it changes which frames have
        # detections and therefore the output; adding it invalidated every cached detection once,
        # which is the price of a fingerprinted key and was paid knowingly (D-43).
        ("detectEvery", "int"),
        ("maskThreshold", "num"),
        ("maxMissingFrames", "int"),
        ("minConfirmFrames", "int"),
        ("minRegionArea", "int"),
        ("nmsIou", "num"),
    ),
    Scope.RESTORATION: (
        ("alignConfMin", "num"),
        ("featherWidth", "int"),
        ("minRestorationConfidence", "num"),
        ("paddingRatio", "num"),
        ("preset", "str"),
        # The blend weight of the single-frame path's temporal smoother (D-43).
        ("temporalAlpha", "num"),
        # The *requested* value, never the per-frame resolved K (§5.6.1, §9.3).
        ("temporalWindow", "str"),
    ),
    Scope.ENCODE: (
        ("codec", "str"),
        ("constantQuality", "int"),
        ("profile", "str"),
    ),
}


def _render(value: Any, kind: str) -> str:
    if kind == "num":
        # Four decimals so C# "0.0000" and Python f"{v:.4f}" agree without either language's
        # shortest-round-trip formatting getting involved.
        return f"{float(value):.4f}"
    if kind == "int":
        return str(int(value))
    return str(value)


def canonicalize(settings: Mapping[str, Any], scope: Scope) -> str:
    """Builds the canonical text that :func:`compute` hashes.

    Exposed so a parity mismatch can report *what* differs rather than only that two hashes do.
    """
    section = settings.get(scope.value, {})
    lines = []

    for key, kind in _FIELDS[scope]:
        if key not in section:
            raise KeyError(f"settings[{scope.value!r}] is missing {key!r} (prd.md §9.3)")
        lines.append(f"{key}={_render(section[key], kind)}")

    # Ordinal sort, matching StringComparer.Ordinal on the host side.
    lines.sort()
    return "\n".join(lines)


def compute(settings: Mapping[str, Any], scope: Scope) -> str:
    """Computes the fingerprint for one artifact scope."""
    digest = hashlib.sha256(canonicalize(settings, scope).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def changed(recorded: str | None, current: str | None) -> bool:
    """True when a recorded fingerprint does not match the current one.

    An unknown fingerprint compares as **changed**, never as equal. A null-lifting comparison that
    evaluates false on unknown data silently reuses a previous file's artifacts for a different
    source, which is data corruption rather than a UX bug (§9.3).
    """
    if recorded is None or current is None:
        return True
    return recorded != current


def invalidated(
    recorded: Mapping[str, str | None],
    current: Mapping[str, str | None],
) -> set[str]:
    """Artifacts to discard on resume, cascading top-down from the first changed stage (§9.3)."""
    discard: set[str] = set()

    if changed(recorded.get("detection"), current.get("detection")):
        discard.update({"analysis", "video"})

    if changed(recorded.get("restoration"), current.get("restoration")) or changed(
        recorded.get("encode"), current.get("encode")
    ):
        discard.add("video")

    return discard
