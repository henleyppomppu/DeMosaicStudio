"""Protocol invariants. prd.md §8."""

from __future__ import annotations

import pytest

import demosaic_worker
from demosaic_worker import protocol
from demosaic_worker.protocol import STAGE_ORDER, Stage, Version


def test_the_package_reexports_the_version_rather_than_restating_it() -> None:
    """prd.md §4.3 — exactly one definition.

    ``demosaic_worker.PROTOCOL_VERSION`` must be the *same object* as the one in ``protocol``, not
    a copy that happens to be equal today. A duplicated constant drifts, and it drifts silently.
    """
    assert demosaic_worker.PROTOCOL_VERSION is protocol.PROTOCOL_VERSION


def test_the_current_version_parses_from_the_constant() -> None:
    assert str(protocol.CURRENT) == protocol.PROTOCOL_VERSION


@pytest.mark.parametrize(
    ("other", "expected"),
    [("1.0", True), ("1.4", True), ("1.99", True), ("2.0", False), ("0.9", False)],
)
def test_compatibility_is_decided_by_the_major_version(other: str, expected: bool) -> None:
    """prd.md §8.1: differing major is refused (E7001); differing minor is accepted."""
    assert protocol.CURRENT.is_compatible_with(Version.parse(other)) is expected


@pytest.mark.parametrize("text", ["", "1", "1.0.0", "v1.0", "-1.0", "1.x"])
def test_malformed_versions_are_rejected(text: str) -> None:
    with pytest.raises(ValueError):
        Version.parse(text)


def test_stage_order_covers_every_stage_exactly_once() -> None:
    """The forward-only progress rule (§8.4) needs a total order over the stages."""
    assert set(STAGE_ORDER) == set(Stage)
    assert len(STAGE_ORDER) == len(Stage)


def test_stage_order_is_the_documented_pipeline_order() -> None:
    assert STAGE_ORDER == (
        Stage.PROBING,
        Stage.ANALYZING,
        Stage.RESTORING,
        Stage.ENCODING,
        Stage.MUXING,
        Stage.FINALIZING,
    )
