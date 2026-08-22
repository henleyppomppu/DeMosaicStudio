"""Confidence bucket parity. prd.md §5.9.4, §13.4."""

from __future__ import annotations

import pytest

from demosaic_worker.confidence import Bucket, bucket


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.00, Bucket.LOW),
        (0.32, Bucket.LOW),
        (0.33, Bucket.MEDIUM),
        (0.65, Bucket.MEDIUM),
        (0.66, Bucket.HIGH),
        (1.00, Bucket.HIGH),
    ],
)
def test_buckets_use_the_documented_boundaries(value: float, expected: Bucket) -> None:
    assert bucket(value) is expected
