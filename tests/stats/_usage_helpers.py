"""Helpers shared by the usage_store test modules (test_usage_store_jobs,
test_usage_overview, test_usage_tail).

Import as ``from tests.stats._usage_helpers import ...``. The day math is
spelled out here rather than borrowed from usage_store._epoch_day, so a
change to the wire form of `day` still fails the suites.
"""

import datetime

_EPOCH = datetime.date(1970, 1, 1)


def epoch_day(iso: str) -> int:
    """days-since-epoch of an ISO date — the wire form of `day`."""
    return (datetime.date.fromisoformat(iso) - _EPOCH).days
