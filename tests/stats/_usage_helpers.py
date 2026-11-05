"""Helpers shared by the usage_store test modules (test_usage_store_jobs,
test_usage_overview, test_usage_tail, test_usage_sweep_retention).

Import as ``from tests.stats._usage_helpers import ...``. The day math is
spelled out here rather than borrowed from usage_store._epoch_day, so a
change to the wire form of `day` still fails the suites.
"""

import datetime
import zoneinfo

_EPOCH = datetime.date(1970, 1, 1)
_UTC = zoneinfo.ZoneInfo("UTC")


def epoch_day(iso: str) -> int:
    """days-since-epoch of an ISO date — the wire form of `day`."""
    return (datetime.date.fromisoformat(iso) - _EPOCH).days


def rec(us, *, key_id="k", user_id="u", status="ok", **kw):
    """record_usage for the suites' default caller and a good outcome."""
    us.record_usage(key_id=key_id, user_id=user_id, status=status, **kw)


def utc_doc(us, user_id="u", **kw):
    """document() for a UTC caller; user_id=None is the whole server."""
    return us.document(user_id, tz=_UTC, tz_name="UTC", **kw)
