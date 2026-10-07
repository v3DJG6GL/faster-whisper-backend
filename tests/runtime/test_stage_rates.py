"""stage_rates: the learned-throughput ledger behind the run plan.

Runs against a real temp file — the locked atomic write is the part most
likely to break. The path is resolved at call time (no default-arg trap),
so repointing PATH is the whole fixture.
"""
import json

import pytest

from faster_whisper_backend.runtime import stage_rates


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    p = str(tmp_path / "stage_rates.json")
    monkeypatch.setattr(stage_rates, "PATH", p, raising=False)
    stage_rates._reset_for_tests()
    yield p
    stage_rates._reset_for_tests()


def test_seed_until_first_measurement(ledger):
    rec = stage_rates.lookup("transcribing", "large-v3", "cuda", "float16")
    assert rec == {"rate": stage_rates.SEEDS["transcribing"], "src": "seed",
                   "n": 0}
    stage_rates.record("transcribing", "large-v3", "cuda", "float16", 9.0)
    rec = stage_rates.lookup("transcribing", "large-v3", "cuda", "float16")
    # The first sample REPLACES the seed outright.
    assert rec == {"rate": 9.0, "src": "measured", "n": 1}


def test_ewma_and_restart_roundtrip(ledger):
    stage_rates.record("diarizing", "pyannote/x", "cuda", None, 10.0)
    stage_rates.record("diarizing", "pyannote/x", "cuda", None, 20.0)
    rec = stage_rates.lookup("diarizing", "pyannote/x", "cuda")
    assert rec["rate"] == pytest.approx(15.0)
    assert rec["n"] == 2
    # Fresh process: the module cache is empty, the file is not.
    stage_rates._reset_for_tests()
    assert stage_rates.lookup("diarizing", "pyannote/x", "cuda")["rate"] == \
        pytest.approx(15.0)
    doc = json.load(open(ledger, encoding="utf-8"))
    assert doc["version"] == stage_rates.SCHEMA_VERSION
    assert "diarizing|pyannote/x|cuda|" in doc["rates"]


def test_mode_is_part_of_the_translating_key(ledger):
    stage_rates.record("translating", "org/m:Q4", "cuda", "fluent", 2.0)
    assert stage_rates.lookup("translating", "org/m:Q4", "cuda",
                              "faithful")["src"] == "seed"
    assert stage_rates.lookup("translating", "org/m:Q4", "cuda",
                              "fluent")["rate"] == 2.0


def test_bad_samples_are_dropped(ledger):
    for bad in (0, -1, float("nan"), float("inf"), "x", None):
        stage_rates.record("separating", "uvr", "cuda", None, bad)
    assert stage_rates.lookup("separating", "uvr", "cuda")["src"] == "seed"


def test_unreadable_file_degrades_to_seeds(ledger):
    with open(ledger, "w", encoding="utf-8") as f:
        f.write("{not json")
    assert stage_rates.lookup("separating", "uvr", "cuda")["src"] == "seed"
    # ...and a later write recovers the file.
    stage_rates.record("separating", "uvr", "cuda", None, 8.5)
    assert stage_rates.lookup("separating", "uvr", "cuda")["rate"] == 8.5


def test_a_hand_edited_count_never_raises(ledger):
    with open(ledger, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "rates": {
            "transcribing|||": {"rate": 6.0, "n": "many"},
            "separating|||": {"rate": 4.0, "n": {}},
            "downloading|||": {"rate": 5.0, "n": float("nan")},
        }}, f)
    stage_rates._reset_for_tests()
    assert stage_rates.lookup("transcribing", None, None) == \
        {"rate": 6.0, "src": "measured", "n": 0}
    assert stage_rates.lookup("separating", None, None)["n"] == 0
    assert stage_rates.lookup("downloading", None, None)["n"] == 0
    # ...and the fold on top of such a row counts from there.
    stage_rates.record("transcribing", None, None, None, 8.0)
    assert stage_rates.lookup("transcribing", None, None) == \
        {"rate": 7.0, "src": "measured", "n": 1}


@pytest.mark.parametrize("body", [
    '{"version": 1, "rates": {"transcribing|||": {"rate": 1%s}}}' % ("0" * 400),
    '{"version": 1, "rates": {"transcribing|||": {"rate": 6.0, "n": 1%s}}}'
    % ("0" * 400),
    "[" * 100000 + "]" * 100000,
])
def test_an_overflowing_or_deeply_nested_file_degrades_to_seeds(ledger, body):
    with open(ledger, "w", encoding="utf-8") as f:
        f.write(body)
    stage_rates._reset_for_tests()
    assert stage_rates.lookup("transcribing", None, None)["src"] == "seed"


def test_record_many_is_one_write_folded_in_order(ledger, monkeypatch):
    from faster_whisper_backend.core import atomic_json
    writes = []
    real = atomic_json.atomic_write_json

    def counting(*a, **kw):
        writes.append(1)
        return real(*a, **kw)
    monkeypatch.setattr(atomic_json, "atomic_write_json", counting)
    stage_rates.record_many([
        ("translating", "m", "cuda", "fluent", 2.0),
        ("separating", "uvr", "cuda", None, float("nan")),   # dropped
        ("translating", "m", "cuda", "fluent", 8.0),
        ("diarizing.embeddings", "p", "cuda", None, 20.0),
        ("bad",),                                            # dropped
    ])
    assert len(writes) == 1
    rec = stage_rates.lookup("translating", "m", "cuda", "fluent")
    assert rec == {"rate": 5.0, "src": "measured", "n": 2}
    assert stage_rates.lookup("diarizing.embeddings", "p", "cuda")["n"] == 1
    assert stage_rates.lookup("separating", "uvr", "cuda")["src"] == "seed"
    # Nothing valid: no write at all.
    stage_rates.record_many([("separating", "uvr", "cuda", None, 0)])
    stage_rates.record_many([])
    assert len(writes) == 1


def test_unknown_stage_has_no_rate(ledger):
    assert stage_rates.lookup("mystery", None, None)["rate"] is None
