"""POST /settings/test-pipeline gives a rule with `repeat` on exactly the
engine's passes (pl_engine._postprocess_text): parity is the dry run's
contract, so the panel and the per-card status show what a transcription
gets. One step spans every pass (the matches summed); `passes` counts only
the passes that changed the text, not the last one that found nothing."""

import pytest

from faster_whisper_backend.pipeline import engine as pl_engine

_HYPHEN = {"name": "hyphen", "label": "Hyphen", "type": "callback:upper",
           "enabled": True, "pattern": "(\\b[A-ZÄÖÜ]\\w*-)([a-zäöüß])"}
_BATCH = {"name": "batch", "label": "Batch", "type": "regex-list", "enabled": True,
          "entries": [{"pattern": "a", "replacement": "b"},
                      {"pattern": "bb", "replacement": "a"}]}
_GROW = {"name": "grow", "label": "Grow", "type": "regex-list", "enabled": True,
         "entries": [{"pattern": "$", "replacement": "x"}]}


def _dry(client, rules, sample):
    r = client.post("/settings/test-pipeline", json={"sample": sample, "rules": rules})
    assert r.status_code == 200, r.text
    return r.json()


def _engine(app_module, rules, sample):
    app_module.cfg.PIPELINE_RULES = rules
    pl_engine.rebuild_caches()
    return pl_engine._postprocess_text(sample)


@pytest.mark.parametrize("rule, sample, want, passes, matches", [
    (_HYPHEN, "Hals-nasen-ohren-arzt", "Hals-Nasen-Ohren-Arzt", 3, 3),
    (_HYPHEN, "Herz-kreislauf-system", "Herz-Kreislauf-System", 2, 2),
    (_HYPHEN, "schwarz-weiß", "schwarz-weiß", 0, 0),
    (_BATCH, "aaaa", "b", 3, 10),
    # Never settles: cut at the engine's pass cap.
    (_GROW, "a", "a" + "x" * pl_engine._REPEAT_MAX_PASSES, pl_engine._REPEAT_MAX_PASSES,
     pl_engine._REPEAT_MAX_PASSES),
])
def test_a_repeating_rule_dry_runs_as_the_engine_runs_it(
        client, app_module, rule, sample, want, passes, matches):
    rules = [dict(rule, repeat=True)]
    j = _dry(client, rules, sample)
    step = j["steps"][0]
    assert (step["before"], step["after"], j["final"]) == (sample, want, want)
    assert (step["passes"], step["matches"]) == (passes, matches)
    assert j["final"] == _engine(app_module, rules, sample)


def test_the_rules_after_a_repeating_one_get_its_last_pass(client, app_module):
    """The killable probe walked the rule after it on pass 1's output; it is
    walked again on the text the last pass leaves, and runs on that."""
    rules = [dict(_BATCH, repeat=True),
             {"name": "up", "label": "Up", "type": "callback:upper", "enabled": True,
              "pattern": "(^)(b)"}]
    j = _dry(client, rules, "aaaa")
    assert [s["after"] for s in j["steps"]] == ["b", "B"]
    assert not any(s.get("not_run") or s.get("error") for s in j["steps"])
    assert j["final"] == _engine(app_module, rules, "aaaa") == "B"


@pytest.mark.parametrize("extra", [{}, {"repeat": False}, {"repeat": "yes"}])
def test_without_repeat_the_dry_run_is_one_pass(client, app_module, extra):
    rules = [dict(_HYPHEN, **extra)]
    step = _dry(client, rules, "Hals-nasen-ohren-arzt")["steps"][0]
    assert step["after"] == "Hals-Nasen-ohren-arzt"
    assert "passes" not in step
    # The engine reads a hand-edited "yes" the same way (`is True`).
    app_module.cfg.PIPELINE_RULES = rules
    pl_engine.rebuild_caches()
    assert pl_engine._postprocess_text("Hals-nasen-ohren-arzt") == step["after"]


def test_a_disabled_repeating_rule_is_skipped_once(client):
    step = _dry(client, [dict(_HYPHEN, repeat=True, enabled=False)],
                "Hals-nasen-ohren-arzt")["steps"][0]
    assert step["skipped"] and step["after"] == "Hals-nasen-ohren-arzt"
    assert step["passes"] == 0
