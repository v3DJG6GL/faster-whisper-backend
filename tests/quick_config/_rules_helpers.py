"""Shared helpers for the tests/quick_config/ rule-exposure tests."""
import copy


def expose_first(app_module, rule_type, **extra):
    """Mark the first `rule_type` rule exposed (plus any `extra` fields) so
    /quick-config can see + patch it. Returns its slug (None if none).
    Mutates a deep copy assigned back onto cfg so the test's monkeypatched
    view is isolated; the per-test config reload restores it."""
    rules = copy.deepcopy(list(app_module.cfg.PIPELINE_RULES))
    slug = None
    for r in rules:
        if isinstance(r, dict) and r.get("type") == rule_type:
            r.update(exposed=True, **extra)
            slug = r["name"]
            break
    app_module.cfg.PIPELINE_RULES = rules
    return slug
