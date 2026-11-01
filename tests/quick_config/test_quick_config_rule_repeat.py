"""/quick-config and the per-rule `repeat` switch: it is admin-only (not in
_PATCH_ALLOWED_FIELDS, like note / color), and a body patch from
/quick-config keeps the admin's value (the merge overlays the patched
fields only)."""

import copy


def _expose_upper(app_module, **extra):
    rules = copy.deepcopy(list(app_module.cfg.PIPELINE_RULES))
    r = next(r for r in rules if r.get("type") == "callback:upper")
    r.update(exposed=True, **extra)
    app_module.cfg.PIPELINE_RULES = rules
    return r["name"]


def test_a_quick_config_body_edit_keeps_the_admins_repeat(client, app_module):
    """/quick-config patches the body only: the admin's switch survives it,
    and the rule's fingerprint there covers it like the other fields."""
    slug = _expose_upper(app_module, repeat=True)
    state = client.get("/quick-config/state").json()
    mine = next(r for r in state["rules"] if r["name"] == slug)
    assert mine["repeat"] is True
    r = client.post("/quick-config/state", json={
        "rules_patch": {slug: {"pattern": mine["pattern"] + "?"}},
        "fingerprints": {slug: mine["_fp"]}})
    assert r.status_code == 200, r.text
    assert r.json()["saved"] == [slug]
    after = next(x for x in app_module.cfg.PIPELINE_RULES if x["name"] == slug)
    assert after["repeat"] is True and after["pattern"] == mine["pattern"] + "?"


def test_quick_config_cannot_switch_repeat(client, app_module):
    slug = _expose_upper(app_module)
    r = client.post("/quick-config/state", json={"rules_patch": {slug: {"repeat": True}}})
    assert r.status_code == 400, r.text
    assert "'repeat' is not editable" in r.json()["detail"]
