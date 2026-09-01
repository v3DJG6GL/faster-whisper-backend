"""Integration tests for /quick-config routes."""

import copy

import pytest


def _expose_first_regex_list_rule(app_module):
    """Mark the first regex-list rule exposed so /quick-config can see + patch it.
    Returns its slug. Mutates a deep copy assigned back onto cfg so the test's
    monkeypatched view is isolated; the per-test config reload restores it."""
    rules = copy.deepcopy(list(app_module.cfg.PIPELINE_RULES))
    slug = None
    for r in rules:
        if isinstance(r, dict) and r.get("type") == "regex-list":
            r["exposed"] = True
            slug = r["name"]
            break
    app_module.cfg.PIPELINE_RULES = rules
    return slug


def test_quick_config_page(client):
    r = client.get("/quick-config")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    # renderTrace() builds the metadata row (timestamp, model, source, language,
    # speaker, duration) into a detached node; a refactor once dropped the append
    # and the whole row silently vanished from every trace. Nothing else in the
    # suite looks at the rendered markup, so guard the attachment itself.
    assert "item.appendChild(meta);" in r.text
    # Stage rail + chip colours: a download row's only stage is {"name":
    # "download"} and the standalone translate route writes "translate" (not
    # the "-ing" names), so both need rules or they render unpainted.
    assert ".seg-download" in r.text
    assert ".stage-translate " in r.text
    assert ".stage-download " in r.text
    # The map editor's "n / cap" readout must repaint on row add/remove: it is
    # painted by paintCount(), wired to a childList MutationObserver (the add
    # and delete paths mutate <tr>s directly without re-rendering the editor).
    assert ("new MutationObserver(paintCount).observe(tbl, { childList: true });"
            in r.text)


def test_state_open_mode(client):
    r = client.get("/quick-config/state")
    assert r.status_code == 200
    body = r.json()
    assert "rules" in body
    assert body["role"] == "admin"  # open mode = synthetic admin


def test_usage_open_mode(client):
    r = client.get("/quick-config/usage")
    assert r.status_code == 200
    body = r.json()
    assert "today" in body and "total" in body


def test_post_patch_empty_is_noop(client):
    r = client.post("/quick-config/state", json={"rules_patch": {}})
    assert r.status_code == 200
    assert r.json()["saved"] == []


def test_post_patch_unknown_slug_400(client):
    r = client.post(
        "/quick-config/state",
        json={"rules_patch": {"no-such-rule": {"enabled": False}}},
    )
    assert r.status_code == 400


def test_post_patch_unknown_field_400(client, app_module):
    slug = _expose_first_regex_list_rule(app_module)
    assert slug is not None
    # `label` is admin-only / not in the per-type allow-list -> 400.
    r = client.post(
        "/quick-config/state",
        json={"rules_patch": {slug: {"label": "hacked"}}},
    )
    assert r.status_code == 400


def test_post_patch_unknown_top_level_field_422(client):
    # QuickPatchPayload has extra="forbid".
    r = client.post(
        "/quick-config/state",
        json={"rules_patch": {}, "bogus": 1},
    )
    assert r.status_code == 422


def test_post_patch_valid_field_saves(client, app_module):
    slug = _expose_first_regex_list_rule(app_module)
    r = client.post(
        "/quick-config/state",
        json={"rules_patch": {slug: {"enabled": False}}},
    )
    assert r.status_code == 200
    assert slug in r.json()["saved"]


def test_post_patch_succeeds_despite_guard_failing_untouched_rule(
        client, app_module):
    """A rule the CURRENT guard refuses can sit in the live pipeline (the load
    path never runs the probe — e.g. it was saved before a guard tightening).
    The save-time guard is scoped to the rules the patch changed, so patching a
    DIFFERENT rule must succeed instead of 422ing with an error naming a rule
    the user never touched (the toast then read "admin pipeline has a
    validation error" on every save of any word)."""
    slug = _expose_first_regex_list_rule(app_module)
    assert slug is not None
    rules = copy.deepcopy(list(app_module.cfg.PIPELINE_RULES))
    # Compiles fine, so load-path validation accepts it; only the guard's
    # structural screen (nested/ambiguous repetition) refuses it.
    rules.insert(0, {"name": "legacy-boom", "label": "legacy",
                     "type": "regex-list",
                     "entries": [{"pattern": "(n|d|nd)+#", "replacement": "X"}]})
    app_module.cfg.PIPELINE_RULES = rules
    r = client.post(
        "/quick-config/state",
        json={"rules_patch": {slug: {"enabled": False}}},
    )
    assert r.status_code == 200, r.text
    assert slug in r.json()["saved"]


def test_post_patch_own_guard_failing_pattern_still_422(client, app_module):
    """The scoping must not wave through the caller's OWN bad pattern — a
    patched rule is always in the guard set."""
    slug = _expose_first_regex_list_rule(app_module)
    assert slug is not None
    r = client.post(
        "/quick-config/state",
        json={"rules_patch": {slug: {
            "entries": [{"pattern": "(n|d|nd)+#", "replacement": "X"}]}}},
    )
    assert r.status_code == 422, r.text
    assert "catastrophic backtracking" in r.text


def test_recent_open_mode(client):
    r = client.get("/quick-config/recent")
    assert r.status_code == 200
    body = r.json()
    assert "recent" in body


def test_recent_query_filters_raw_and_final(client):
    import transcriptions_store
    transcriptions_store.record_trace(
        request_id="q1", model="m", raw="patient hat Fieber",
        final="Patient hat Fieber", created_ts=1.0)
    transcriptions_store.record_trace(
        request_id="q2", model="m", raw="andere notiz",
        final="Aspirin verordnet", created_ts=2.0)
    # The term matches the substring across raw OR final, case-insensitively.
    # It travels in the POST body, never the query string: it is dictation
    # text, and a query string is copied into every access log and into
    # browser history, neither of which is the 0600 log file.
    r = client.post("/quick-config/recent/search", json={"q": "fieber"})
    assert r.status_code == 200
    ids = [t["request_id"] for t in r.json()["recent"]]
    assert ids == ["q1"]
    r = client.post("/quick-config/recent/search", json={"q": "ASPIRIN"})
    assert [t["request_id"] for t in r.json()["recent"]] == ["q2"]
    # No query → both rows (newest-first). The unfiltered slice stays a GET.
    r = client.get("/quick-config/recent")
    assert [t["request_id"] for t in r.json()["recent"]] == ["q2", "q1"]
    # No match → empty.
    r = client.post("/quick-config/recent/search", json={"q": "zzznope"})
    assert r.json()["recent"] == []


def test_recent_get_no_longer_filters_by_query_string(client):
    """A dictation term must not be accepted in the URL — that is the whole
    point of the POST endpoint. A stray ?q= is ignored, not honoured."""
    import transcriptions_store
    transcriptions_store.record_trace(
        request_id="q1", model="m", raw="patient hat Fieber",
        final="Patient hat Fieber", created_ts=1.0)
    transcriptions_store.record_trace(
        request_id="q2", model="m", raw="andere notiz",
        final="Aspirin verordnet", created_ts=2.0)
    r = client.get("/quick-config/recent", params={"q": "fieber"})
    assert r.status_code == 200
    assert [t["request_id"] for t in r.json()["recent"]] == ["q2", "q1"]


def test_reapply_rules_status(client):
    r = client.get("/quick-config/reapply-rules/status")
    assert r.status_code == 200
    # captures_reapply.status() returns the worker state dict.
    assert isinstance(r.json(), dict)


def test_reapply_rules_start_captures_disabled(client, app_module):
    # CAPTURE_RECORDINGS_ENABLED defaults False -> idle, no-op note.
    app_module.cfg.CAPTURE_RECORDINGS_ENABLED = False
    r = client.post("/quick-config/reapply-rules")
    assert r.status_code == 200
    assert r.json().get("status") == "idle"


def _expose_first_map_rule(app_module):
    """Mark a callback:map rule exposed and return its slug (None if none)."""
    rules = copy.deepcopy(list(app_module.cfg.PIPELINE_RULES))
    slug = None
    for r in rules:
        if isinstance(r, dict) and r.get("type") == "callback:map":
            r["exposed"] = True
            slug = r["name"]
            break
    app_module.cfg.PIPELINE_RULES = rules
    return slug


def test_post_patch_oversized_map_400(client, app_module):
    """A map patch bigger than the schema cap is rejected at ingress, before
    the stamping loop walks the caller's dict twice on the event loop. It was
    already doomed (Pydantic 422s it inside save_overrides) — this only moves
    the rejection earlier and makes it a 400 like the sibling map guards."""
    import quick_config_routes

    slug = _expose_first_map_rule(app_module)
    assert slug is not None, "fixture config has no callback:map rule"
    import config_store

    cap = quick_config_routes._MAP_MAX_ENTRIES
    # Track the schema rather than re-pinning the literal: this still catches
    # the derivation silently taking its fallback if pydantic's metadata
    # layout changes, without failing on a deliberate schema-side cap change.
    expected = next(
        m.max_length
        for m in config_store.MapRule.model_fields["map"].metadata
        if getattr(m, "max_length", None) is not None)
    assert cap == expected
    big = {f"wort{i}": str(i) for i in range(cap + 1)}
    r = client.post("/quick-config/state", json={"rules_patch": {slug: {"map": big}}})
    assert r.status_code == 400, r.text
    assert str(cap) in r.json()["detail"]


def test_post_patch_map_at_cap_is_not_rejected_by_the_guard(client, app_module):
    """Exactly at the cap must still pass the ingress guard (off-by-one)."""
    import quick_config_routes

    slug = _expose_first_map_rule(app_module)
    assert slug is not None
    at_cap = {f"wort{i}": str(i)
              for i in range(quick_config_routes._MAP_MAX_ENTRIES)}
    r = client.post("/quick-config/state", json={"rules_patch": {slug: {"map": at_cap}}})
    assert r.status_code != 400, r.text


def test_patch_response_hides_global_capture_count_from_nonadmin(
    client, app_module, make_user_key,
):
    """captures_store.count() with no args is the unfiltered admin-scope total,
    and a quick_config-only identity has captures scope "none" — so a non-admin
    save must report 0. It also keeps the page's silent re-apply kick (fired on
    captures_count > 0) shut, since POST /reapply-rules is admin-only and would
    leave a permanent "Re-apply failed: HTTP 403" strip."""
    from tests.conftest import bearer

    slug = _expose_first_regex_list_rule(app_module)
    app_module.cfg.CAPTURE_RECORDINGS_ENABLED = True

    calls = []

    import captures_store
    orig = captures_store.count

    def _counting(*a, **kw):
        calls.append((a, kw))
        return 7

    captures_store.count = _counting
    try:
        _uid, admin_raw = make_user_key("root", is_admin=True)
        _uid2, user_raw = make_user_key("alice", pages={"quick_config": "own"})

        r = client.post(
            "/quick-config/state",
            json={"rules_patch": {slug: {"enabled": False}}},
            headers=bearer(user_raw),
        )
        assert r.status_code == 200, r.text
        assert slug in r.json()["saved"]          # the save itself still works
        assert r.json()["captures_count"] == 0
        assert calls == []                        # and the query never ran

        r = client.post(
            "/quick-config/state",
            json={"rules_patch": {slug: {"enabled": True}}},
            headers=bearer(admin_raw),
        )
        assert r.status_code == 200, r.text
        assert r.json()["captures_count"] == 7    # admin behaviour unchanged
        assert len(calls) == 1
    finally:
        captures_store.count = orig


def test_state_carries_locked_and_role_for_nonadmin(client, app_module,
                                                    make_user_key):
    """The /quick-config page renders a locked rule read-only for a non-admin,
    so the state payload must carry both signals it gates on: `locked` on each
    rule and the caller's `role`."""
    from tests.conftest import bearer

    rules = copy.deepcopy(list(app_module.cfg.PIPELINE_RULES))
    slug = None
    for r in rules:
        if isinstance(r, dict) and r.get("type") == "regex-list":
            r["exposed"] = True
            r["locked"] = True
            slug = r["name"]
            break
    app_module.cfg.PIPELINE_RULES = rules
    assert slug is not None

    make_user_key("root", is_admin=True)  # flips lockdown
    _uid, raw = make_user_key("alice", pages={"quick_config": "own"})

    body = client.get("/quick-config/state", headers=bearer(raw)).json()
    assert body["role"] == "user"
    by_name = {r["name"]: r for r in body["rules"]}
    assert by_name[slug]["locked"] is True
    # Every visible rule carries the flag (never dropped by exclude_none),
    # so the page can decide per card without a second lookup.
    assert all("locked" in r for r in body["rules"])


@pytest.mark.parametrize("bad", [None, [], ""])
def test_post_patch_falsy_nondict_map_400(client, app_module, bad):
    """A FALSY non-dict map (None, [], "") used to be coerced to {} by an
    `or {}` before the isinstance guard, slipping past the precise 400 into a
    generic Pydantic 422 from save_overrides."""
    slug = _expose_first_map_rule(app_module)
    assert slug is not None
    r = client.post("/quick-config/state",
                    json={"rules_patch": {slug: {"map": bad}}})
    assert r.status_code == 400, r.text
    assert "must be an object" in r.json()["detail"]


def _expose_two_map_rules(app_module):
    """Expose the first callback:map rule and a clone of it, returning both
    slugs."""
    rules = copy.deepcopy(list(app_module.cfg.PIPELINE_RULES))
    first = None
    for r in rules:
        if isinstance(r, dict) and r.get("type") == "callback:map":
            r["exposed"] = True
            first = r
            break
    if first is None:
        return None, None
    clone = copy.deepcopy(first)
    clone["name"] = "zweite-map"
    rules.insert(rules.index(first) + 1, clone)
    app_module.cfg.PIPELINE_RULES = rules
    return first["name"], clone["name"]


def test_post_patch_map_total_over_cap_400(client, app_module):
    """The per-slug cap bounds ONE map; a patch naming several map rules must
    not walk a multiple of the cap on the event loop. The request-wide sum is
    bounded too, while each map alone stays under the per-slug cap."""
    import quick_config_routes

    slug_a, slug_b = _expose_two_map_rules(app_module)
    assert slug_a and slug_b, "fixture config has no callback:map rule"
    half = quick_config_routes._MAP_MAX_ENTRIES // 2 + 1
    map_a = {f"awort{i}": str(i) for i in range(half)}
    map_b = {f"bwort{i}": str(i) for i in range(half)}
    r = client.post("/quick-config/state", json={"rules_patch": {
        slug_a: {"map": map_a}, slug_b: {"map": map_b}}})
    assert r.status_code == 400, r.text
    # Each alone is under the per-slug cap and passes the ingress guards.
    r = client.post("/quick-config/state",
                    json={"rules_patch": {slug_a: {"map": map_a}}})
    assert r.status_code != 400, r.text


def test_hidden_rule_validation_error_is_fully_redacted(
        client, app_module, make_user_key):
    """A 422 caused by a rule the caller may not see must not leak the slug,
    its list position, or its rule type — a plain Pydantic field error carries
    all three in its dotted loc."""
    from tests.conftest import bearer

    slug = _expose_first_regex_list_rule(app_module)
    assert slug is not None
    rules = copy.deepcopy(list(app_module.cfg.PIPELINE_RULES))
    # Hidden (not exposed) rule whose map KEY fails the schema pattern —
    # sits fine in the live config (assignment is unvalidated) but 422s the
    # next full save.
    rules.insert(0, {"name": "geheim", "label": "geheim",
                     "type": "callback:map", "map": {"böse<>#key": "x"}})
    app_module.cfg.PIPELINE_RULES = rules

    make_user_key("root", is_admin=True)  # flips lockdown
    _uid, raw = make_user_key("alice", pages={"quick_config": "own"})

    r = client.post("/quick-config/state",
                    json={"rules_patch": {slug: {"enabled": False}}},
                    headers=bearer(raw))
    assert r.status_code == 422, r.text
    assert "geheim" not in r.text
    errs = r.json()["errors"]
    assert errs, "the hidden rule's schema error must still be reported"
    for e in errs:
        assert e["loc"] == "<hidden rule>", e
        assert e["msg"] == "<hidden rule>", e


def test_redact_collapses_hidden_rule_ordinals():
    """config_store's guard messages read `rule {idx} ({slug!r}) entry {e}:`
    — after the slug swap the ordinal still gave away the hidden rule's list
    position and entry count."""
    import quick_config_routes as q

    class _Perms:
        def can_see_rule(self, rule):
            return bool(rule.get("exposed"))

    user = {"is_admin": False, "permissions": _Perms()}
    rules = [{"name": "geheim", "exposed": False},
             {"name": "meins", "exposed": True}]
    out = q._redact_invisible_slugs(
        [{"loc": "PIPELINE_RULES", "msg": "rule 0 ('geheim') entry 2: boom"}],
        user, rules)
    assert out == [{"loc": "PIPELINE_RULES", "msg": "a hidden rule: boom"}]
    # A dotted Pydantic loc on the hidden index is blanked wholesale so the
    # page's '<hidden rule>' guard routes it to the generic toast.
    out = q._redact_invisible_slugs(
        [{"loc": "PIPELINE_RULES.0.callback:map.map.x",
          "msg": "Input should be a valid string"}],
        user, rules)
    assert out == [{"loc": "<hidden rule>", "msg": "<hidden rule>"}]
    # The visible rule's errors stay legible.
    out = q._redact_invisible_slugs(
        [{"loc": "PIPELINE_RULES.1.regex-list.entries.0.pattern",
          "msg": "rule 1 ('meins'): bad"}],
        user, rules)
    assert out == [{"loc": "PIPELINE_RULES.1.regex-list.entries.0.pattern",
                    "msg": "rule 1 ('meins'): bad"}]


def test_concurrent_patches_do_not_lose_an_update(client, app_module,
                                                  monkeypatch):
    """Two overlapping patches of DIFFERENT rules both used to snapshot the
    pre-update PIPELINE_RULES and both write the whole key across the
    offloaded save — the second save silently reverted the first caller's
    edit. apply_rules_patch is serialized now, so both edits must survive."""
    import time as _time
    from concurrent.futures import ThreadPoolExecutor

    import config_store

    slug_a = _expose_first_regex_list_rule(app_module)
    slug_b = _expose_first_map_rule(app_module)
    assert slug_a and slug_b and slug_a != slug_b

    orig_save = config_store.save_overrides

    def _slow_save(*a, **kw):
        _time.sleep(0.2)  # hold the offloaded-save window open
        return orig_save(*a, **kw)

    monkeypatch.setattr(config_store, "save_overrides", _slow_save)

    def _patch(slug):
        return client.post("/quick-config/state",
                           json={"rules_patch": {slug: {"enabled": False}}})

    with ThreadPoolExecutor(max_workers=2) as pool:
        ra, rb = pool.map(lambda s: _patch(s), [slug_a, slug_b])
    assert ra.status_code == 200, ra.text
    assert rb.status_code == 200, rb.text

    by_name = {}
    for r in app_module.cfg.PIPELINE_RULES:
        d = r.model_dump() if hasattr(r, "model_dump") else dict(r)
        by_name[d.get("name")] = d
    assert by_name[slug_a].get("enabled") is False
    assert by_name[slug_b].get("enabled") is False
