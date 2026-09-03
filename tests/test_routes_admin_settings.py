"""Integration tests for /settings admin routes (admin UI enabled by default)."""

import pytest


def test_settings_page_loopback(client):
    r = client.get("/settings")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_get_state_open_mode(client):
    r = client.get("/settings/state")
    assert r.status_code == 200
    body = r.json()
    assert "fields" in body
    assert "BEAM_SIZE" in body["fields"]


def test_post_state_valid(client):
    # Use a value that DIFFERS from the baseline (BEAM_SIZE default is 10) so
    # save_overrides actually records it as a changed field.
    r = client.post("/settings/state", json={"BEAM_SIZE": 3})
    assert r.status_code == 200
    body = r.json()
    assert "BEAM_SIZE" in body["saved"]
    assert "hot_applied" in body
    assert "requires_restart" in body


def test_stats_own_shows_machine_round_trip(client, app_module):
    """The /settings switch behind decision 2 (own-scope users see machine
    cards) saves, hot-applies onto cfg and resets back to the default."""
    cfg = app_module.cfg
    assert cfg.STATS_OWN_SCOPE_SHOW_SYSTEM_METRICS is False
    r = client.post("/settings/state", json={"STATS_OWN_SCOPE_SHOW_SYSTEM_METRICS": True})
    assert r.status_code == 200
    body = r.json()
    assert "STATS_OWN_SCOPE_SHOW_SYSTEM_METRICS" in body["saved"]
    assert cfg.STATS_OWN_SCOPE_SHOW_SYSTEM_METRICS is True
    state = client.get("/settings/state").json()["fields"]["STATS_OWN_SCOPE_SHOW_SYSTEM_METRICS"]
    assert state["value"] is True and state["provenance"] == "local.json"
    r = client.post("/settings/state", json={"STATS_OWN_SCOPE_SHOW_SYSTEM_METRICS": False})
    assert r.status_code == 200
    assert cfg.STATS_OWN_SCOPE_SHOW_SYSTEM_METRICS is False


def test_post_state_invalid_value_422(client):
    # BEAM_SIZE is Annotated[int, Field(ge=1, le=20)] -> 999 fails validation.
    r = client.post("/settings/state", json={"BEAM_SIZE": 999})
    assert r.status_code == 422
    assert "errors" in r.json()


def test_reset_to_default_clears_local_override(client):
    """Resetting a setting to its in-repo default must DELETE the override key
    from config.local.json — clearing the 'local.json' badge AND reverting the
    running value. Regression: the WebUI '↺ Reset to default' button submits the
    default *value* (not a removal), which previously rewrote the key and left
    the badge stuck on 'local.json' (and the running cfg on the stale value)."""
    from faster_whisper_backend import config_store

    default_val = client.get("/settings/state").json()["fields"]["BEST_OF"]["default_value"]
    assert default_val is not None
    override_val = default_val + 1   # BEST_OF is ge=1, le=20 -> still valid

    # Override it: key present on disk, badge = local.json, value applied live.
    client.post("/settings/state", json={"BEST_OF": override_val})
    field = client.get("/settings/state").json()["fields"]["BEST_OF"]
    assert field["provenance"] == "local.json"
    assert field["value"] == override_val
    assert "BEST_OF" in config_store.load_overrides()

    # Reset = submit the default value back (exactly what the ↺ button sends).
    saved = client.post("/settings/state", json={"BEST_OF": default_val}).json()["saved"]
    assert "BEST_OF" in saved

    # Key gone from disk, badge cleared, running value reverted to the baseline.
    field = client.get("/settings/state").json()["fields"]["BEST_OF"]
    assert "BEST_OF" not in config_store.load_overrides()
    assert field["provenance"] == "default"
    assert field["value"] == default_val


def test_post_default_value_creates_no_override(client):
    """Submitting a value equal to the baseline when nothing was overridden is a
    no-op: prune-on-default keeps the key out of config.local.json entirely."""
    from faster_whisper_backend import config_store

    default_val = client.get("/settings/state").json()["fields"]["BEST_OF"]["default_value"]
    body = client.post("/settings/state", json={"BEST_OF": default_val}).json()
    assert body["saved"] == []
    assert "BEST_OF" not in config_store.load_overrides()


def test_reset_float_field_default_sent_as_int_clears_override(client):
    """REPETITION_PENALTY's default is a float (1.0), but the JS client submits a
    whole-number float without its decimal (JSON.stringify(1.0) -> '1'), so the
    server receives int 1. Prune-on-default must treat int 1 == float 1.0 as the
    default and drop the override — a json.dumps comparison ('1' != '1.0') would
    miss it and leave the 'local.json' badge stuck."""
    from faster_whisper_backend import config_store

    default_val = client.get("/settings/state").json()["fields"]["REPETITION_PENALTY"]["default_value"]
    assert default_val == 1.0

    # Override it, then "reset" by POSTing the default as a bare int — exactly
    # what the WebUI sends for a whole-number float.
    client.post("/settings/state", json={"REPETITION_PENALTY": 2.0})
    assert "REPETITION_PENALTY" in config_store.load_overrides()

    client.post("/settings/state", json={"REPETITION_PENALTY": 1})   # int, not 1.0
    field = client.get("/settings/state").json()["fields"]["REPETITION_PENALTY"]
    assert "REPETITION_PENALTY" not in config_store.load_overrides()
    assert field["provenance"] == "default"
    assert field["value"] == default_val


def _as_js_sends(v):
    """Mimic how the browser serializes a value on reset: JSON.stringify has no
    int/float distinction, so a whole-number float (1.0, 0.0) goes out without
    its decimal and arrives server-side as an int. Applied recursively so this
    reproduces the real wire payload for list/tuple/dict-valued fields too."""
    if isinstance(v, bool):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, list):
        return [_as_js_sends(x) for x in v]
    if isinstance(v, dict):
        return {k: _as_js_sends(x) for k, x in v.items()}
    return v


def test_every_field_reset_to_default_is_pruned(client):
    """Sweep EVERY admin setting: posting its in-repo default (coerced the way
    the browser sends it) must be recognized as 'not an override' and leave
    config.local.json untouched. Guards against type-specific prune gaps like
    the int/float REPETITION_PENALTY bug for any current or future field."""
    from faster_whisper_backend.admin import routes as admin_routes
    from faster_whisper_backend import config_store

    fields = client.get("/settings/state").json()["fields"]
    checked = 0
    not_pruned = []
    post_failed = []
    for name, meta in fields.items():
        if meta["provenance"] == "env":
            continue                      # env-pinned saves are ignored anyway
        if name in admin_routes._PRUNE_EXEMPT:
            continue                      # bespoke override mgmt (see below)
        dv = meta["default_value"]
        if dv is None:
            continue                      # null default -> reset sends null (handled)
        checked += 1
        r = client.post("/settings/state", json={name: _as_js_sends(dv)})
        if r.status_code != 200:
            post_failed.append((name, r.status_code, r.text[:160]))
            continue
        if name in r.json()["saved"]:
            not_pruned.append((name, dv, _as_js_sends(dv)))

    assert not not_pruned, (
        "default value NOT recognized as default (override persists, badge sticks):\n"
        + "\n".join(f"  {n}: baseline={d!r} sent={s!r}" for n, d, s in not_pruned)
    )
    assert not post_failed, f"posting the default value failed: {post_failed}"
    assert config_store.load_overrides() == {}   # nothing leaked onto disk
    assert checked >= 10                          # sanity: the sweep ran broadly


def test_pipeline_rules_not_auto_pruned(client):
    """PIPELINE_RULES is intentionally exempt from prune-on-default: a local copy
    equal to the factory rules SHADOWS config.json (managed by the pipeline
    page's dedicated 'clear local override' action), so saving rules equal to the
    factory default must KEEP the override, not silently drop it. Locks in the
    _PRUNE_EXEMPT carve-out so a future change can't start auto-pruning it."""
    from faster_whisper_backend import config_store

    rules = client.get("/settings/state").json()["fields"]["PIPELINE_RULES"]["default_value"]
    body = client.post("/settings/state", json={"PIPELINE_RULES": rules}).json()
    assert "PIPELINE_RULES" in body["saved"]
    assert "PIPELINE_RULES" in config_store.load_overrides()


def test_enum_choices_match_schema(client):
    """Dropdown options are derived from the AdminConfig Literal (single source).
    GET /settings/state must surface `choices` == the field's Literal values for
    every enum field, and None for non-enum fields — so the UI <select> and the
    server-side validation can never disagree. Guards the _field_choices dedup."""
    import typing
    from faster_whisper_backend import config_store

    def literal_args(field):
        ann = config_store.AdminConfig.model_fields[field].annotation
        for c in (ann, *typing.get_args(ann)):
            if typing.get_origin(c) is typing.Literal:
                return list(typing.get_args(c))
        return None

    fields = client.get("/settings/state").json()["fields"]
    enum_count = 0
    for name in fields:
        if name not in config_store.AdminConfig.model_fields:
            continue
        expected = literal_args(name)
        assert fields[name].get("choices") == expected, name
        if expected is not None:
            enum_count += 1

    # Concrete spot-checks: API choices ARE the Literal (catch a broken extractor).
    assert fields["MODEL_COMPUTE_TYPE"]["choices"] == list(config_store.ComputeLit.__args__)
    assert fields["CONVERT_QUANTIZATION"]["choices"] == list(config_store.ConvertQuantLit.__args__)
    assert fields["MODEL_DEVICE"]["choices"] == ["cuda", "cpu"]
    assert fields["BEST_OF"]["choices"] is None        # non-enum -> free input, no dropdown
    assert enum_count >= 6                              # sanity: the sweep found the enums


def test_settings_page_has_no_hardcoded_enum_opts(client):
    """Both editors (main form + per-model pane) build dropdowns from the server
    `choices`, not hardcoded JS arrays — so options can't drift from the schema."""
    page = client.get("/settings").text
    assert "_ed.choices" in page                        # main form derives from choices
    assert "fieldDef(field).choices" in page            # per-model pane derives too
    # The old hardcoded compute-type option arrays must be gone.
    assert "'int8_float16','int8','float32'" not in page


def test_get_factory_rules(client):
    r = client.get("/settings/factory-rules")
    assert r.status_code == 200
    assert "PIPELINE_RULES" in r.json()
    assert isinstance(r.json()["PIPELINE_RULES"], list)


def test_post_factory_rules_non_list_400(client):
    r = client.post("/settings/factory-rules", json={"PIPELINE_RULES": "not-a-list"})
    assert r.status_code == 400


@pytest.mark.parametrize("bad", [[[]], ["x"], [1], [None]])
def test_post_factory_rules_non_mapping_element_400(client, bad):
    """A list whose ELEMENTS aren't objects must 400 like a non-list does.
    save_factory_rules splats each element (`{**r, "seeded": True}`) before it
    validates, so a non-mapping raised TypeError past the handler's
    ValidationError/OSError catches — a bare 500 with a stack trace."""
    r = client.post("/settings/factory-rules", json={"PIPELINE_RULES": bad})
    assert r.status_code == 400, r.text


def _seed_factory(monkeypatch, tmp_path, rules):
    """Repoint config_store's factory file at a per-test temp config.json and
    seed it with `rules`, so persisting factory-rules tests never clobber the
    committed repo config.json. Mirrors conftest's OVERRIDES_PATH repoint:
    FACTORY_PATH is a default ARG bound at def time, so each function's
    __defaults__ is rewritten in addition to the module constant."""
    import json
    from faster_whisper_backend import config_store

    tmp_factory = str(tmp_path / "factory_config.json")
    with open(tmp_factory, "w", encoding="utf-8") as f:
        json.dump({"schema_version": 1, "PIPELINE_RULES": rules}, f)
    monkeypatch.setattr(config_store, "FACTORY_PATH", tmp_factory, raising=False)
    for fn in (config_store.load_factory_rules, config_store.save_factory_rules):
        defaults = list(fn.__defaults__ or ())
        if defaults:
            defaults[-1] = tmp_factory
            monkeypatch.setattr(fn, "__defaults__", tuple(defaults), raising=False)
    return tmp_factory


def _rules(*names):
    """A valid rule list: one one-entry regex-list rule per name, terminal last."""
    out = [{"name": n, "label": n.title(), "type": "regex-list",
            "entries": [{"pattern": n[0], "replacement": n[0].upper()}]} for n in names]
    out.append({"name": "trim-edges", "label": "Trim edges", "type": "terminal"})
    return out


def test_post_factory_rules_preserves_order(client, tmp_path, monkeypatch):
    """An order-only promote persists the posted order to config.json verbatim —
    response, GET, and on-disk file all agree. (Backs the JS "Promote order".)"""
    import json

    tmp_factory = _seed_factory(monkeypatch, tmp_path, _rules("alpha", "beta", "gamma"))
    reordered = _rules("gamma", "alpha", "beta")   # terminal stays last
    r = client.post("/settings/factory-rules", json={"PIPELINE_RULES": reordered})
    assert r.status_code == 200, r.text
    expected = ["gamma", "alpha", "beta", "trim-edges"]
    assert [x["name"] for x in r.json()["rules"]] == expected
    g = client.get("/settings/factory-rules")
    assert [x["name"] for x in g.json()["PIPELINE_RULES"]] == expected
    with open(tmp_factory, encoding="utf-8") as f:
        raw = json.load(f)
    assert [x["name"] for x in raw["PIPELINE_RULES"]] == expected


def test_post_factory_rules_reports_shadowed_by_local(client, tmp_path, monkeypatch):
    """shadowed_by_local is False with no local PIPELINE_RULES override and True
    once one exists — the flag the post-promote "clear local override" UX keys on."""
    from faster_whisper_backend import config_store

    _seed_factory(monkeypatch, tmp_path, _rules("alpha", "beta"))
    r = client.post("/settings/factory-rules", json={"PIPELINE_RULES": _rules("beta", "alpha")})
    assert r.status_code == 200, r.text
    assert r.json()["shadowed_by_local"] is False

    config_store.save_overrides({"PIPELINE_RULES": _rules("alpha", "beta")})
    r2 = client.post("/settings/factory-rules", json={"PIPELINE_RULES": _rules("beta", "alpha")})
    assert r2.status_code == 200, r2.text
    assert r2.json()["shadowed_by_local"] is True


def test_test_pipeline_dry_run(client):
    r = client.post(
        "/settings/test-pipeline",
        json={
            "sample": "  hallo welt  ",
            "rules": [
                {"name": "noop", "type": "regex-list", "enabled": True,
                 "entries": [{"pattern": "welt", "replacement": "world"}]},
            ],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "steps" in body and "final" in body
    # The regex-list entry runs, then the implicit terminal trim strips edges.
    assert body["final"] == "hallo world"


def test_test_pipeline_regex_list_skips_bad_entry(client):
    # A regex-list with one uncompilable entry must NOT blank the whole card:
    # the engine (main.rebuild_caches) skips the bad entry per-entry and still
    # applies the valid ones. The dry-run mirrors that and reports the bad
    # pattern as an advisory rather than discarding every entry's effect.
    r = client.post(
        "/settings/test-pipeline",
        json={
            "sample": "foo bar",
            "rules": [
                {"name": "rl", "type": "regex-list", "enabled": True, "entries": [
                    {"pattern": "foo", "replacement": "X"},
                    {"pattern": "(", "replacement": "Y"},   # uncompilable
                    {"pattern": "bar", "replacement": "Z"},
                ]},
            ],
        },
    )
    assert r.status_code == 200
    step = r.json()["steps"][0]
    assert step["after"] == "X Z"        # valid entries applied despite the bad one
    assert step["matches"] == 2
    assert step["error"]                 # bad pattern surfaced as an advisory
    assert r.json()["final"] == "X Z"


def test_test_pipeline_rules_not_list_400(client):
    r = client.post("/settings/test-pipeline", json={"sample": "x", "rules": "nope"})
    assert r.status_code == 400


def test_test_pipeline_nested_repetition_screened_not_run(client):
    """An exponential shape must be refused on structure, not started: a
    timed-out dry-run thread is abandoned, not killed, and would pin a core
    for the life of the process. Both the regex-list and single-pattern paths
    screen with regex_guard before compiling."""
    r = client.post(
        "/settings/test-pipeline",
        json={
            "sample": "aaaaaaaaaaaaaaaaaaaa!",
            "rules": [
                {"name": "boom-list", "type": "regex-list", "enabled": True,
                 "entries": [{"pattern": r"(\w+ )+$", "replacement": ""}]},
                {"name": "boom-single", "type": "callback:upper",
                 "enabled": True, "pattern": r"(\w+ ?)+"},
            ],
        },
    )
    assert r.status_code == 200, r.text
    steps = r.json()["steps"]
    assert "nested repetition" in (steps[0]["error"] or "")
    assert not steps[0]["slow"]
    assert "nested repetition" in (steps[1]["error"] or "")
    assert not steps[1]["slow"]
    # Screened steps carry `not_run` so the panel renders a warning, not an
    # engine-parity claim: an already-saved rule of this shape still runs in
    # the live pipeline (main.rebuild_caches does no structural screen).
    assert steps[0].get("not_run") is True
    assert steps[1].get("not_run") is True


def test_test_pipeline_map_rule_non_dict_map_is_step_error(client):
    """A callback:map rule whose `map` is a list/string (editor rules skip
    _PIPELINE_RULE_ADAPTER) must degrade into a per-step error card, not an
    unhandled 500 from m.items()."""
    r = client.post(
        "/settings/test-pipeline",
        json={"sample": "hallo",
              "rules": [{"name": "x", "label": "x", "type": "callback:map",
                         "enabled": True, "map": ["hallo"]}]},
    )
    assert r.status_code == 200, r.text
    step = r.json()["steps"][0]
    assert step["error"]
    assert step["after"] == "hallo"      # sample passed through unchanged


def test_admin_rules_save_survives_concurrent_quick_config_patch(
        client, app_module, monkeypatch):
    """POST /settings/state's PIPELINE_RULES save queues on the SAME lock as
    /quick-config's read-modify-write patches. Without it, a quick-config
    patch whose cfg snapshot was taken mid-admin-save wrote the whole key
    back and silently reverted the admin's edit (both requests 200)."""
    import copy
    import time as _time
    from concurrent.futures import ThreadPoolExecutor

    from faster_whisper_backend import config_store

    rules = copy.deepcopy(list(app_module.cfg.PIPELINE_RULES))
    slug = None
    for r in rules:
        if isinstance(r, dict) and r.get("type") == "regex-list":
            r["exposed"] = True
            slug = r["name"]
            break
    app_module.cfg.PIPELINE_RULES = rules
    assert slug is not None

    # The admin edits a DIFFERENT rule's label than the one quick-config
    # patches, so both edits can coexist in the final document.
    admin_rules = copy.deepcopy(rules)
    edited = next(r for r in admin_rules
                  if isinstance(r, dict) and r.get("name") != slug)
    edited["label"] = "ADMIN-EDIT"

    orig_save = config_store.save_overrides

    def _slow_save(*a, **kw):
        _time.sleep(0.3)              # hold the offloaded-save window open
        return orig_save(*a, **kw)

    monkeypatch.setattr(config_store, "save_overrides", _slow_save)

    def _admin():
        return client.post("/settings/state",
                           json={"PIPELINE_RULES": admin_rules})

    def _quick():
        _time.sleep(0.1)              # land inside the admin save's window
        return client.post("/quick-config/state",
                           json={"rules_patch": {slug: {"enabled": False}}})

    with ThreadPoolExecutor(max_workers=2) as pool:
        fa, fq = pool.submit(_admin), pool.submit(_quick)
        ra, rq = fa.result(), fq.result()
    assert ra.status_code == 200, ra.text
    assert rq.status_code == 200, rq.text

    by_name = {}
    for r in config_store.load_overrides()["PIPELINE_RULES"]:
        d = r.model_dump() if hasattr(r, "model_dump") else dict(r)
        by_name[d.get("name")] = d
    assert by_name[edited["name"]].get("label") == "ADMIN-EDIT"
    assert by_name[slug].get("enabled") is False


def test_test_pipeline_rule_cap_tracks_schema():
    """_TEST_PIPELINE_MAX_RULES is derived from AdminConfig.PIPELINE_RULES'
    max_length; a metadata-layout change must not silently fall back."""
    from typing import get_args

    from faster_whisper_backend.admin import routes as admin_routes
    from faster_whisper_backend import config_store

    ann = config_store.AdminConfig.model_fields["PIPELINE_RULES"].annotation
    fi = get_args(ann)[0].__metadata__[0]
    expected = next(m.max_length for m in fi.metadata
                    if getattr(m, "max_length", None) is not None)
    assert admin_routes._TEST_PIPELINE_MAX_RULES == expected


def test_post_state_requires_admin_when_locked(client, make_user_key):
    from conftest import bearer

    make_user_key("root", is_admin=True)
    _uid, raw = make_user_key("alice", is_admin=False)
    r = client.post("/settings/state", json={"BEAM_SIZE": 5}, headers=bearer(raw))
    assert r.status_code == 403


# ---------------------------------------------------------------------------
# Env-pinned provenance + read-only WebUI behaviour
# ---------------------------------------------------------------------------
# env_pinned_fields() reads os.environ live, and get_state()/post_state() use it,
# so setting the env var is enough to exercise the precedence + badge path
# without reloading config.

def test_get_state_marks_env_pinned(client, monkeypatch):
    monkeypatch.setenv("WHISPER_BEAM_SIZE", "3")
    field = client.get("/settings/state").json()["fields"]["BEAM_SIZE"]
    assert field["provenance"] == "env"
    assert field["env_var"] == "WHISPER_BEAM_SIZE"


def test_post_state_env_pinned_is_ignored_at_runtime(client, monkeypatch):
    # An env-pinned field saves to config.local.json but must NOT change the
    # running cfg — the env var wins. The response flags it as ignored.
    monkeypatch.setenv("WHISPER_BEAM_SIZE", "3")
    body = client.post("/settings/state", json={"BEAM_SIZE": 7}).json()
    assert "BEAM_SIZE" in body["env_pinned_ignored"]
    assert "BEAM_SIZE" not in body["hot_applied"]


def test_settings_page_greys_out_env_pinned_inputs(client):
    # The rendered admin page ships the JS/CSS that disables + greys env-pinned
    # editors (the runtime DOM disabling is driven by provenance=="env").
    text = client.get("/settings").text
    assert "function disableEnvPinnedEditor" in text
    assert "if (isEnvPinned(name)) return;" in text   # setDirty guard
    assert ".field.env-pinned" in text                # greyed styling


def test_settings_page_injects_mo_constants(client):
    """The per-model pane's LOAD_TIME_FIELDS set and FIELD_META are no longer
    hand-written JS literals — they are derived from config_store /
    ModelOverride and substituted into the {{MO_*_JSON}} placeholders at
    render time. The rendered script must carry the real data (virtual
    REVISION in the load-time set; schema-derived NUM_WORKERS bounds in the
    field meta) and no unsubstituted placeholder."""
    import json
    import re

    text = client.get("/settings").text
    assert not re.findall(r"\{\{MO_[A-Z_]+\}\}", text)

    m = re.search(r"const LOAD_TIME_FIELDS = new Set\((\[.*?\])\);", text)
    assert m, "injected LOAD_TIME_FIELDS not found"
    load_time = json.loads(m.group(1))
    assert "REVISION" in load_time
    assert "MODEL_DEVICE" in load_time

    m = re.search(r"const FIELD_META = (\{.*?\});\n", text)
    assert m, "injected FIELD_META not found"
    meta = json.loads(m.group(1))
    assert "NUM_WORKERS" in meta
    assert meta["NUM_WORKERS"]["kind"] == "int"
    assert meta["NUM_WORKERS"]["min"] == 1 and meta["NUM_WORKERS"]["max"] == 8
    # Overlay extras survive the merge (kinds the schema can't express).
    assert meta["DEFAULT_PROMPT"]["kind"] == "textarea"
    assert meta["NO_SPEECH_THRESHOLD"]["kind"] == "nullable_float"

    # Every FIELD_META key is a real ModelOverride field — the payload once
    # shipped seven CAPTURES_* rows no renderer could ever read.
    from faster_whisper_backend.admin import routes as admin_routes
    from faster_whisper_backend import config_store
    assert (set(json.loads(admin_routes._MO_FIELD_META_JSON))
            <= set(config_store.ModelOverride.model_fields))


def test_model_sections_cover_every_model_override_field(client):
    """The per-model editor renders only the fields SECTIONS names (no
    catch-all), so a ModelOverride field absent from the literal is invisible
    and uneditable there — 15 stage/translation fields once shipped that way.
    Mirrors test_field_groups_cover_every_setting for the per-model pane."""
    import re

    from faster_whisper_backend import config_store

    text = client.get("/settings").text
    m = re.search(r"const SECTIONS = \[(.*?)\n  \];", text, re.S)
    assert m, "SECTIONS literal not found in the rendered page"
    listed = set()
    for arr in re.findall(r"(?:basic|adv):\s*\[([^\]]*)\]", m.group(1)):
        listed.update(re.findall(r"'([A-Z][A-Z0-9_]*)'", arr))
    expected = (set(config_store.ModelOverride.model_fields)
                - {"PIPELINE_RULES_INCLUDE", "PIPELINE_RULES_EXCLUDE"})
    missing = expected - listed
    stale = listed - expected
    assert not missing, f"ModelOverride fields missing from SECTIONS: {sorted(missing)}"
    assert not stale, f"SECTIONS entries that are not ModelOverride fields: {sorted(stale)}"


def test_server_ident_report_never_prints_none(app_module, monkeypatch):
    """DOWNLOAD_ROOT is `str | None` (None = standard HF cache); the copy
    report and the card fields must resolve the same fallback string rather
    than f-stringing a literal 'models None'."""
    from faster_whisper_backend.admin import routes as admin_routes

    monkeypatch.setattr(admin_routes.cfg, "DOWNLOAD_ROOT", None)
    ident = admin_routes._server_ident_fields()
    assert "None" not in ident["report"]
    tail = ident["report"].splitlines()[-1]
    assert tail.split(" · models ")[-1] == ident["models_dir"]


def test_save_dispatches_extras_eviction(client, monkeypatch):
    """Editing a field in an EXTRAS_EVICTION bucket awaits that bucket's
    evictor; untouched buckets stay quiet; an evictor failure never breaks
    the save (error-swallowing semantics)."""
    from faster_whisper_backend.admin import routes as admin_routes

    calls = []

    def _spy(name, fail=False):
        async def _f():
            calls.append(name)
            if fail:
                raise RuntimeError("boom")
        return _f

    monkeypatch.setitem(admin_routes._EVICTORS, "diarization",
                        _spy("diarization"))
    monkeypatch.setitem(admin_routes._EVICTORS, "bgm", _spy("bgm"))
    r = client.post("/settings/state", json={"DIARIZATION_DEVICE": "cpu"})
    assert r.status_code == 200
    assert calls == ["diarization"]

    # A raising evictor is swallowed — the save still succeeds.
    calls.clear()
    monkeypatch.setitem(admin_routes._EVICTORS, "bgm",
                        _spy("bgm", fail=True))
    r = client.post("/settings/state",
                    json={"BGM_SEPARATION_DEVICE": "cpu"})
    assert r.status_code == 200
    assert calls == ["bgm"]


def test_field_groups_cover_every_setting():
    """The WebUI layout (_FIELD_GROUPS) must list exactly the AdminConfig schema
    fields: no setting silently missing from the form, no stale/typo'd entry.
    Regression guard — adding a config field without wiring its WebUI group (as
    happened with STREAMING_HARD_BREAK_*) should fail here, not ship invisible."""
    from faster_whisper_backend.admin import routes as admin_routes
    from faster_whisper_backend import config_store

    displayed = admin_routes._all_fields()
    assert len(displayed) == len(set(displayed)), "duplicate field in _FIELD_GROUPS"
    schema = set(config_store.AdminConfig.model_fields)
    # Fields intentionally edited on a DEDICATED page, not the /settings form.
    # OVERRIDE_PROFILES has its own master-detail editor on /settings/overrides
    # (served by /settings/overrides/state), so it is not in _FIELD_GROUPS.
    managed_elsewhere = {"OVERRIDE_PROFILES"}
    missing = schema - set(displayed) - managed_elsewhere
    stale = set(displayed) - schema
    assert not missing, f"settings missing from the WebUI layout: {sorted(missing)}"
    assert not stale, f"_FIELD_GROUPS entries that are not config fields: {sorted(stale)}"


# ---------------------------------------------------------------------------
# Translation model editors + template preview / test endpoint
# ---------------------------------------------------------------------------

def test_settings_page_wires_translation_model_editors(client):
    """The model dropdown / multi-select editors are parameterized by a
    source-lists spec; the dispatch wires the translation + separation model
    fields to their OWN allowlists (not the whisper ALLOWED_MODELS)."""
    text = client.get("/settings").text
    # Dispatch: translation fields source from the translation lists…
    # (Asserted by the data wiring, not JS function signatures / local
    # variable names — those are implementation detail, not contract.)
    assert "allowed: 'TRANSLATION_ALLOWED_MODELS'" in text
    assert "preload: 'TRANSLATION_PRELOAD_MODELS'" in text
    # …and the UVR model from the separation allowlist (no preload concept).
    assert "{ allowed: 'BGM_SEPARATION_ALLOWED_MODELS' }" in text
    # The re-render event fires for the new source lists too.
    assert "name === 'TRANSLATION_ALLOWED_MODELS'" in text
    assert "name === 'BGM_SEPARATION_ALLOWED_MODELS'" in text


def test_settings_page_ships_template_editor_and_preview(client):
    text = client.get("/settings").text
    assert "function translationTemplateEditor" in text
    # Client-side preview uses the canned sample + .replace semantics.
    assert "Wir haben die Messung gestern wiederholt." in text
    # The test button posts the UNSAVED textarea value to the new endpoint.
    assert "/settings/translation-test" in text
    # Visibility follows the (unsaved) family select.
    assert "TRANSLATION_PROMPT_FAMILY" in text
    # Env-pinning TRANSLATION_PROMPT_TEMPLATE must not disable the lab's
    # read-only diagnostics (model/from/to/sample/glossary/test button).
    assert "dataset.diagnostic = '1'" in text
    assert "if (c.dataset.diagnostic) return;" in text


_GUARD_WARNING = "segment 1: guard rejected the model output; kept the source text"


def test_translation_test_endpoint_threads_template_override(
        client, app_module, monkeypatch):
    from faster_whisper_backend.audio import translation

    app_module.cfg.TRANSLATION_ENABLED = True
    seen = {}

    async def fake_translate(segments, targets, **kwargs):
        seen["segments"] = segments
        seen["targets"] = targets
        seen["kwargs"] = kwargs
        return ([{"en": "We repeated the measurement yesterday."}],
                [_GUARD_WARNING],
                {"model": "org/model-GGUF:Q4", "source": "de",
                 "mode": "faithful"})

    monkeypatch.setattr(translation, "translate_segments", fake_translate)
    r = client.post("/settings/translation-test", json={
        "text": "Wir haben die Messung gestern wiederholt.",
        "target": "en", "source": "de", "family": "custom",
        "template": "X {text} -> {target_language}"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["output"] == "We repeated the measurement yesterday."
    assert j["model"] == "org/model-GGUF:Q4"
    assert isinstance(j["ms"], int) and j["ms"] >= 0
    # A guard-failed test falls back to the untranslated source — the
    # warnings are the only signal, so they must reach the admin.
    assert j["warnings"] == [_GUARD_WARNING]
    assert seen["segments"] == [
        {"text": "Wir haben die Messung gestern wiederholt."}]
    assert seen["targets"] == ["en"]
    assert seen["kwargs"]["source_lang"] == "de"
    assert seen["kwargs"]["mode"] == "faithful"
    assert seen["kwargs"]["template_override"] == "X {text} -> {target_language}"


def test_translation_test_family_auto_drops_template_for_builtin_family(
        client, app_module, monkeypatch):
    """The lab sends family=null for "auto" but still posts the (saved)
    textarea value. translate_segments treats any non-None template as the
    custom family, so a hunyuan model would have been tested — and its chip
    reported — as "custom". The effective family must decide."""
    from faster_whisper_backend.audio import translation

    app_module.cfg.TRANSLATION_ENABLED = True
    app_module.cfg.TRANSLATION_ALLOWED_MODELS = set()
    app_module.cfg.TRANSLATION_PROMPT_FAMILY = "auto"
    seen = {}

    async def fake_translate(segments, targets, **kwargs):
        seen["kwargs"] = kwargs
        return ([{"en": "Hello"}], [],
                {"model": "tencent/HY-MT1.5-7B-GGUF:Q4_K_M", "source": "de",
                 "mode": "faithful"})

    monkeypatch.setattr(translation, "translate_segments", fake_translate)
    body = {"text": "Hallo", "target": "en", "source": "de",
            "model": "tencent/HY-MT1.5-7B-GGUF:Q4_K_M",
            "template": "X {text} -> {target_language}"}
    # Preview: the rendered family follows the model, not the stale template.
    r = client.post("/settings/translation-test", json={**body, "preview": True})
    assert r.status_code == 200, r.text
    assert r.json()["prompt"]["family"] == "hunyuan"
    # Real run: no template_override reaches translate_segments.
    r = client.post("/settings/translation-test", json=body)
    assert r.status_code == 200, r.text
    assert seen["kwargs"]["template_override"] is None
    assert seen["kwargs"]["family_override"] is None


def test_translation_test_403_when_disabled(client, app_module):
    app_module.cfg.TRANSLATION_ENABLED = False
    r = client.post("/settings/translation-test",
                    json={"text": "hi", "target": "en"})
    assert r.status_code == 403


def test_translation_test_admin_gated(client, app_module, make_user_key):
    from conftest import bearer

    app_module.cfg.TRANSLATION_ENABLED = True
    make_user_key("root", is_admin=True)
    r = client.post("/settings/translation-test",
                    json={"text": "hi", "target": "en"})
    assert r.status_code == 401                       # locked down, no key
    _uid, raw = make_user_key("alice", is_admin=False)
    r = client.post("/settings/translation-test",
                    json={"text": "hi", "target": "en"},
                    headers=bearer(raw))
    assert r.status_code == 403                       # valid key, not admin


def test_translation_test_422_bad_shape(client, app_module):
    app_module.cfg.TRANSLATION_ENABLED = True
    assert client.post("/settings/translation-test",
                       json={"target": "en"}).status_code == 422  # no text
    assert client.post("/settings/translation-test",
                       json={"text": "x" * 2001,
                             "target": "en"}).status_code == 422  # over cap
    assert client.post("/settings/translation-test",
                       json={"text": "hi", "target": "en",
                             "template": "y" * 8001}).status_code == 422


def test_translation_test_translation_error_is_400(
        client, app_module, monkeypatch):
    from faster_whisper_backend.audio import translation

    app_module.cfg.TRANSLATION_ENABLED = True
    pid = "deadbeef"
    seen = {}

    async def boom(*a, **k):
        # The entry must exist while the run is in flight...
        seen["registered"] = pid in app_module._BATCH_PROGRESS
        raise translation.TranslationError(
            "translation dependencies are not installed on this server — "
            "pip install -r requirements-translate.txt")

    monkeypatch.setattr(translation, "translate_segments", boom)
    r = client.post("/settings/translation-test",
                    json={"text": "hi", "target": "en", "progress_id": pid})
    assert r.status_code == 400
    assert "requirements-translate" in r.json()["error"]
    assert seen["registered"] is True
    # ...and the failure path must not strand it, or the lab's poller reads
    # a permanent "starting" stage for a run that already died.
    assert pid not in app_module._BATCH_PROGRESS
    assert pid not in app_module._BATCH_CANCELLED


def test_translation_test_preview_renders_without_model(
        client, app_module, monkeypatch):
    """preview=True returns the rendered prompt and NEVER calls the model."""
    from faster_whisper_backend.audio import translation

    app_module.cfg.TRANSLATION_ENABLED = True

    async def boom(*a, **k):  # pragma: no cover - must not run
        raise AssertionError("translate_segments called on a preview")

    monkeypatch.setattr(translation, "translate_segments", boom)
    r = client.post("/settings/translation-test", json={
        "text": "Hallo Welt", "target": "en", "source": "de",
        "family": "milmmt", "preview": True})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["warnings"] == []
    p = j["prompt"]
    assert p["family"] == "milmmt"
    assert p["chat"] is False
    assert "Hallo Welt" in p["text"]
    assert "German" in p["text"] and "English" in p["text"]


def test_translation_test_preview_family_auto_follows_model(
        client, app_module):
    app_module.cfg.TRANSLATION_ENABLED = True
    app_module.cfg.TRANSLATION_ALLOWED_MODELS = set()
    r = client.post("/settings/translation-test", json={
        "text": "Hallo", "target": "en",
        "model": "tencent/HY-MT1.5-7B-GGUF:Q4_K_M", "preview": True})
    assert r.status_code == 200, r.text
    assert r.json()["prompt"]["family"] == "hunyuan"


def test_translation_test_progress_id_seeds_registry(
        client, app_module, monkeypatch):
    """A progress_id in the test body joins _BATCH_PROGRESS (stage
    'starting' before the run, 'downloading'/'translating' as the hooks
    fire) and the endpoint's finally pops it. Malformed ids are ignored,
    not 422."""
    from faster_whisper_backend.audio import translation

    app_module.cfg.TRANSLATION_ENABLED = True
    pid = "beef" * 8
    seen = {}

    async def fake_translate(segments, targets, **kwargs):
        seen["at_start"] = dict(app_module._BATCH_PROGRESS.get(pid) or {})
        kwargs["download_cb"](512, 1024)
        seen["downloading"] = dict(app_module._BATCH_PROGRESS.get(pid) or {})
        kwargs["progress_cb"](1.0, "en 1/1", None)
        seen["translating"] = dict(app_module._BATCH_PROGRESS.get(pid) or {})
        return ([{"en": "hi"}], [], {"model": "org/m", "source": "de",
                                     "mode": "faithful"})

    monkeypatch.setattr(translation, "translate_segments", fake_translate)
    r = client.post("/settings/translation-test", json={
        "text": "hallo", "target": "en", "progress_id": pid})
    assert r.status_code == 200, r.text
    assert seen["at_start"].get("stage") == "starting"
    assert seen["downloading"].get("stage") == "downloading"
    assert seen["downloading"].get("progress") == 0.5
    assert seen["downloading"].get("total_bytes") == 1024
    assert seen["translating"].get("stage") == "translating"
    assert seen["translating"].get("progress") == 1.0
    assert pid not in app_module._BATCH_PROGRESS       # popped by finally

    # Malformed id: ignored (no registry entry), request still succeeds.
    r = client.post("/settings/translation-test", json={
        "text": "hallo", "target": "en", "progress_id": "NOT-HEX"})
    assert r.status_code == 200, r.text


def test_translation_test_progress_entry_is_owner_stamped(
        client, app_module, monkeypatch, make_user_key):
    """The lab's seed joins _BATCH_PROGRESS with an owner, like the batch and
    stage-ahead seeds in main — an owner-less entry would be readable (and
    cancellable) by ANY authenticated caller holding the id."""
    from faster_whisper_backend.audio import translation
    from conftest import bearer

    app_module.cfg.TRANSLATION_ENABLED = True
    _uid, raw = make_user_key("root", is_admin=True)
    pid = "feed" * 8
    seen = {}

    async def fake_translate(segments, targets, **kwargs):
        seen["owner"] = (app_module._BATCH_PROGRESS.get(pid) or {}).get("owner")
        return ([{"en": "hi"}], [],
                {"model": "org/m", "source": "de", "mode": "faithful"})

    monkeypatch.setattr(translation, "translate_segments", fake_translate)
    r = client.post("/settings/translation-test",
                    json={"text": "hallo", "target": "en", "progress_id": pid},
                    headers=bearer(raw))
    assert r.status_code == 200, r.text
    assert seen["owner"] is not None


def test_translation_test_model_allowlist_gate(client, app_module):
    app_module.cfg.TRANSLATION_ENABLED = True
    app_module.cfg.TRANSLATION_ALLOWED_MODELS = {"org/allowed-GGUF:Q4"}
    app_module.cfg.TRANSLATION_DEFAULT_MODEL = "org/default-GGUF:Q4"
    r = client.post("/settings/translation-test", json={
        "text": "hi", "target": "en", "model": "org/other-GGUF:Q4",
        "preview": True})
    assert r.status_code == 400
    assert "TRANSLATION_ALLOWED_MODELS" in r.json()["error"]
    # The configured default is exempt, like the request path.
    r = client.post("/settings/translation-test", json={
        "text": "hi", "target": "en", "model": "org/default-GGUF:Q4",
        "preview": True})
    assert r.status_code == 200, r.text


def test_translation_test_unknown_family_400(client, app_module):
    app_module.cfg.TRANSLATION_ENABLED = True
    r = client.post("/settings/translation-test", json={
        "text": "hi", "target": "en", "family": "nope", "preview": True})
    assert r.status_code == 400
    assert "family" in r.json()["error"]


def test_translation_test_threads_model_family_glossary(
        client, app_module, monkeypatch):
    from faster_whisper_backend.audio import translation

    app_module.cfg.TRANSLATION_ENABLED = True
    app_module.cfg.TRANSLATION_ALLOWED_MODELS = set()
    seen = {}

    async def fake_translate(segments, targets, **kwargs):
        seen["kwargs"] = kwargs
        return ([{"en": "ok"}], [], {"model": "org/m-GGUF:Q4",
                                     "source": "de", "mode": "faithful"})

    monkeypatch.setattr(translation, "translate_segments", fake_translate)
    r = client.post("/settings/translation-test", json={
        "text": "Hallo", "target": "en", "source": "de",
        "model": "org/m-GGUF:Q4", "family": "hunyuan",
        "glossary": "Messung = measurement",
        "template": "stale {text} {target_language}"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["prompt"]["family"] == "hunyuan"
    assert isinstance(j["cold"], bool)
    assert seen["kwargs"]["model_ref"] == "org/m-GGUF:Q4"
    assert seen["kwargs"]["family_override"] == "hunyuan"
    assert seen["kwargs"]["glossary"] == "Messung = measurement"
    # A stale custom-template textarea must not leak into a built-in family.
    assert seen["kwargs"]["template_override"] is None
