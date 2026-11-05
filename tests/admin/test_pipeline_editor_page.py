"""Source pins for the /settings/pipeline editor script (inline JS), plus
node-gated tests at the end that run the real _sansRevs / _setRulesDirty /
_postFactory. The five sync dialogs were checked once in headless Chromium
against a local list holding an edited rule, a not-edited-but-outdated rule
and a missing config.json rule, no JS errors.
"""

import json

import pytest

from faster_whisper_backend.settings import config_store as cs
from faster_whisper_backend.settings import schema as settings_schema
from tests._page_helpers import TOP_CLOSE, js_body, js_upto, needs_node, node_json


@pytest.fixture
def html(client):
    r = client.get("/settings/pipeline")
    assert r.status_code == 200
    return r.text


def test_every_sync_action_goes_through_the_review_dialog(html):
    """Promote showed a diff without naming its sides; reset and reset-all
    showed none (reset-all was a browser confirm). One labelled dialog now
    serves promote, reset / update, add, promote all and reset all."""
    for fn in ("_promoteOne", "_promoteAll", "_resetOne", "_resetAll", "_addFromConfig"):
        body = js_body(html, f"function {fn}(")
        assert "_reviewDialog({" in body, fn
    # Both sides are named, and the left column is what gets overwritten.
    assert "const _RV_SERVER = { t: 'This server'" in html
    assert "const _RV_CONFIG = { t: 'config.json'" in html
    assert "'gets changed'" in html
    # Colour is never the only signal.
    assert "cell('o', '−', r.o)" in html and "cell('n', '+', r.n)" in html
    # No native confirm() left on the reset path.
    assert "Discard every local edit and restore all rules" not in html


def test_diff_is_per_map_key_and_per_list_entry(html):
    assert "add(k + ' · ' + JSON.stringify(key), js(o[key]), js(n[key]));" in html
    # A key-order-only map difference is no change (the server re-sorts map
    # keys): no spurious "moved" row next to the real ones.
    assert "'same entries, different order'" not in html


def test_sync_actions_are_buttons_above_the_list(html):
    assert "syncBar.appendChild(promoteAllBtn);" in html
    assert "syncBar.appendChild(resetAllBtn);" in html
    assert "resetAllBtn.className = 'sync-btn down';" in html
    assert "promoteAllBtn.className = 'sync-btn up';" in html
    assert "reset.className = 'sync-btn sm down row-reset';" in html
    assert "'↺ reset to default'" not in html
    assert ".sync-notice[hidden] { display: none; }" in html


def test_new_config_json_rules_get_a_notice(html):
    """A saved local rule list replaces the factory list, so a rule that an
    update adds to config.json stays invisible without this notice."""
    assert "function _missingFactoryRules()" in html
    assert "'↓ Review and add'" in html


def test_promote_all_never_removes_a_config_json_rule_silently(html):
    """A config.json rule missing here used to be dropped by "Promote all" as
    "deleted locally" — it may just be new. Removal is an explicit tick."""
    assert "tick to REMOVE it from config.json" in html
    assert "_buildFactoryPayload(promote, true).filter(r => !remove.has(r.name))" in html


def test_promote_all_removal_ids_cannot_collide_with_a_slug(html):
    """Removal groups were '-' + name and split on i[0] === '-', but RuleSlug
    allows a leading '-': ticking a hand-edited rule "-foo" promoted nothing
    and silently deleted config.json's "foo"."""
    body = js_body(html, "function _promoteAll(")
    assert "id: 'rm:' + b.name," in body
    assert "ids.filter(i => i.startsWith('rm:')).map(i => i.slice(3))" in body
    assert "i[0] === '-'" not in body and "i[0] !== '-'" not in body


def test_outdated_rules_are_told_apart_from_edited_ones(html):
    assert "return _ruleHash(rule) === rev ? 'behind' : 'diverged';" in html
    assert "'◇ config.json is newer'" in html
    # 'behind' has no edits of its own: promoting it would write OLD content.
    assert "(st === 'edited' || st === 'diverged' || st === 'local-only')" in html


def test_promote_all_leaves_diverged_rules_unticked(html):
    """A diverged rule's config.json side is newer than the base of the local
    edit; promote-all ticked it by default, so one click replaced that update
    with no warning (the single-rule promote warns). It is now opt-in."""
    body = js_body(html, "function _promoteAll(")
    assert "checked: !diverged && !ambiguous," in body
    assert "checked: true," not in body.split("_missingFactoryRules()")[0]
    assert "start'\n          + ' unticked: ticking one replaces those newer config.json changes'" in body
    # A rev-less 'edited' rule may be either case (_ruleStatus cannot tell),
    # so it starts unticked too and counts into the same warning.
    assert "const ambiguous = st === 'edited' && !r.config_rev;" in body
    assert "if (diverged || ambiguous) nDiverged++;" in body
    assert "(or have no sync'\n          + ' record)" in body


def _rule(**kw):
    return {"name": "r", "label": "R", "type": "regex-list",
            "entries": [{"pattern": "a", "replacement": "b"}], **kw}


def test_config_rev_is_stored_and_forgiving():
    ok = settings_schema.AdminConfig.model_validate({"PIPELINE_RULES": [_rule(config_rev="09efe7b7acfad2")]})
    assert ok.model_dump(exclude_none=True)["PIPELINE_RULES"][0]["config_rev"] == "09efe7b7acfad2"
    # A malformed value must never fail validation (that drops ALL overrides).
    bad = settings_schema.AdminConfig.model_validate({"PIPELINE_RULES": [_rule(config_rev="not hex!")]})
    assert "config_rev" not in bad.model_dump(exclude_none=True)["PIPELINE_RULES"][0]


def test_config_rev_never_reaches_config_json(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"BEAM_SIZE": 5}', encoding="utf-8")
    term = {"name": "trim-edges", "label": "Trim", "type": "terminal"}
    out = cs.save_factory_rules([_rule(config_rev="09efe7b7acfad2"), term], str(path))
    assert all("config_rev" not in r for r in out)
    assert "config_rev" not in path.read_text(encoding="utf-8")


def test_language_picker_change_never_rebuilds_the_rows(html):
    """commitFull() -> paintAll() wipes every row, which destroyed the
    language picker and its open multi-select dropdown on the first pick
    (and leaked the popover's scroll/resize listeners)."""
    body = js_upto(html, "_renderLanguagePicker({", "});")
    assert "commitData();" in body
    assert "commitFull();" not in body


def test_config_revs_are_stamped_before_the_first_edit(html):
    """Every commit runs AFTER the rule was mutated, so without a stamp at
    construction a rule's first edit never gets a config_rev and the
    'diverged' state (plus its overwrite warning) is unreachable for it."""
    tail = html[html.rindex("function _afterPromoteAll("):]
    assert "  _stampConfigRevs();\n  paintAll();\n  return wrap;" in tail


def test_promote_keeps_an_absent_config_json_rule_in_place(html):
    """A config.json rule missing here (a new es-punctuation in the middle of
    the list) was re-appended at the END of the payload: a silent pipeline
    reorder for every other deployment. It is spliced in before the next
    config.json rule that exists here instead — the NEAREST one in config.json
    order: after a local reorder (config A,B,C,D; local D,A,C) the first local
    rule that is any successor (D) put B before its own predecessor A."""
    body = js_body(html, "function _buildFactoryPayload(")
    assert "if (!out.some(o => o.name === b.name)) out.push(" not in body
    assert "after.indexOf(o.name) !== -1" not in body
    assert "for (const n of after) {" in body
    assert "at = out.findIndex(o => o.name === n);" in body
    assert "out.splice(at, 0, JSON.parse(JSON.stringify(b)));" in body
    # _addFromConfig's nextLocal mirrors it (insert position + dialog meta).
    nl = js_upto(html, "const nextLocal = (b) => {", "\n    };\n")
    assert "after.indexOf(r.name) !== -1" not in nl
    assert "for (const n of after) {" in nl


def test_commit_compares_rules_without_config_rev(html):
    """_stampConfigRevs adds config_rev to every in-sync rule, but a never-saved
    list's server value (canonical config.json) has none, so an edit undone by
    hand kept Save lit — and a Save pinned the server to a local copy."""
    assert "delete c.config_rev;" in html
    for fn in ("function commitData() {", "function commitFull() {"):
        body = js_body(html, fn)
        assert "_setRulesDirty();" in body, fn
        assert "setDirty(name, JSON.parse(JSON.stringify(rules)));" not in body, fn
    helper = js_body(html, "function _setRulesDirty() {")
    assert "_sansRevs(rules, keep) === _sansRevs(cur, keep)" in helper
    # A rev-only change against a server rule that carries a config_rev (the
    # _afterPromoteStamp restamp) is real: only revs the server copy lacks
    # are stripped, so the dirty entry survives a later no-op commit.
    assert "const keep = new Map((cur || []).map(r => [r && r.name, r && r.config_rev]));" in helper
    sans = js_body(html, "function _sansRevs(v, keep) {")
    assert "if (!(keep && keep.get(r && r.name))) delete c.config_rev;" in sans


def test_entry_diff_pairs_duplicate_labels_separately(html):
    """Entry labels are free text: two entries sharing one collapsed onto a
    single Map key, so the review dialog hid the earlier one's change."""
    assert "const idOf = (e, i) => (e.label ? 'label:' + e.label : 'pos:' + i);" not in html
    assert "return 'label:' + e.label + (c === 1 ? '' : '#' + c);" in html
    # A fresh counter per side: the Nth duplicate pairs with the Nth.
    assert "const oid = mkId(), nid = mkId();" in html


def test_single_rule_dialog_with_no_rows_cannot_be_confirmed(html):
    assert "const nothing = single && !o.groups[0].rows.length;" in html
    assert "ok.disabled = sel.length === 0 || nothing;" in html
    assert "No differences — both sides are already identical." in html


def test_down_actions_reread_config_json_first(html):
    """config.json can move under an open page; reset / update / add acted on
    the copy captured at page load while every promote re-fetched it."""
    for fn in ("_resetOne", "_resetAll", "_addFromConfig"):
        body = html[html.index(f"async function {fn}("):]
        assert body.index("await _refreshFactory()") < body.index("_reviewDialog({"), fn


def test_sync_dialog_leftovers_are_gone(html):
    for dead in ("_changeListEl", "_baselineList", "const selected = ", "allowEmpty",
                 "⇪ Promote order", "wrap.appendChild(ctrls);\n  wrap.appendChild(ctrls);"):
        assert dead not in html, dead


def test_language_badge_mentions_the_unknown_language_case(client, html):
    """pipeline.engine._postprocess_text skips a language-scoped rule only
    when a language IS known; with none detected the rule runs."""
    assert "Only runs when the detected language is" not in html
    assert "(or when the language is unknown)" in html
    ov = client.get("/settings/overrides")
    assert ov.status_code == 200
    assert "Only runs when the detected language is" not in ov.text
    assert "(or when the language is unknown)" in ov.text


def test_drop_local_list_after_promote_counts_unpromoted_edits(html):
    """Promote-all lets the admin untick edited rules and an order-promote
    writes no content, so "config.json now holds your rules" with Drop as
    the primary button discarded those local edits (and brought back
    unticked removals). The safe-check covers both, not only local-only."""
    body = js_body(html, "function _afterPromoteAll(")
    assert "const orphaned = _localOnlyNames();" in body
    assert "st === 'edited' || st === 'diverged'" in body
    assert "_missingFactoryRules()" in body
    assert "buttons: lossy" in body


def test_reset_on_a_rule_config_json_dropped_says_so(html):
    """↓ Reset on a rule config.json dropped since the paint returned
    silently, leaving a dead button and no × delete until a full repaint."""
    body = js_body(html, "async function _resetOne(")
    assert "if (!base) return;" not in body
    assert "paintAll();" in body and "_toast(" in body


def test_rule_comparisons_fill_in_schema_defaults(html):
    """The server copy (model_dump) always carries note "", tags [] etc.;
    the editor's serializers can leave an empty one out. A rule missing a
    default read as "edited here" (and kept Save lit, and flipped the
    promote-all drop dialog to "lossy"), so both comparisons fill them in,
    and the add dialog seeds them like the server dump."""
    strip = js_body(html, "function _ruleStrip(r) {")
    assert "const c = _withDefaults(r);" in strip
    sans = js_body(html, "function _sansRevs(v, keep) {")
    assert "const c = _withDefaults(r);" in sans
    assert "return _sortDeep(c);" in sans
    fill = js_body(html, "function _withDefaults(r) {")
    for key in ("languages: []", "exposed: false", "tags: []", "note: ''",
                "color: ''", "label: ''"):
        assert key in fill, key
    add = js_upto(html, "const newRule = {", "};")
    for key in ("languages: []", "exposed: false", "tags: []", "note: ''",
                "color: ''"):
        assert key in add, key


def test_regex_list_entries_are_read_back_in_the_schema_shape(html):
    """The server copy of a regex-list entry always carries label and note
    ("" when empty). The row reader dropped an empty one, so typing into a
    note and deleting it again read the rule as "edited here"."""
    body = js_upto(html, "function _readEntries(parent) {", TOP_CLOSE)
    assert "if (lbl) o.label = lbl;" not in body
    assert "if (note) o.note = note;" not in body
    assert "label: lbl, note: note" in body


def test_promote_paths_share_the_factory_refresh(html):
    """The three promote paths hand-copied _refreshFactory's fetch / alert /
    assign / repaint; one copy drifting would change only that path."""
    for fn in ("_promoteOne", "_promoteAll", "_promoteOrder"):
        body = js_body(html, f"async function {fn}(")
        assert "if (!(await _refreshFactory())) return;" in body, fn
        assert "_fetchFactory()" not in body, fn


def test_promote_all_label_names_removals(html):
    """A ticked 'rm:' group deletes that rule from config.json, yet the
    confirm read '↑ Promote 1 selected rule'."""
    assert "ok.textContent = o.confirm(sel.length, sel);" in html
    body = js_body(html, "async function _promoteAll(")
    assert "const rm = sel.filter(g => g.id.startsWith('rm:')).length;" in body
    assert "' from config.json'" in body


def test_unshadowed_promote_syncs_the_running_value(html):
    """With no config.local.json list, a promote makes the promoted list the
    running PIPELINE_RULES, but fieldDef kept the old config.json: Save stayed
    lit (_afterPromoteStamp even re-set it raw) and the "press Save as well"
    note led to a full local copy shadowing every later config.json update."""
    post = js_body(html, "async function _postFactory(arr) {")
    unsh = post[post.index("if (!out.shadowed_by_local && !out.env_pinned) {"):]
    assert "fd.value = JSON.parse(JSON.stringify(out.rules));" in unsh
    assert "_setRulesDirty();" in unsh
    stamp = js_body(html, "function _afterPromoteStamp() {")
    assert "_setRulesDirty();" in stamp
    assert "setDirty(name," not in stamp
    note = js_body(html, "function _unsavedNoteEl() {")
    assert "fd.provenance !== 'local.json'" in note


def test_single_rule_promote_discloses_an_order_change(html):
    """_promoteOne's payload walks the CURRENT order, so a local reorder is
    written into config.json too; its dialog now says so like Promote all."""
    body = js_body(html, "async function _promoteOne(rule) {")
    assert "extraEls: [_seededOrderDirty() ? _orderChangeEl() : null]" in body
    assert "is written as well." in body


def test_label_reslug_is_gated_on_config_json_membership(html):
    """A rule promoted from here keeps seeded:false locally; a label edit
    re-slugged it and detached it from its config.json copy."""
    assert "const slugFixed = r => r.seeded || _factoryHas(r.name) ||" in html
    assert "if (!slugFixed(rule)) {" in html
    assert "labelLbl.textContent = slugFixed(rule)" in html
    assert "if (!rule.seeded) {" not in html


def test_new_slugs_are_unique_against_config_json_too(html):
    """A custom rule could take the name of a config.json rule this server's
    list predates: it then read as "edited here" against an unrelated rule
    and a promote overwrote the config.json one."""
    assert ("const slugSet = new Set(rules.map(r => r.name)"
            ".concat(factoryRules.map(b => b.name)));") in html
    assert ".concat(factoryRules.map(b => b.name)));\n" in html[
        html.index("if (!slugFixed(rule)) {"):]


def _slice(html, start, end):
    return html[html.index(start):html.index(end)]


# The page globals _postFactory touches; the driver defines `name` and `term`
# before them and its own `rules` / `api` after.
_POSTFACTORY_STUBS = r"""const fields = { PIPELINE_RULES: { value: [term], default_value: [term] } };
const fieldDef = n => fields[n];
const dirty = {};
const setDirty = (n, v) => { dirty[n] = v; };
const paintAll = () => {};
const alert = m => { throw new Error(m); };
let factoryRules = [term];
"""


def _run_postfactory(html, drive, extra_fns=""):
    """Run `drive` under node after the page's real _sortDeep / _withDefaults /
    _sansRevs / _postFactory (+ `extra_fns`); return its stdout as JSON."""
    fns = (_slice(html, "function _sortDeep(o) {", "  // The server copy (schema model_dump)")
           + _slice(html, "function _withDefaults(r) {", "  // Content equality ignoring")
           + _slice(html, "function _sansRevs(v, keep) {", "function commitData() {")
           + _slice(html, "async function _postFactory(arr) {", "async function _fetchFactory() {")
           + extra_fns)
    return node_json(fns + drive)


def _promote_out(html, editor_rule, served, extra_out="", extra_fns=""):
    """_postFactory under node with `editor_rule` + served's terminal in the
    editor and `served` as the server's reply; return {clean, + extra_out's keys}."""
    drive = ("const name = 'PIPELINE_RULES';\nconst term = %s;\n" % json.dumps(served[1])
             + _POSTFACTORY_STUBS + "let rules = [%s, term];\n" % json.dumps(editor_rule)
             + "const api = async () => ({ ok: true, status: 200,"
               " json: async () => ({ rules: %s }) });\n" % json.dumps(served)
             + "(async () => {\n  await _postFactory(rules);\n"
               "  console.log(JSON.stringify({ clean: dirty[name] === fields[name].value,"
               " %s }));\n})();\n" % extra_out)
    return _run_postfactory(html, drive, extra_fns)


@needs_node
def test_unshadowed_promote_of_a_new_rule_leaves_save_unlit(html):
    """A new custom rule is seeded:false in the editor; the server writes it
    to config.json as seeded:true and _postFactory synced fieldDef to that,
    but _sansRevs keeps `seeded`, so Save stayed lit after the promote — and
    that Save pinned a full local copy shadowing later config.json updates.
    The real _sansRevs / _setRulesDirty / _postFactory run under node."""
    rule = {"name": "my-rule", "type": "regex", "pattern": "a", "replacement": "b",
            "seeded": False}
    term = {"name": "terminal", "type": "terminal", "seeded": True}
    assert _promote_out(html, rule, [dict(rule, seeded=True), term],
                        "seeded: rules[0].seeded") == {"clean": True, "seeded": True}


@needs_node
def test_unshadowed_promote_of_a_new_map_rule_leaves_save_unlit(html):
    """The server's canonical MapRule always carries `map_meta: {}` (a
    default_factory field survives exclude_none); the '+ add' dialog builds a
    callback:map rule without it, so _sansRevs saw a diff and Save stayed lit
    after the promote — a Save that pinned a full local copy of the rules."""
    from faster_whisper_backend.pipeline.apply import canon_rules

    new_rule = {"name": "my-map", "label": "My map", "type": "callback:map",
                "enabled": True, "languages": [], "locked": False, "seeded": False,
                "exposed": False, "tags": [], "note": "", "color": "", "map": {"foo": "bar"}}
    served = canon_rules([dict(new_rule, seeded=True),
                          {"name": "terminal", "type": "terminal", "seeded": True}])
    assert served[0]["map_meta"] == {}
    strip = _slice(html, "function _ruleStrip(r) {", "function _ruleContentEqual(a, b) {")
    assert _promote_out(html, new_rule, served, "hash_free: !('map_meta' in _ruleStrip(rules[0]))",
                        strip) == {"clean": True, "hash_free": True}


@needs_node
def test_unshadowed_promote_after_a_map_key_delete_leaves_save_unlit(html):
    """The map editor's row delete leaves the key's map_meta timestamp behind;
    the server's MapRule validator drops it, so after the promote fieldDef
    (synced to the server copy) and the editor differed in map_meta only and
    Save stayed lit under a 'same as config.json' badge — a Save that pinned
    a full local copy of the rules."""
    from faster_whisper_backend.pipeline.apply import canon_rules

    edited = {"name": "my-map", "label": "My map", "type": "callback:map",
              "enabled": True, "languages": [], "locked": False, "seeded": True,
              "exposed": False, "tags": [], "note": "", "color": "", "map": {"foo": "bar"},
              "map_meta": {"foo": 1700000000, "komma": 1700000001}}
    served = canon_rules([edited, {"name": "terminal", "type": "terminal", "seeded": True}])
    assert served[0]["map_meta"] == {"foo": 1700000000}
    assert _promote_out(html, edited, served) == {"clean": True}
