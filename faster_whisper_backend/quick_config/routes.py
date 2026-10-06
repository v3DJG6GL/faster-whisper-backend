"""
End-user simple-config WebUI for faster-whisper-backend.

Mounted at /quick-config. Endpoints:

  GET  /quick-config                      HTML page (loopback / USER_WEBUI_ALLOWED_HOSTS)
  GET  /quick-config/state                Returns ONLY the rules an admin marked exposed
  POST /quick-config/state                Patch enabled / body fields on exposed rules
  POST /quick-config/reapply-rules        Kick off bulk-reapply job over existing captures
  GET  /quick-config/reapply-rules/status Poll bulk-reapply job state
  GET  /quick-config/recent               Snapshot of the recent traces
  POST /quick-config/recent/search        Free-text search over recent traces
  GET  /quick-config/stream               SSE stream of recent traces (live updates)

Security model:
  1. IP gate:           require_user_webui_host (loopback always permitted)
  2. API key:           Depends(get_current_user) — bearer must resolve to
                        an active key. Admin = is_admin=True.
  3. Rule allow-list:   POST enforces `exposed == True` AND a per-type
                        field allow-list, regardless of caller role. Defends
                        against `{"locked": false}`-style bypass attempts —
                        the filter runs BEFORE the merge into PIPELINE_RULES.

The recent traces hold literal dictation snippets, which can be sensitive.
They live in recent_transcriptions_store — a durable SQLite/WAL database at
`cfg.RECENT_TRANSCRIPTIONS_DB`, row-capped by RECENT_TRANSCRIPTIONS_MAX and
aged out by RECENT_TRANSCRIPTIONS_RETENTION_DAYS — so they survive restarts.
Never log trace contents.
"""

from __future__ import annotations

import asyncio
import datetime
import functools
import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from faster_whisper_backend.settings import config as cfg
from faster_whisper_backend.settings import config_store
from faster_whisper_backend.settings import schema as settings_schema
from faster_whisper_backend.settings import version as settings_version
from faster_whisper_backend.core import store_common
from faster_whisper_backend.quick_config import recent_feed as qc_recent_feed
from faster_whisper_backend.stats import recent_transcriptions_store
from faster_whisper_backend.core import web_common
from faster_whisper_backend.pipeline import apply as pl_apply
from faster_whisper_backend.core.web_common import require_user_webui_host
from faster_whisper_backend.auth import dependencies as auth
from faster_whisper_backend.auth.dependencies import get_current_user, require_page
from faster_whisper_backend.core import templates
from faster_whisper_backend.captures import reapply as captures_reapply
from faster_whisper_backend.stats import usage_store

logger = logging.getLogger("whisper-api")

router = APIRouter(prefix="/quick-config")


# Per-type allow-list of fields an end-user is allowed to patch on an
# already-existing exposed rule. Anything else in a patch dict triggers a
# 400. `name`, `label`, `type`, `exposed`, `locked`, `seeded` are NEVER
# editable from /quick-config (admin-only). Adding a new rule, deleting a
# rule, or reordering is also admin-only.

# Dedicated, deliberately tiny pool for the one blocking call that can hold a
# thread for seconds: config_store.save_overrides runs the ReDoS guard, which
# forks a `sys.executable` child and waits up to _GUARD_TIMEOUT (2 s) per
# catastrophic pattern. Running that on asyncio's DEFAULT executor (which
# to_thread uses) lets a caller with only the `quick_config` page scope — no
# admin, no host gate, no rate limit — occupy every one of its
# min(32, cpu_count+4) threads and add seconds of scheduling latency to every
# unrelated to_thread call in the app. Bounding it at 2 confines the damage to
# this endpoint. Legitimate saves take ~25 ms, so queueing at 2 is not
# observable. Created once at import; never per-request (a per-request executor
# would leak threads and defeat the bound).
_GUARDED_SAVE_EXECUTOR = ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="regex-guard",
)

# Ingress cap for a callback:map patch: the schema-derived value web_common
# bakes into the shared editor, so the save path and the editor's "n / cap"
# readout read one copy. The same bound is enforced by Pydantic inside
# save_overrides, but only AFTER the stamping loop below has already walked
# the caller's dict twice on the event loop.
_MAP_MAX_ENTRIES: int = web_common._MAP_MAX_ENTRIES
# Request-wide bound for a patch naming SEVERAL map rules: it caps the
# event-loop work of one POST, and is deliberately NOT the per-dictionary
# cap above — the page sends every dirty map in full in a single request, so
# reusing _MAP_MAX_ENTRIES here made two individually valid dictionaries
# unsaveable together.
_MAP_MAX_TOTAL_ENTRIES: int = 4 * _MAP_MAX_ENTRIES


_PATCH_ALLOWED_FIELDS: dict[str, frozenset[str]] = {
    "regex-list":                  frozenset({"enabled", "entries"}),
    "callback:map":                frozenset({"enabled", "map"}),
    "callback:lowercase-wordlist": frozenset({"enabled", "pattern", "wordlist"}),
    "callback:dedup":              frozenset({"enabled", "pattern"}),
    "callback:upper":              frozenset({"enabled", "pattern"}),
}


# SSE endpoint compatibility: EventSource has no way to attach an
# Authorization header, so the /stream endpoint rides the session cookie
# the browser sends automatically on a same-origin stream.

def require_user_or_admin_sse(request: Request) -> dict[str, Any]:
    """SSE-aware get_current_user + require_page("quick_config") — one line
    over the shared resolver in auth, so this gate can never drift from the
    Depends path (bearer header, then the session cookie; open mode only on
    the admin host allowlist)."""
    return auth.resolve_user_for_page_sse(request, "quick_config")


def _reauth_on_version_change(request: Request, seen_version: int
                              ) -> tuple[dict[str, Any], int] | None:
    """stream_recent helper (the /stats/stream _rescope_on_version_change
    pattern): when settings_version.config_version() moved since `seen_version`
    (revoke / permission edit / logout bump it), re-resolve the caller and
    return the fresh (record, version); None when nothing changed. Raises
    HTTPException when the caller lost access, which ends the stream —
    otherwise a revoked user's open tab kept receiving every new trace in
    its old scope until the browser closed the EventSource."""
    current = settings_version.config_version()
    if current == seen_version:
        return None
    return require_user_or_admin_sse(request), current


class QuickPatchPayload(BaseModel):
    """POST body shape:
        {"rules_patch": {slug: {field: value, ...}, ...},
         "fingerprints": {slug: <hex>, ...}}      # optional, recommended

    `fingerprints` carries the per-rule hash each slug had when the
    client loaded /state. Server compares against the current rule's
    fingerprint and reports a conflict for any mismatch. Without it,
    last-writer-wins (legacy behavior; clients are expected to send it
    now). See _rule_fingerprint()."""
    model_config = {"extra": "forbid"}
    rules_patch: dict[str, dict[str, Any]]
    fingerprints: dict[str, str] | None = None


def _rule_fingerprint(rule: dict[str, Any]) -> str:
    """Stable short hash of a rule dict for optimistic concurrency
    control (HTTP-ETag style). Order-insensitive on dict fields. Uses
    sha1 because it's fast and we don't need cryptographic strength —
    the worst case of a collision is "two different rule states hash
    the same" which is statistically irrelevant at our buffer cap. Cut
    to 12 hex chars (~48 bits) to keep the wire payload small."""
    canonical = json.dumps(
        rule, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Shared view/edit helpers — single home for the pipeline-rule policy, reused
# by the /quick-config WebUI routes below AND the /v1/pipeline-rules client API
# (v1_router, further down). Keeping these in one place means the desktop
# client and the browser behave identically.
# ---------------------------------------------------------------------------

def build_visible_rules(user: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """The pipeline rules `user` may see (canonical + fingerprinted) and their
    role. The visibility policy (Permissions.can_see_rule: rule must be
    `exposed`, then admin OR untagged rule OR rule.tags ∩ caller.tags) lives in
    exactly one place. The terminal sentinel is always excluded. Each returned
    rule carries a `_fp` (computed AFTER pl_apply.canon_rules, so client + server hash
    the same canonical bytes) for optimistic-concurrency on the patch path."""
    perms = user["permissions"]
    canonical = [
        dict(r) for r in pl_apply.canon_rules(list(cfg.PIPELINE_RULES))
        if isinstance(r, dict)
        and r.get("type") != "terminal"
        and perms.can_see_rule(r)
    ]
    for rd in canonical:
        rd["_fp"] = _rule_fingerprint(rd)
    role = "admin" if user.get("is_admin") else "user"
    return canonical, role


def editable_fields_map() -> dict[str, list[str]]:
    """The per-rule-type field allow-list as JSON-serialisable sorted lists, so
    a client can render exactly the editable fields without hardcoding the
    policy. Mirrors _PATCH_ALLOWED_FIELDS (the server-side enforcement)."""
    return {rtype: sorted(fields) for rtype, fields in _PATCH_ALLOWED_FIELDS.items()}


def build_word_suggestions(user: dict[str, Any], *, max_words: int) -> list[str]:
    """Recently-transcribed word + phrase suggestions for `user`, scoped exactly
    like GET /quick-config/recent (own rows unless quick_config scope == "all";
    admins / open-mode see everything). Aggregates each row's tokens ∪ bigrams
    newest-first with a case-insensitive dedup (first-seen = newest casing wins),
    capped at `max_words`. This is the single-source-of-truth Python port of the
    web page's rebuildDatalist() JS, reused by GET /v1/recent-words so the
    desktop Dictionary and the browser autocomplete agree. max_words <= 0 → [].
    Resilient: any store error (e.g. store not yet initialised) → []."""
    if max_words <= 0:
        return []
    perms = user["permissions"]
    sees_all = perms.scope("quick_config") == "all"
    user_filter = None if sees_all else (user.get("user_id") or "")
    scan = int(getattr(cfg, "RECENT_TRANSCRIPTIONS_PAGE_SIZE", 100))
    try:
        rows = recent_transcriptions_store.list_recent(limit=scan, user_id_filter=user_filter)
    except Exception:
        return []
    seen: dict[str, str] = {}
    for row in rows:
        for word in list(row.get("tokens") or []) + list(row.get("bigrams") or []):
            ws = str(word)
            key = ws.lower()
            if key and key not in seen:
                seen[key] = ws
                if len(seen) >= max_words:
                    return list(seen.values())
    return list(seen.values())


# The regex-guard reasons (settings/schema.py's compile/template checks and
# pipeline/regex_guard.validate) after the ordinal collapse below. Their reason
# is the actionable part of a guard error and names nothing but the failure.
_HIDDEN_GUARD_REASON = re.compile(
    r"<hidden rule>: (?:invalid regex|regex test failed|regex took)\b")


def _redact_invisible_slugs(
    errors: list[dict[str, str]],
    user: dict[str, Any],
    rules: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Blank out rule slugs the caller is not allowed to see.

    save_overrides re-validates the ENTIRE merged rule list, and config_store
    builds its messages as `rule {idx} ({slug!r}) ...` regardless of who asked.
    Rule visibility is a real authorization boundary — the GET path filters on
    `can_see_rule` and the PATCH path answers an invisible slug with the same
    400 as an unknown one — so returning
    those messages verbatim to a non-admin leaks the names (and list positions)
    of rules they cannot otherwise learn exist. Two ways that fires: a rule that
    compiles but fails the guard (the load path validates without `guard_regex`,
    so it can sit on disk), or the shared guard budget expiring while the child
    is on a later hidden rule.

    The caller's OWN rules are always visible to them — the PATCH path rejected
    anything else before we got here — so their real errors stay legible.
    Admins see everything unchanged.
    """
    if user.get("is_admin"):
        return errors
    perms = user.get("permissions")
    if perms is None:
        return errors
    # The slug lives under "name" on a rule dict (see by_slug below); _RuleBase
    # forbids extras, so no rule ever carries a "slug" key and keying on one
    # made this a silent no-op.
    # Also collect list POSITIONS of hidden rules: a plain Pydantic field
    # error carries no slug at all, only a dotted loc like
    # `PIPELINE_RULES.7.regex-list...` — the index (and rule type) leak the
    # same existence information the slug substitution below exists to hide.
    hidden: list[str] = []
    hidden_idx: set[int] = set()
    for i, r in enumerate(rules):
        name = r.get("name")
        if name and not perms.can_see_rule(r):
            hidden.append(str(name))
            hidden_idx.add(i)
    if not hidden:
        return errors
    # format_validation_errors returns {"loc": ..., "msg": ...} dicts, not
    # bare strings — redact each field rather than the mapping.
    out: list[dict[str, str]] = []
    for entry in errors:
        red = dict(entry)
        loc = str(red.get("loc") or "")
        m = re.match(r"PIPELINE_RULES\.(\d+)\b", loc)
        if m and int(m.group(1)) in hidden_idx:
            # The whole error is about a rule this user may not see — blank
            # both fields so the page's '<hidden rule>' guard routes it to the
            # generic "contact admin" toast.
            red["loc"] = "<hidden rule>"
            red["msg"] = "<hidden rule>"
            out.append(red)
            continue
        swapped = False
        for key in ("loc", "msg"):
            orig = str(red.get(key) or "")
            val = orig
            for slug in hidden:
                # Only the two slug-bearing forms settings/schema.py emits:
                # `rule {idx} ({slug!r})` and `duplicate rule name '{slug}'`.
                # A bare `'{slug}'` also matched the caller's own map keys,
                # which the collision message echoes in quotes — a key equal
                # to a hidden slug blanked their own error and answered
                # "does this hidden rule exist?" one save at a time. Map
                # keys can hold neither parentheses nor quotes.
                val = val.replace(f"('{slug}')", "('<hidden rule>')")
                val = val.replace(
                    f"rule name '{slug}'", "rule name '<hidden rule>'")
            swapped = swapped or val != orig
            # The schema's guard messages read `rule {idx} ({slug!r}) entry
            # {eidx}: ...` — after the slug swap the ordinal still gives away
            # the hidden rule's list position and entry count. Collapse it,
            # but the '<hidden rule>' sentinel MUST survive: the page's doSave
            # keys on it to route the error to the generic "contact admin"
            # toast instead of showing the admin rule's regex error verbatim.
            val = re.sub(
                r"rule \d+ \('<hidden rule>'\)(?: entry \d+)?",
                "<hidden rule>", val,
            )
            red[key] = val
        # Only the regex-guard forms keep their reason. Anything else that
        # named a hidden rule (map keys colliding when lowercased, "duplicate
        # rule name ... at index N") carries the rule's own content or
        # position in the rest of the message, so it is blanked wholesale.
        if swapped and not _HIDDEN_GUARD_REASON.search(red.get("msg") or ""):
            red["loc"] = "<hidden rule>"
            red["msg"] = "<hidden rule>"
        out.append(red)
    return out


async def apply_rules_patch(
    user: dict[str, Any],
    rules_patch: dict[str, dict[str, Any]],
    fingerprints: dict[str, str] | None = None,
    *,
    client_host: str = "?",
) -> tuple[int, dict[str, Any]]:
    """Validate + apply a per-rule patch to PIPELINE_RULES. Shared by
    POST /quick-config/state and PATCH /v1/pipeline-rules so the edit policy
    lives in one home: per-type field allow-list, the same can_see_rule
    re-check the GET uses, terminal + locked guards, optimistic-concurrency by
    fingerprint, server-owned map_meta stamping, then the full Pydantic
    re-validation (incl. the 2 s ReDoS guard) via save_overrides.

    Returns (status_code, body): 200 on success or conflict-only; 422
    {"errors": [...]} when save validation fails (an error may name a rule the
    user didn't touch — the admin's pipeline is invalid). Raises HTTPException
    for 400 (malformed patch / unknown or invisible slug / disallowed field),
    403 (terminal rule / rule locked by an admin / an admin naming a rule
    outside their view) or 500 (config write failure)."""
    async with pl_apply.rules_lock():
        return await _apply_rules_patch_locked(
            user, rules_patch, fingerprints, client_host=client_host,
        )


async def _apply_rules_patch_locked(
    user: dict[str, Any],
    rules_patch: dict[str, dict[str, Any]],
    fingerprints: dict[str, str] | None = None,
    *,
    client_host: str = "?",
) -> tuple[int, dict[str, Any]]:
    """Body of apply_rules_patch — call only under the shared `pl_apply.rules_lock()`."""
    fingerprints = fingerprints or {}
    if not rules_patch:
        return 200, {
            "saved": [], "conflicts": [],
            "hot_applied": [], "cold_pending": [],
            "env_pinned_ignored": [], "evicted": [],
            "requires_restart": False,
        }

    # Snapshot the current PIPELINE_RULES as plain dicts so we can overlay
    # patches deterministically; the merged list replaces cfg.PIPELINE_RULES
    # verbatim (count + order preserved → terminal rule stays last).
    current_rules: list[dict[str, Any]] = []
    for r in cfg.PIPELINE_RULES:
        if hasattr(r, "model_dump"):
            current_rules.append(r.model_dump())
        else:
            current_rules.append(dict(r))
    by_slug = {r.get("name"): i for i, r in enumerate(current_rules)}
    # Canonicalize for fingerprint comparison — must hash the same shape /state
    # served. pl_apply.canon_rules drops None fields and sorts dict keys.
    canonical_now = {r["name"]: r for r in pl_apply.canon_rules(current_rules)
                     if isinstance(r, dict) and r.get("name")}

    # Bound every map BEFORE walking any of them: the stamping loop and the
    # comprehension in the per-slug loop below both iterate a caller-supplied
    # dict on the event loop, and the only other cap (MapRule.map's
    # max_length) is not reached until save_overrides, which runs after.
    # Anything over the cap was already destined for a 422 there, so only the
    # shape of the rejection changes.
    # User-facing wording: this 400 fires BEFORE save_overrides, so Pydantic's
    # "...at most N items..." 422 (which the page used to translate) is
    # unreachable and this detail lands in the toast verbatim. The
    # per-dictionary check runs first so a single full dictionary gets the
    # actionable message rather than the request-wide one.
    total_map_entries = 0
    for p in rules_patch.values():
        if isinstance(p, dict) and isinstance(p.get("map"), dict):
            if len(p["map"]) > _MAP_MAX_ENTRIES:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    f"this dictionary is full (server cap: {_MAP_MAX_ENTRIES} "
                    "entries) — delete some entries before adding new ones",
                )
            total_map_entries += len(p["map"])
    # Whole-request bound: the per-dictionary cap limits ONE map, but a patch
    # naming several map rules could still walk a multiple of it on the event
    # loop. Bounded against the (larger) request-wide cap, not the per-map one.
    if total_map_entries > _MAP_MAX_TOTAL_ENTRIES:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"too many dictionary entries in one request (server cap: "
            f"{_MAP_MAX_TOTAL_ENTRIES} entries in total)",
        )

    saved: list[str] = []
    conflicts: list[dict[str, Any]] = []
    for slug, patch in rules_patch.items():
        if not isinstance(patch, dict):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"rules_patch['{slug}'] must be an object",
            )
        # Must pass the same can_see_rule check the GET uses, so a user can't
        # PATCH a rule their tag set forbids them from seeing. The exposed +
        # terminal + admin checks all live inside can_see_rule.
        #
        # `by_slug` is built from the FULL rule list, so answering "no such
        # rule" and "a rule you may not see" differently was an existence
        # oracle: a non-admin could enumerate the slugs the admin curated out
        # of their view, one guess per request, with no rate limit. For a
        # non-admin the two collapse into the same 400 — the doctrine
        # auth.Permissions.assert_can_read_row states in its own comment
        # ("404, not 403 — a 403 would confirm the row exists").
        #
        # Rules that ARE exposed to the caller are unaffected: they resolve
        # normally here and keep their precise errors. Admins keep both.
        idx = by_slug.get(slug)
        visible = (
            idx is not None
            and user["permissions"].can_see_rule(current_rules[idx])
        )
        if idx is None or (not visible and not user.get("is_admin")):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"unknown rule slug: '{slug}' (adding rules is not allowed)",
            )
        target = current_rules[idx]
        if not visible:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"rule '{slug}' is not visible to your user",
            )
        rtype = target.get("type", "?")
        if rtype == "terminal":
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                "the terminal rule cannot be edited",
            )
        # `locked` is the admin's freeze switch on /settings — enforced here,
        # not just greyed out in the editor. Admins can still patch (they own
        # the flag and can lift it on /settings).
        if target.get("locked") and not user.get("is_admin"):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"rule '{slug}' is locked and cannot be edited",
            )
        allowed = _PATCH_ALLOWED_FIELDS.get(rtype, frozenset())
        for field in patch.keys():
            if field not in allowed:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    f"field '{field}' is not editable on rule '{slug}' "
                    f"(type={rtype})",
                )
        # Optimistic-concurrency: if the client sent a fingerprint, it must
        # match what's on disk now. Mismatch → another writer changed this rule
        # between load and save; skip + report rather than clobber. Patches
        # without a fingerprint fall through to legacy last-writer-wins.
        client_fp = fingerprints.get(slug)
        if client_fp:
            current = canonical_now.get(slug, {})
            current_fp = _rule_fingerprint(current) if current else None
            if current_fp != client_fp:
                conflicts.append({"slug": slug, "current_fp": current_fp})
                continue
        # Server-owned map_meta: stamp added/value-changed cb:map entries with
        # the current epoch, drop meta for removed keys. Done after the conflict
        # check, before the overlay — so the client can't forge timestamps.
        if rtype == "callback:map" and "map" in patch:
            old_map = target.get("map") or {}
            new_map = patch["map"]
            # rules_patch is typed dict[str, dict[str, Any]], so the per-FIELD
            # value is unconstrained and Pydantic only sees it later, inside
            # save_overrides. A non-dict here reached .items() below as an
            # unhandled AttributeError -> bare 500. Checked on the RAW value:
            # an `or {}` coercion first let every falsy non-dict (None, [],
            # "") slip past this guard into a generic 422 from Pydantic.
            if not isinstance(new_map, dict):
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    f"rules_patch['{slug}']['map'] must be an object",
                )
            # Size was bounded by the pre-pass above, before any map was
            # walked; nothing over _MAP_MAX_ENTRIES reaches this loop.
            if not isinstance(old_map, dict):
                old_map = {}
            meta = dict(target.get("map_meta") or {})
            now = int(time.time())
            for k, v in new_map.items():
                if k not in old_map or old_map[k] != v:
                    meta[k] = now
            target["map_meta"] = {k: meta[k] for k in new_map if k in meta}
        target.update(patch)
        saved.append(slug)

    # If every patch conflicted, skip the save+rebuild — nothing to write.
    if not saved:
        return 200, {
            "saved": [],
            "conflicts": conflicts,
            "hot_applied": [], "cold_pending": [],
            "env_pinned_ignored": [], "evicted": [],
            "requires_restart": False,
        }

    # Hand the merged list to the same save path the admin uses: full Pydantic
    # re-validation, with the 2 s ReDoS guard scoped (guard_slugs) to the rules
    # THIS patch changed. Unscoped, every user save re-probed every admin rule:
    # one pre-existing rule the current guard refuses — saved before a guard
    # tightening, loading fine ever since — 422'd every save of any rule, and a
    # large rule set could burn the shared guard budget on its own. An error
    # may still name an untouched rule (non-guard validation covers the whole
    # merged list) — the client surfaces this gracefully.
    # Off the event loop. The guard runs the candidate patterns in a child
    # process and waits up to _GUARD_TIMEOUT for it, so calling save_overrides
    # inline freezes the whole worker — every HTTP request, SSE stream and
    # WebSocket on it — for the duration, and this endpoint is reachable by a
    # non-admin with no rate limit. admin_routes' rule dry-run already offloads
    # the same hazard the same way.
    # It rides a dedicated 2-thread pool rather than asyncio's default executor
    # so a flood of guard-tripping saves here cannot starve every other
    # to_thread call in the app — see _GUARDED_SAVE_EXECUTOR.
    try:
        written = await asyncio.get_running_loop().run_in_executor(
            _GUARDED_SAVE_EXECUTOR,
            functools.partial(
                config_store.save_overrides, {"PIPELINE_RULES": current_rules},
                guard_slugs=frozenset(saved),
            ),
        )
    except ValidationError as e:
        errs = settings_schema.format_validation_errors(e)
        # Unlike the success line below, this branch used to return with NO
        # log at all — a save could fail for every user while the log showed
        # only interleaved successes. Full (unredacted) detail is fine here:
        # the log is admin-eyes-only. log_safe for the same CR/LF-forgery
        # reason as the success line.
        logger.warning(
            "[pipeline-rules] save validation failed from=%s user=%s admin=%s "
            "patched=%s errors=%s",
            client_host, store_common.log_safe(user.get("username") or "?"),
            user.get("is_admin"), saved, store_common.log_safe(str(errs)),
        )
        return status.HTTP_422_UNPROCESSABLE_CONTENT, {
            "errors": _redact_invisible_slugs(errs, user, current_rules),
        }
    except OSError as e:
        logger.error("[pipeline-rules] save failed: %s", e)
        # Generic client detail — the OSError text embeds the absolute
        # config.local.json path, and this patch endpoint is reachable by a
        # non-admin user (exposed-rule edits). Full detail is logged above.
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "could not save configuration",
        )

    applied = await pl_apply.apply_hot_changes(written)

    logger.info(
        "[pipeline-rules] patch from=%s user=%s admin=%s saved=%s conflicts=%s",
        # Usernames are length-capped but never character-screened, so a bare
        # CR/LF here would forge extra records in the /logs viewer.
        client_host, store_common.log_safe(user.get("username") or "?"),
        user.get("is_admin"),
        saved, [c["slug"] for c in conflicts],
    )

    # Admin-only: captures_store.count() with no argument is the unfiltered
    # "admin / scope=all" query, and a caller holding only the quick_config
    # page has captures scope "none" — the global total is not theirs to see.
    # It is also useless to them: the page fires the silent re-apply job on
    # captures_count > 0, and POST /quick-config/reapply-rules is admin-only,
    # so a non-admin save on a server with any capture ended in a permanent
    # red "Re-apply failed: HTTP 403" strip. Reporting 0 keeps that branch shut
    # and takes the blocking SQLite COUNT(*) off the non-admin path entirely.
    captures_count = 0
    if user.get("is_admin") and getattr(cfg, "CAPTURES_RECORDING_ENABLED", False):
        try:
            from faster_whisper_backend.captures import store as captures_store
            captures_count = await asyncio.to_thread(captures_store.count)
        except Exception as _e:
            logger.warning("[pipeline-rules] capture count lookup failed: %s", _e)

    return 200, {
        "saved": saved,
        "conflicts": conflicts,
        **applied,
        "requires_restart": bool(applied["cold_pending"]),
        "captures_count": captures_count,
    }


# ---------------------------------------------------------------------------
# /v1/pipeline-rules — client API (desktop app). NOT host-gated.
# ---------------------------------------------------------------------------
# Same tag/exposed gating + per-type field allow-list + validation as the
# /quick-config WebUI below (shared build_visible_rules / apply_rules_patch),
# but in the /v1 namespace with NO host allowlist — auth is the per-user API
# key (bearer) plus the quick_config page permission. Mounted always-on in
# main.py alongside /v1/me, so the desktop client can manage rules regardless
# of the ADMIN_UI_ENABLED WebUI switch.
# Gate on the router constructor so every present + future sub-route inherits
# the quick_config page check (the auth.require_page convention — "closes the
# 'forgot to gate this endpoint' hole"). Both routes below are user-tier.
v1_router = APIRouter(
    prefix="/v1",
    dependencies=[Depends(require_page("quick_config"))],
)


@v1_router.get("/pipeline-rules")
async def v1_get_pipeline_rules(
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """The post-processing (pipeline) rules THIS caller may view + edit, with
    the same `exposed` + tag gating as /quick-config. Returns
    `{rules, role, editable_fields}`; each rule carries an `_fp` to echo back on
    PATCH. User-tier auth (any valid key); 403 if the caller's quick_config page
    scope is "none". No host allowlist (unlike the /quick-config WebUI)."""
    rules, role = build_visible_rules(user)
    return {
        "rules": rules,
        "role": role,
        "editable_fields": editable_fields_map(),
        "map_collapse_after": int(getattr(cfg, "QUICK_CONFIG_MAP_COLLAPSE_AFTER", 15)),
        "map_max_entries": _MAP_MAX_ENTRIES,
    }


@v1_router.patch("/pipeline-rules")
async def v1_patch_pipeline_rules(
    payload: QuickPatchPayload,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Apply a per-rule patch — `enabled` + the per-type body fields only
    (regex-list→entries, callback:map→map, lowercase-wordlist→pattern+wordlist,
    dedup/upper→pattern). name/label/type/tags/colour/exposed/locked are never
    editable here, and rules can't be added/removed/reordered (admin-only, on
    the /settings WebUI). Body `{rules_patch:{slug:{field:val}}, fingerprints:
    {slug:fp}}`. 200 `{saved, conflicts, requires_restart, …}`; 422
    `{errors:[…]}` on validation; 400 on a malformed patch or an unknown (or
    invisible) slug, 403 on a terminal or admin-locked rule.
    Identical semantics to POST /quick-config/state (shared apply_rules_patch)."""
    client_host = request.client.host if request.client else "?"
    code, body = await apply_rules_patch(
        user, payload.rules_patch, payload.fingerprints, client_host=client_host,
    )
    return JSONResponse(body, status_code=code)


@v1_router.get("/recent-words")
async def v1_get_recent_words(
    user: dict[str, Any] = Depends(get_current_user),
    limit: int | None = None,
) -> dict[str, Any]:
    """Recently-transcribed word + phrase suggestions for filling a spoken-symbol
    (callback:map) key in the desktop Dictionary — the /v1 analogue of the
    /quick-config autocomplete datalist. Scoped to the caller (own rows unless
    their quick_config scope is "all"). Returns `{words, max}` where `max` is the
    server cap (QUICK_CONFIG_WORD_SUGGESTIONS_MAX; 0 = disabled). Optional
    `?limit=` lets the client request fewer (clamped to the cap). User-tier; 403
    if the caller's quick_config scope is "none". No host allowlist (unlike the
    host-gated /quick-config/recent)."""
    cap = int(getattr(cfg, "QUICK_CONFIG_WORD_SUGGESTIONS_MAX", 200))
    max_words = cap
    if limit is not None:
        # FastAPI already rejected a non-integer `limit` with a 422.
        max_words = max(0, min(cap, limit))
    words = await asyncio.to_thread(
        functools.partial(build_word_suggestions, user, max_words=max_words))
    return {"words": words, "max": cap}


@v1_router.get("/usage")
async def v1_get_my_usage(
    days: int | None = None,
    tz: str | None = None,
    from_: int | None = Query(default=None, alias="from"),
    to: int | None = None,
    all: bool = False,
    with_: str | None = Query(default=None, alias="with"),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """The caller's OWN usage statistics for the desktop app's Statistics
    page, Home tiles and chip readout — the /v1 analogue of the host-gated
    /quick-config 'usage' banner. Lives in the /v1 namespace with NO host
    allowlist, so a remote desktop client isn't 403'd by
    USER_WEBUI_ALLOWED_HOSTS. User-tier auth (any valid key); 403 if the
    caller's quick_config page scope is 'none'.

    STRICTLY self-scoped: reads only the authenticated user's user_id, so no
    admin scope is needed and no other user's numbers are ever returned — even
    an admin sees only their own here (the global view lives on /stats).
    Best-effort: any store failure yields the zeroed document so the client
    just renders an empty page instead of erroring.

    Window (days-since-epoch, reckoned in `tz`): `days` (1..3650, default 30)
    ending today, or an explicit inclusive `from`/`to`, or `all=1` from the
    first day with usage. `with` = comma list of optional stages
    (translating, diarizing, separating, vad); when given, the document is
    recomputed from the per-job rows restricted to jobs that ran ALL of them.
    `tz` is the caller's IANA zone name; days — including 'today' — are
    reckoned in it, and in the server's local zone when it is absent or
    unknown. See usage_store.document for the shape."""
    try:
        w = usage_store.parse_window_params(
            days=days, from_day=from_, to_day=to, all_time=all, with_=with_,
            tz=tz)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    jobs_retention = int(getattr(cfg, "USAGE_JOBS_RETENTION_DAYS", 365) or 0)
    today = datetime.datetime.now(w.tz).date()
    # The zeroed fallback reckons the same window the store would, minus
    # the first-day lookup it cannot make.
    f0, t0 = usage_store.resolve_window(
        today=today, days=w.days, from_day=w.from_day, to_day=w.to_day,
        all_time=w.all_time, first_day=None)
    doc = usage_store.empty_document(
        from_day=f0, to_day=t0, tz=w.tz_name, source=w.source,
        jobs_retention_days=jobs_retention)
    uid = user.get("user_id") or ""
    if uid:
        try:
            # Off the loop: a lifetime scan over the hourly rollups (or the
            # per-job rows under `with`), like the /stats gather.
            doc = await asyncio.to_thread(
                functools.partial(
                    usage_store.document, uid, tz=w.tz, tz_name=w.tz_name,
                    days=w.days, from_day=w.from_day, to_day=w.to_day,
                    all_time=w.all_time, with_stages=w.with_stages,
                    jobs_retention_days=jobs_retention))
        except Exception as _e:
            logger.warning("[usage] document failed: %s", _e)
    # `dom_hours` (the day-of-month × hour grid) rides along: the desktop app's
    # busy panel has a "days" rhythm that reads it, like the stats console.
    return {"username": user.get("username") or "", **doc}


@router.get(
    "",
    # HTML page is host-only — the login modal runs in this page's
    # own JS, so the bearer isn't available on the initial navigation.
    # The per-page permission check lives on each API route below; if
    # the user lacks /quick-config access, the page's first state fetch
    # 403s and the JS renders a "no access" landing.
    dependencies=[Depends(require_user_webui_host)],
)
async def get_quick_config_page() -> HTMLResponse:
    return HTMLResponse(
        web_common.render_page(_QUICK_CONFIG_HTML, current="quick-config"),
        media_type="text/html",
    )


@router.get(
    "/state",
    dependencies=[
        Depends(require_user_webui_host),
        Depends(require_page("quick_config")),
    ],
)
async def get_state(
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Return rules the caller can see on /quick-config.

    Visibility is a triple-AND: (a) the rule must not be the terminal
    sentinel, (b) it must be `exposed=True`, (c) the caller's
    `quick_config_tags` must intersect `rule.tags` (or `rule.tags` must
    be empty = visible-to-all). Admins bypass (b)+(c).

    All three checks live behind `perms.can_see_rule()` (auth/dependencies.py) so the
    policy lives in exactly one place — same pattern as the existing
    page-permission gates."""
    # Visibility + fingerprinting lives in build_visible_rules (shared with
    # GET /v1/pipeline-rules) so the policy has exactly one home.
    canonical, role = build_visible_rules(user)
    # Server-authoritative "this transcription has been reported" set
    # + the user's previously-submitted chip corrections per request_id.
    # The /quick-config page reads both: badges sync from the id list;
    # the chip map seeds form._corrections so re-opening a reported
    # trace shows what was submitted, instead of an empty form. Capped
    # at 100 newest per user. Failure (init_db never ran, etc.) is
    # non-fatal — the client falls back to its localStorage hint and
    # an empty chip map.
    reported_chips: dict[str, dict[str, Any]] = {}
    try:
        from faster_whisper_backend.reports import store as reports_store
        uid = user.get("user_id") or ""
        if uid:
            # Off the loop like every sibling read in this file (_recent_page,
            # the SSE replay, /v1/recent-words): this is a SELECT * returning
            # full transcript rows plus a json.loads per row, and /state is the
            # page's auth probe — re-issued on every save and every
            # optimistic-concurrency conflict.
            my_reports = await asyncio.to_thread(
                reports_store.recent_reports_for_user, uid, limit=100,
            )
            for rep in my_reports:
                rid = rep.get("request_id")
                if not rid:
                    continue
                # Newest-first iteration above; first occurrence wins on
                # duplicate request_id (the upsert keeps a single row
                # per (request_id, user_id), so duplicates are rare).
                if rid not in reported_chips:
                    reported_chips[rid] = {
                        "corrections": rep.get("corrections") or [],
                    }
        # No fallback when uid is empty: reports are per-user; open-mode
        # callers see no reported-badge state at all.
    except Exception:
        pass
    return {
        "rules": canonical,
        "role": role,
        "reported_chips": reported_chips,
        "map_collapse_after": int(getattr(cfg, "QUICK_CONFIG_MAP_COLLAPSE_AFTER", 15)),
        "word_suggestions_max": int(getattr(cfg, "QUICK_CONFIG_WORD_SUGGESTIONS_MAX", 200)),
        # Schema cap on a callback:map's entry count — the page shows a
        # "n / cap" readout so a full dictionary is visible BEFORE a save
        # bounces off the Pydantic max_length.
        "map_max_entries": _MAP_MAX_ENTRIES,
    }


@router.get(
    "/usage",
    dependencies=[
        Depends(require_user_webui_host),
        Depends(require_page("quick_config")),
    ],
)
async def get_my_usage(
    tz_midnight: float | None = None,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """The caller's OWN transcription usage (today + lifetime) for the
    personal banner in the /quick-config subbar. Self-scoped: reads only the
    authenticated user's user_id, so no admin scope is needed and no other
    user's numbers are ever returned. Best-effort — any failure yields zeros
    so the page never breaks.

    'today' resets at the VIEWER's local midnight: the browser passes
    `tz_midnight` (epoch seconds of its local 00:00), and we sum the UTC hours
    since then. When the param is absent/invalid we fall back to the server's
    local day."""
    zero = {"requests": 0, "errors": 0, "words": 0, "audio_s": 0.0}
    uid = user.get("user_id") or ""
    today = dict(zero)
    total = dict(zero)
    if uid:
        try:
            if tz_midnight and tz_midnight > 0:
                start_hour = usage_store.hour_for_ts(float(tz_midnight))
            else:
                start_hour = usage_store.local_day_start_hour()
            total = await asyncio.to_thread(usage_store.totals_for_user, uid)
            today = await asyncio.to_thread(
                usage_store.totals_for_user, uid, start_hour=start_hour,
            )
        except Exception:
            today, total = dict(zero), dict(zero)
    return {
        "username": user.get("username") or "",
        "today": today,
        "total": total,
    }


@router.post(
    "/state",
    dependencies=[
        Depends(require_user_webui_host),
        Depends(require_page("quick_config")),
    ],
)
async def post_state(
    payload: QuickPatchPayload,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Apply a per-rule patch. Rejects:
       - patches against rules that aren't currently exposed
       - patches against terminal rules
       - non-admin patches against rules the admin marked `locked`
       - any field outside the per-type allow-list (including admin-only
         fields like `locked`, `name`, `label`, `exposed`, `seeded`)

    Defense order matters: validate + filter the patch BEFORE merging into
    PIPELINE_RULES, so an end-user can't sneak `locked: false` past the
    Pydantic schema (which accepts `locked` as a real field) by routing it
    through the merge step.
    """
    client_host = request.client.host if request.client else "?"
    code, body = await apply_rules_patch(
        user, payload.rules_patch, payload.fingerprints, client_host=client_host,
    )
    return JSONResponse(body, status_code=code)


# --- Re-apply current pipeline rules to existing captures ------------
#
# Quick-config rules are only baked into a capture's `final` at
# transcription time. After a rule edit, historical captures are
# frozen against the rule set at their time of capture. These two
# endpoints drive a background backfill job that re-runs the current
# pipeline over every capture's raw text, updates `final`, and
# rebuilds affected (unlocked) group transcript snapshots from the
# now-refreshed member text. `corrected_text` and chip corrections
# are admin-authoritative — preserved untouched.

@router.post(
    "/reapply-rules",
    dependencies=[
        Depends(require_user_webui_host),
        Depends(require_page("quick_config")),
    ],
)
async def post_reapply_rules(
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    if not user.get("is_admin"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "admin only")
    if not getattr(cfg, "CAPTURES_RECORDING_ENABLED", False):
        return JSONResponse({"status": "idle", "note": "captures disabled"})
    return JSONResponse(captures_reapply.start())


@router.get(
    "/reapply-rules/status",
    dependencies=[
        Depends(require_user_webui_host),
        Depends(require_page("quick_config")),
    ],
)
async def get_reapply_rules_status(
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    if not user.get("is_admin"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "admin only")
    return JSONResponse(captures_reapply.status())


# --- Recent transcription traces (panel + autocomplete source) -------------
#
# Traces live in stats.recent_transcriptions_store (SQLite); main.py's
# transcribe handler records one row per completed transcription. Both endpoints below are
# token-gated so end-users without a valid token can't enumerate recent
# dictation snippets.

@router.get(
    "/recent",
    dependencies=[
        Depends(require_user_webui_host),
        Depends(require_page("quick_config")),
    ],
)
async def get_recent(
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Newest-first slice of the durable recent-transcriptions store.
    The /stream endpoint replays the same slice on connect; /recent
    additionally supports the "Load older" pagination cursor.

    Query params:
      before_ts (float, optional) — cursor: return rows STRICTLY older
        than this created_ts. Omit / 0 to fetch the freshest slice.
      limit (int, optional)       — clamped to cfg.RECENT_TRANSCRIPTIONS_PAGE_SIZE.

    Response:
      {recent: [...], next_before_ts: <float|null>}
        next_before_ts is the oldest entry's created_ts when the slice
        was fully filled (caller can re-query with that as before_ts);
        null when this is the last batch.

    Scope-aware: scope=own users see only their own rows; scope=all
    users (admins) see every row. The username column is materialized
    at write so the read path doesn't hit api_keys_store.

    Free-text search lives on POST /quick-config/recent/search, NOT here:
    the term is by construction words out of a dictation, and a query string
    is copied verbatim into uvicorn's access log, the reverse proxy's access
    log and the browser's history — none of which are the 0600 log file the
    transcript text is otherwise confined to."""
    perms = user["permissions"]
    caller_uid = user.get("user_id") or ""
    sees_all = perms.scope("quick_config") == "all"

    page_size = int(getattr(cfg, "RECENT_TRANSCRIPTIONS_PAGE_SIZE", 100))
    try:
        q_before = float(request.query_params.get("before_ts", "") or 0.0)
    except (TypeError, ValueError):
        q_before = 0.0
    try:
        q_limit = int(request.query_params.get("limit", "") or page_size)
    except (TypeError, ValueError):
        q_limit = page_size
    q_limit = max(1, min(q_limit, page_size))

    return await _recent_page(
        before_ts=q_before, limit=q_limit,
        query=None, sees_all=sees_all, caller_uid=caller_uid,
    )


async def _recent_page(
    *,
    before_ts: float,
    limit: int,
    query: str | None,
    sees_all: bool,
    caller_uid: str,
) -> dict[str, Any]:
    """Shared body of GET /recent and POST /recent/search — identical scoping
    and pagination, the two differ only in where the search term travels.

    Off the loop: a search is an unindexable LIKE '%needle%' over two TEXT
    columns capped at 50 000 chars each, so SQLite scans every retained row.
    Measured ~30 ms per search at the shipped RECENT_TRANSCRIPTIONS_MAX of 500,
    unthrottled, and .env.example documents 0 = unbounded."""
    traces = await asyncio.to_thread(
        functools.partial(
            recent_transcriptions_store.list_recent,
            before_ts=before_ts if before_ts > 0 else None,
            limit=limit,
            user_id_filter=None if sees_all else caller_uid,
            query=query or None,
        )
    )
    next_before = traces[-1]["created_ts"] if len(traces) >= limit else None
    return {"recent": traces, "next_before_ts": next_before}


class RecentSearchIn(BaseModel):
    model_config = {"extra": "forbid"}
    q: str = Field(default="", max_length=512)
    before_ts: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    limit: int | None = Field(default=None, ge=1, le=1000)


@router.post(
    "/recent/search",
    dependencies=[
        Depends(require_user_webui_host),
        Depends(require_page("quick_config")),
    ],
)
async def post_recent_search(
    payload: RecentSearchIn,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Same slice as GET /recent, with the free-text filter supplied in the
    BODY. A POST purely to keep dictation words out of every access log and
    out of browser history — the scoping, bounds and response shape are the
    GET's, unchanged."""
    perms = user["permissions"]
    page_size = int(getattr(cfg, "RECENT_TRANSCRIPTIONS_PAGE_SIZE", 100))
    limit = max(1, min(payload.limit or page_size, page_size))
    return await _recent_page(
        before_ts=payload.before_ts,
        limit=limit,
        query=(payload.q or "").strip() or None,
        sees_all=perms.scope("quick_config") == "all",
        caller_uid=user.get("user_id") or "",
    )


@router.get(
    "/stream",
    dependencies=[Depends(require_user_webui_host)],
)
async def stream_recent(
    request: Request,
    user: dict[str, Any] = Depends(require_user_or_admin_sse),
) -> StreamingResponse:
    """Server-sent events stream of recent transcriptions.

    On connect, replays the current buffer (`event: trace` for each entry,
    oldest first). After the replay, pushes any new transcription as
    another `event: trace`. Sends a `: keepalive` SSE comment line every
    15 s so reverse proxies don't kill an idle connection.

    Scope-aware: `scope=own` filters replay AND live items to the
    caller's user_id; `scope=all` lets everything through. Live items
    flow through qc_recent_feed.subscribe()'s shared queue — every
    subscriber gets every event — so filtering happens here per
    subscriber rather than at the publisher (no extra queue infra)."""
    perms = user["permissions"]
    caller_uid = user.get("user_id") or ""
    sees_all = perms.scope("quick_config") == "all"
    seen = settings_version.config_version()

    def _visible(entry: dict[str, Any] | None) -> bool:
        if sees_all:
            return True
        # caller_uid is already coerced from None to "" above; coerce the
        # entry side too so a persisted row with user_id=NULL doesn't get
        # silently excluded for a caller whose own user_id is missing.
        return bool(entry) and (entry.get("user_id") or "") == caller_uid

    async def _rescope() -> bool:
        """Re-resolve the caller when the config version moved; False when
        they lost access (the stream ends)."""
        nonlocal caller_uid, sees_all, seen
        try:
            # Off the loop: config_version() + the re-resolve hit SQLite.
            fresh = await asyncio.to_thread(
                _reauth_on_version_change, request, seen)
        except HTTPException:
            return False
        if fresh is not None:
            rec, seen = fresh
            caller_uid = rec.get("user_id") or ""
            sees_all = rec["permissions"].scope("quick_config") == "all"
        return True

    async def gen():
        q = qc_recent_feed.subscribe()
        try:
            # Replay the freshest page from the durable store (oldest-
            # first so the client receives them in chronological order,
            # matching the prior in-memory deque iteration semantics).
            page_size = int(getattr(cfg, "RECENT_TRANSCRIPTIONS_PAGE_SIZE", 100))
            # Off the loop with its siblings — this runs once per SSE connect.
            replay = await asyncio.to_thread(
                functools.partial(
                    recent_transcriptions_store.list_recent,
                    limit=page_size,
                    user_id_filter=None if sees_all else caller_uid,
                )
            )
            for entry in reversed(replay):
                if _visible(entry):
                    yield f"event: trace\ndata: {json.dumps(entry)}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                if not await _rescope():
                    break
                try:
                    item = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                # Again after the wait: a revoke or a downgrade to own that
                # landed during the up-to-15 s q.get() must apply to the item
                # it woke up for, not only to the next one.
                if not await _rescope():
                    break
                ev = item.get("event", "trace")
                payload = item.get("data") or {}
                if ev == "trace" and not _visible(payload):
                    continue
                yield f"event: {ev}\ndata: {json.dumps(payload)}\n\n"
        finally:
            qc_recent_feed.unsubscribe(q)

    return web_common.sse_response(gen())


_QUICK_CONFIG_HTML = templates.load(__file__, "quick_config.html")
