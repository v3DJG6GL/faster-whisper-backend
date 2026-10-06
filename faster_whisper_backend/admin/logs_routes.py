"""The /logs live log viewer (page shell, SSE tail, "Load older" paging
over the rotation chain) and the /sev severity-pill endpoint. The page
template is templates/log_viewer.html.
"""
import asyncio
import io
import os

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from faster_whisper_backend.auth import dependencies as auth_dependencies
from faster_whisper_backend.auth.dependencies import get_current_user as _get_current_user_dep
from faster_whisper_backend.auth.hosts import require_user_webui_host
from faster_whisper_backend.core import templates
from faster_whisper_backend.settings import config as cfg
from faster_whisper_backend.settings import version as settings_version
from faster_whisper_backend.core import web_common

router = APIRouter()


# =============================================================================
# /logs - live log viewer
# =============================================================================
# A self-contained dark-theme log tailer. Loads recent context from the log
# file, then streams new lines via Server-Sent Events. Color is reapplied
# client-side based on content (since we strip ANSI before writing the file).


def _rotated_chain(active_path: str) -> list[str]:
    """Newest→oldest list of paths in the rotation chain: the active log
    followed by .1, .2, … up to LOG_BACKUP_COUNT. Files that don't exist
    are silently skipped — rotation may have produced fewer backups than
    the configured count, and a freshly-deployed service starts with
    just the active file."""
    out = [active_path]
    for i in range(1, int(getattr(cfg, "LOG_BACKUP_COUNT", 10)) + 1):
        p = f"{active_path}.{i}"
        if os.path.exists(p):
            out.append(p)
    return out


# How deep the "Load older" cursor may go, in pages of LOG_VIEWER_INITIAL_LINES.
# Past this the reader walks the whole chain for a window no browser is still
# holding, so paging ends there (next_skip=null) rather than being served.
_LOG_OLDER_MAX_PAGES = 500


def _read_chain_window(active_path: str, skip: int, want: int) -> "tuple[list[str], int | None]":
    """Read up to `want` lines from the rotation chain (newest→oldest),
    starting `skip` lines back from the chain head. Returns
    (lines_oldest_first, next_skip).

    next_skip is None when the chain has no more older content — either
    we returned a partial page (fewer than `want` lines) or we exactly
    reached the chain tail. Otherwise next_skip == skip + len(lines)
    and the caller may re-call with that value to fetch the next
    older window.

    Walks the chain newest-file first (active log → .1 → .2 → …),
    reading each file backward in 8 KB blocks until we've accumulated
    `skip + want` lines across the chain. One file is held in memory
    at a time; ~10 MB worst case for the default LOG_MAX_BYTES."""
    target = skip + want
    # `collected` is built in oldest→newest order: each older file's
    # tail is prepended to the running list as we walk the chain
    # newest→oldest. By construction the OLDEST line in the chain
    # window we've seen so far sits at collected[0].
    collected: list[str] = []
    chain = _rotated_chain(active_path)
    # When we break out of the file loop because `target` is satisfied
    # without opening every file, older rotated files still on disk
    # remain unread — `exhausted` must account for that or the caller
    # ("Load older" UI) loses access to anything beyond the first file
    # whenever its line count meets `target` exactly.
    more_files_after_break = False
    for i, path in enumerate(chain):
        try:
            with open(path, "rb") as f:
                f.seek(0, io.SEEK_END)
                size = f.tell()
                block = 8192
                # Blocks are appended newest-last and joined once at the end:
                # prepending to a bytes object re-copies (and re-counts) the
                # whole buffer on every 8 KB step, which is quadratic in the
                # file size once `need` is large.
                blocks: list[bytes] = []
                newlines = 0
                need = target - len(collected)
                while size > 0 and newlines <= need:
                    read = min(block, size)
                    size -= read
                    f.seek(size)
                    buf = f.read(read)
                    newlines += buf.count(b"\n")
                    blocks.append(buf)
                data = b"".join(reversed(blocks))
        except OSError:
            continue
        collected = data.decode("utf-8", errors="replace").splitlines() + collected
        if len(collected) >= target:
            more_files_after_break = (i + 1) < len(chain)
            break
    # Slice in newest-first frame so `skip` is unambiguous.
    newest_first = list(reversed(collected))
    window = newest_first[skip:skip + want]
    exhausted = ((skip + len(window)) >= len(newest_first)
                 and not more_files_after_break)
    next_skip = None if exhausted else skip + len(window)
    return list(reversed(window)), next_skip


def _logs_stream_reauth(request: Request, seen_version: int) -> int:
    """Live-tail helper (the /stats/stream _rescope_on_version_change
    pattern): when settings_version.config_version() moved since `seen_version`
    — revoke_user / revoke_key / set_user_permissions / logout all bump it —
    re-resolve the caller through the same "logs" gate the connect used.
    Raises HTTPException when the credential no longer resolves or lost
    scope("logs") == "all", which ends the stream; otherwise returns the
    version to compare against on the next tick. Without this an open tab
    kept receiving every new request block (raw + final text of every
    user) after the admin revoked it, until the browser closed the
    EventSource."""
    current = settings_version.config_version()
    if current == seen_version:
        return seen_version
    _require_logs_page_sse(auth_dependencies.resolve_user_for_page_sse(request, "logs"))
    return current


async def _stream_log_lines(request: Request):
    """Yield SSE events: one for each existing tail line, then live tail.
    Re-authenticates `request` whenever the config version moves (see
    _logs_stream_reauth) and ends the stream once access is gone."""
    seen = settings_version.config_version()
    initial = int(getattr(cfg, "LOG_VIEWER_INITIAL_LINES", 2000))
    # Off the loop, same as /logs/older: this walks the rotation chain
    # backwards in 8 KB blocks and an async generator inside a
    # StreamingResponse runs on the event loop, so doing it inline let one
    # subscriber stall every other request while it read.
    # The tail's start offset is sampled in the same thread BEFORE the
    # backlog read: sampled after the backlog was yielded (which waits on a
    # slow client), every line logged in between was in neither. Now such a
    # line shows twice at worst (backlog + tail), never not at all.
    def _size_then_backlog() -> "tuple[int, list[str]]":
        try:
            size = os.path.getsize(cfg.LOG_FILE)
        except OSError:
            size = 0
        lines, _ = _read_chain_window(cfg.LOG_FILE, 0, initial)
        return size, lines

    pos, backlog = await asyncio.to_thread(_size_then_backlog)
    for line in backlog:
        yield f"data: {line}\n\n"

    # Sentinel — marks the boundary between backlog and the live poll
    # loop. The client's append() early-returns on this line; pill counts
    # are driven entirely by SEV_POLLER_JS against severity_counts().
    yield "data: __LIVE_TAIL__\n\n"

    # Live tail: poll for new lines from the offset sampled above. Reopen on
    # rotation (when the file shrinks below our last position).
    while True:
        await asyncio.sleep(0.5)
        try:
            # Off the loop: config_version() and the re-resolve hit SQLite.
            seen = await asyncio.to_thread(_logs_stream_reauth, request, seen)
        except HTTPException:
            return
        try:
            size = os.path.getsize(cfg.LOG_FILE)
        except OSError:
            yield ": waiting-for-file\n\n"
            continue
        if size < pos:
            pos = 0  # rotated
        if size == pos:
            yield ": keepalive\n\n"
            continue
        # Also off the loop — this runs every 0.5 s for the lifetime of every
        # subscriber, and the delta after a burst can be megabytes.
        def _read_delta(start: int) -> "tuple[str, int]":
            with open(cfg.LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
                f.seek(start)
                return f.read(), f.tell()

        chunk, pos = await asyncio.to_thread(_read_delta, pos)
        for line in chunk.splitlines():
            yield f"data: {line}\n\n"


_LOG_VIEWER_HTML = templates.load(__file__, "log_viewer.html")


def _require_logs_page_sse(user: dict = Depends(_get_current_user_dep)) -> dict:
    """SSE-aware variant of require_page("logs"). Two credential carriers:
    the `Authorization: Bearer` header and the HttpOnly session cookie,
    which EventSource sends automatically on a same-origin stream —
    EventSource cannot set a header. Both, plus the OPEN-mode synthetic
    admin (admin host allowlist only), are exactly auth.get_current_user's
    ladder, so it is reused rather than copied. In locked-down mode the
    credential must resolve to a user with scope("logs") == "all" — the log
    file isn't user-partitionable (a single request block carries every
    user's transcripts, filenames, and final text via
    _format_request_block), so "own" can't be enforced line-by-line and is
    rejected as access-only at the schema layer."""
    # scope("logs") (not can("logs")) — defends against legacy DB rows
    # still storing "own" from before logs joined ACCESS_ONLY_PAGES.
    if user["permissions"].scope("logs") != "all":
        raise HTTPException(403, "no access to /logs")
    return user


@router.get("/logs", response_class=HTMLResponse, dependencies=[Depends(require_user_webui_host)])
async def logs_viewer():
    # User-tier page. Shell gated by USER_WEBUI_ALLOWED_HOSTS (loopback always);
    # a keyless browser navigation loads the shell + login popup. The SSE
    # /logs/stream + /logs/older endpoints stack the host gate with their own
    # require_page("logs") check (bearer header or session cookie — EventSource
    # sends the cookie), so the data layer requires a "logs" API key.
    return HTMLResponse(
        web_common.render_page(_LOG_VIEWER_HTML, current="logs"),
        headers={"Cache-Control": "no-store"},
    )


@router.get(
    "/logs/stream",
    dependencies=[Depends(require_user_webui_host), Depends(_require_logs_page_sse)],
)
async def logs_stream(request: Request):
    return web_common.sse_response(_stream_log_lines(request))


@router.get(
    "/logs/older",
    dependencies=[Depends(require_user_webui_host), Depends(_require_logs_page_sse)],
)
async def logs_older(skip: int = 0, limit: int = 0):
    """Fetch the next older page from the rotation chain. `skip` is the
    number of lines from the chain head that have already been loaded
    into the browser DOM; `limit` defaults to LOG_VIEWER_INITIAL_LINES
    and is server-clamped to the same value (per-click max page size).
    The cap of _LOG_OLDER_MAX_PAGES pages of that size TERMINATES paging
    (next_skip=null) rather than clamping `skip`: a clamped skip would
    re-serve the same window on every further click.

    Response: `{lines: [...], next_skip: <int|null>}`. lines are
    oldest-first so the client can prepend them to the top of the
    log container as a contiguous older window. next_skip=null means
    the rotation chain is exhausted — the client hides the button."""
    initial = int(getattr(cfg, "LOG_VIEWER_INITIAL_LINES", 2000))
    want = max(1, min(limit or initial, initial))
    cap = initial * _LOG_OLDER_MAX_PAGES
    requested = max(0, int(skip))
    if requested >= cap:
        # Past the cap: chain-end, not a silent clamp (which served the same
        # window forever while the viewer kept its own cursor). No disk I/O.
        return {"lines": [], "next_skip": None}
    # _read_chain_window does blocking disk I/O over the whole rotation
    # chain — run it off the event loop so one deep page can't stall the
    # live /logs/stream tail or any concurrent request.
    lines, next_skip = await asyncio.to_thread(
        _read_chain_window, cfg.LOG_FILE, skip=requested, want=want)
    if next_skip is not None and next_skip >= cap:
        next_skip = None
    return {"lines": lines, "next_skip": next_skip}


@router.get(
    "/sev",
    dependencies=[Depends(require_user_webui_host), Depends(_get_current_user_dep)],
)
async def severity_snapshot():
    """Tiny JSON endpoint polled by every page's nav-row pill poller.

    Returns the same `severity_counts()` the server uses everywhere else
    (nav HTML render, /stats payload) — WARNING+ records since process
    start, bounded by the 2000-entry ring. Three integers, no PII.

    User-tier: USER_WEBUI_ALLOWED_HOSTS (loopback always) AND any
    authenticated user (`_get_current_user_dep` — in OPEN mode the synthetic
    admin passes, so the pill works before lockdown). The default-open user
    allowlist covers every page that embeds SEV_POLLER_JS — /stats, /logs,
    /settings, … — so the pill keeps live-updating wherever it's shown. A
    403/401 here fails the poller silently; the nav still shows the
    server-rendered count."""
    return web_common.severity_counts()
