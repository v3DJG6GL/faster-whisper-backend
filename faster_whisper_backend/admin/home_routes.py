"""Root landing hub — GET / serves the WebUI's front door.

Signed out, the page body stays hidden and the shared login gate (injected by
OPEN_MODE_BANNER_JS on the whoami-401) covers the viewport — the visitor sees
exactly the familiar auth screen instead of FastAPI's default 404 JSON.
Signed in, the page reveals a launcher: one "channel strip" tile per WebUI
page, filtered client-side to what the caller's key can actually reach (same
/auth/whoami contract the shared nav uses), plus a slim status strip (model
state, admin-only severity pills, identity, sign-out).

Auth shape matches the other user-tier page shells: the HTML is gated only by
USER_WEBUI_ALLOWED_HOSTS (loopback always allowed); nothing sensitive is
rendered server-side — tile visibility, identity and model state are all
resolved after a successful whoami. The tile hrefs themselves are the same
public knowledge as the shared nav links every page already embeds.
"""

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse

from faster_whisper_backend.settings import config as cfg
from faster_whisper_backend.core.web_common import render_page, require_user_webui_host
from faster_whisper_backend.core import templates

router = APIRouter()


# (tier, permission-key, label, href, description, wave, admin_ui_gated)
#   tier "user"  — data page gated per-key via permissions.pages[<key>]
#   tier "any"   — visible to every signed-in identity (no per-page scope)
#   tier "admin" — visible to admins only
# `wave` is the tile's 5-bar micro-waveform (heights out of a 16-unit viewBox)
# — a per-page signature riff on the brand mark (keys = key teeth, dictate =
# speech burst, stats = rising chart). `admin_ui_gated` mirrors _NAV_SPEC:
# those pages are only registered when cfg.ADMIN_UI_ENABLED, so their tiles
# drop server-side alongside them.
_TILE_SPEC: list[tuple[str, str, str, str, str, list[int], bool]] = [
    ("user", "quick_config", "quick config", "/quick-config",
     "curated pipeline rules", [7, 11, 6, 12, 8], True),
    ("user", "stats", "stats", "/stats",
     "system dashboard", [4, 7, 10, 13, 9], False),
    ("user", "reports", "reports", "/reports",
     "transcription error reports", [10, 6, 11, 7, 12], True),
    ("user", "captures", "captures", "/captures",
     "fine-tuning audio review", [5, 9, 13, 9, 5], True),
    ("user", "logs", "logs", "/logs",
     "live log stream", [6, 7, 12, 7, 6], False),
    ("any", "dictate", "dictate", "/dictate",
     "live dictation", [8, 11, 13, 11, 8], False),
    ("admin", "settings", "settings", "/settings",
     "full configuration", [8, 9, 10, 9, 8], True),
    ("admin", "keys", "keys", "/settings/api-keys",
     "users &amp; API keys", [12, 5, 12, 5, 12], True),
    ("admin", "pipeline", "pipeline", "/settings/pipeline",
     "post-processing rules", [6, 9, 7, 10, 8], True),
    ("admin", "overrides", "overrides", "/settings/overrides",
     "per-identity overrides", [7, 13, 7, 13, 7], True),
]


def _wave_svg(heights: list[int]) -> str:
    """Render a tile's 5-bar micro-waveform. Bars sit on the baseline of a
    33×16 viewBox; per-bar `--i` drives the staggered rise + hover EQ delays."""
    bars = "".join(
        f'<rect class="wb" style="--i:{i}" x="{1 + i * 6.5:g}" y="{16 - h}" '
        f'width="4" height="{h}" rx="2"/>'
        for i, h in enumerate(heights)
    )
    return f'<svg class="wave" viewBox="0 0 33 16" aria-hidden="true">{bars}</svg>'


def _tile_html(tier: str, key: str, label: str, href: str, desc: str,
               wave: list[int], idx: int) -> str:
    page_attr = f' data-page="{key}"' if tier == "user" else ""
    return (
        f'<a class="tile" data-tier="{tier}" data-hub="{key}"{page_attr} '
        f'href="{href}" style="--td:{idx * 55}ms">'
        f"{_wave_svg(wave)}"
        f'<span class="t-label"><span class="t-prompt" aria-hidden="true">&#9656;</span>{label}</span>'
        f'<span class="t-desc">{desc}</span>'
        f'<kbd class="t-key" aria-hidden="true"></kbd>'
        f"</a>"
    )


def _tiles_html() -> tuple[str, str]:
    """Build the (workspace, admin-zone) tile fragments, honouring
    cfg.ADMIN_UI_ENABLED at request time exactly like web_common._nav_items:
    pages that aren't registered don't get tiles."""
    admin_ui = bool(getattr(cfg, "ADMIN_UI_ENABLED", False))
    user_parts: list[str] = []
    admin_parts: list[str] = []
    idx = 0
    for tier, key, label, href, desc, wave, gated in _TILE_SPEC:
        if gated and not admin_ui:
            continue
        html = _tile_html(tier, key, label, href, desc, wave, idx)
        idx += 1
        (admin_parts if tier == "admin" else user_parts).append(html)
    admin_zone = ""
    if admin_parts:
        admin_zone = (
            '<div class="admin-zone">'
            '<div class="hub-rule">admin</div>'
            f'<nav class="hub-grid" aria-label="Admin pages">{"".join(admin_parts)}</nav>'
            "</div>"
        )
    return "".join(user_parts), admin_zone


_HUB_HTML = templates.load(__file__, "hub.html")


@router.get(
    "/",
    response_class=HTMLResponse,
    dependencies=[Depends(require_user_webui_host)],
)
async def home_page():
    # User-tier shell, same contract as /logs and /stats: host-gated HTML, a
    # keyless browser navigation loads the shell + the shared login gate; all
    # data the page shows is fetched with the caller's own credentials.
    tiles, admin_zone = _tiles_html()
    html = (
        _HUB_HTML
        .replace("{{HUB_TILES}}", tiles)
        .replace("{{HUB_ADMIN_ZONE}}", admin_zone)
    )
    return HTMLResponse(
        render_page(html, current="home"),
        headers={"Cache-Control": "no-store"},
    )
