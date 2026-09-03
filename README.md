<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/brand/logo-dark.png">
    <img src="docs/brand/logo-light.png" alt="fasterwhisper — backend" width="380">
  </picture>
</p>

# faster-whisper-backend

Self-hosted [faster-whisper](https://github.com/SYSTRAN/faster-whisper) transcription API with a fully configurable dictation post-processing pipeline (find→replace rules, word maps, spoken punctuation, casing). Exposes an **OpenAI-compatible** `/v1/audio/transcriptions` endpoint for any Whisper client.

## Features

- OpenAI-compatible API — drop-in replacement for `client.audio.transcriptions.create(...)`
- **Live streaming dictation** — WebSocket endpoint `/v1/audio/transcriptions/stream` that emits
  flicker-free partial text *while you speak* (LocalAgreement-2 stabilization) and **post-processed**
  final text per utterance (a locked, append-only `committed` prefix plus a revisable `tail` —
  both sent as full strings the client replaces). Reuses the same models, VAD, and post-processing
  pipeline as the batch route (which is unchanged); accepts raw 16 kHz PCM **or** browser Opus/WebM
  (decoded server-side via a bundled `ffmpeg` — `imageio-ffmpeg`, no system install needed; a
  system `ffmpeg` on PATH is used when present); two-tier Silero/energy endpointing. Try it in the browser at `/dictate`.
  On by default (auth-gated); tune everything via `WHISPER_STREAMING_*` / `/settings`. A shared
  `INFERENCE_CONCURRENCY` limiter governs streaming **and** batch so they don't oversubscribe the GPU.
  **Per-utterance translation handshake:** a client that declares `translate_expect`
  (with `per_utterance: true`) in the WebSocket handshake gets a follow-up frame
  `{"type": "captured", "id": …, "utterance": …}` after each `final`; the server holds that
  utterance's log receipt open and the client must echo the id back as `captured_id` on its
  `POST /v1/text/translations` so both halves log as one block. An id never claimed is released
  by the `LOG_RECEIPT_HOLD_S` idle sweep (default 90 s) with a "never sent" note.
- GPU-accelerated (CUDA) via faster-whisper + CTranslate2, with **automatic CPU fallback** when no GPU is available
- **Per-request model selection** — clients pass `model="large-v3"` / `"large-v3-turbo"` / any HF repo id; LRU-cached in VRAM
- **Text-to-text translation stage** (optional install) — translate the finished transcript into arbitrary target languages with local GGUF models (HY-MT1.5, TranslateGemma, MiLM-MT, … via llama.cpp): `translate_to=de,fr` on a transcription request or standalone `POST /v1/text/translations`; **fluent** (sentence-group merge) or **faithful** (per-cue) mode, glossary enforcement, per-language `translations` in the response. Off by default (`WHISPER_TRANSLATION_ENABLED`).
- **Dictation phrase map**: `"Punkt"` → `.`, `"Komma"` → `,`, `"neue Zeile"` → `\n`, `"Klammer auf"` → `(`, ~80 phrases total — every rule editable/replaceable in the WebUI
- Auto-capitalize after sentence ends; strips Whisper noise commas; lowercases mid-sentence non-nouns after stripped Whisper terminators
- Live HTML log viewer at `/logs` (Server-Sent Events, color-coded pipeline trace per request)
- Live system overview at `/stats` (loaded models + VRAM, GPU/CPU/RAM, request latency, recent transcriptions, sparklines — works fully offline, no CDN)
- Admin WebUI at `/settings` for editing every setting without redeploying (on by default; allowlist + bearer-token gated)
- **Configure everything via environment** — every setting has a `WHISPER_*` variable; pin them via `.env`, docker-compose, or the service env. Env-pinned settings are shown read-only in the admin WebUI.
- Cross-page nav with severity pills (WARN/ERR/CRIT counts since process start) on every page
- Runs anywhere: Windows Service, Linux systemd, Docker, or bare `python main.py`

## Requirements

- Python 3.14 (the Docker image ships `python:3.14-slim`; 3.12+ works for bare installs)
- Linux, macOS, or Windows 10/11
- **GPU (optional)**: NVIDIA GPU + driver supporting CUDA 12.x (WSL2 driver works).

The default config is **GPU-first** (`MODEL_DEVICE = "cuda"`): on a host without a
usable GPU, each model load automatically falls back to CPU (`int8`), logging a
one-time fallback warning per model. To make CPU the primary path (and silence the
warning), set `WHISPER_MODEL_DEVICE=cpu`. The base `requirements.txt` is CPU-capable
and cross-platform; GPU acceleration is additive — `pip install -r requirements-gpu.txt`
on an NVIDIA host.

## Install

### Linux / macOS (development)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt            # CPU, all platforms
# pip install -r requirements-gpu.txt      # add NVIDIA CUDA wheels (GPU box)
python main.py                             # serves on http://0.0.0.0:8000
```

### Linux (production, systemd)

```bash
./install-service.sh              # CPU   (auto-elevates via sudo, creates venv,
./install-service.sh --gpu        # GPU    installs deps, writes + starts the unit)
./install-service.sh --gpu --full # GPU + heavy extras — the bare-metal
                                  # equivalent of the :latest-gpu-full image
                                  # (diarization + music separation +
                                  # translation; --full works without --gpu
                                  # too). Re-run with the SAME flags to refresh.
# manage: systemctl status|restart whisper-api ; journalctl -u whisper-api -f
# remove: ./uninstall-service.sh
```

The generated unit pins `WHISPER_DATA_DIR=<repo>/data` and
`WHISPER_MODELS_DIR=<repo>/models`, so state and models live inside the
checkout (matching the Windows layout) instead of the container-first
`/data` / `/models` defaults.

### Docker (any OS)

CI publishes four images to the Forgejo container registry on every `v*` tag —
and every green push to `main` mints a tag automatically, so images follow each
merge (a commit marked `[skip release]` mints no tag and publishes no images):
`:latest` (CPU) and `:latest-gpu` (adds the CUDA 12 /
cuDNN 9 wheels), plus `:latest-full` / `:latest-gpu-full` — the same images
with the optional heavy extras baked in (speaker diarization: pyannote +
torch + system ffmpeg; music separation; text-to-text translation:
llama-cpp-python; the GPU flavor installs torch from the cu126 index so
it shares ctranslate2's pip CUDA libraries). Every flavor is also tagged
`:v<version>` and `:sha-<short>` with the matching suffix. The lean images
stay fully functional — a diarization/translation request on them soft-fails
with a response warning naming the requirements file
(`requirements-diarize.txt` / `requirements-translate.txt`). Model weights
are never baked into any image: pyannote pipelines and GGUF translation
models download on first use into the models volume (pyannote is gated on
huggingface.co — accept the model terms and set `WHISPER_HF_TOKEN` first).

```bash
# CPU — pulls forgejo.informethic.ch/v3djg6gl/faster-whisper-backend:latest
docker compose up -d

# GPU — standalone file: pulls :latest-gpu and passes through the host NVIDIA GPU(s)
docker compose -f docker-compose.gpu.yml up -d
```

`docker-compose.gpu.yml` is a self-contained mirror of `docker-compose.yml` —
same ports/env/volumes, differing only in the GPU bits (`:latest-gpu` image +
the NVIDIA device reservation). The GPU path needs an NVIDIA driver (CUDA 12.x)
**and** the NVIDIA Container Toolkit on the host. With no GPU visible, model load
auto-falls back to CPU/int8. To build locally instead of pulling, uncomment
`build:` in the compose file (the GPU build uses `Dockerfile.gpu`). The package
inherits the repo's visibility; if the repo is private, `docker login
forgejo.informethic.ch` first. The published images use zstd-compressed OCI
layers, so pulling them needs Docker >= 23, containerd >= 1.5, or a current
podman — an older daemon fails the pull with an unhelpful media-type error.

The container runs as a **non-root user** (default `1000:1000`); set `PUID` /
`PGID` in `.env` (or the environment) to run as a different user/group — no
rebuild needed, volumes work with any UID out of the box.

### Windows (production, service)

```powershell
# Auto-elevates via UAC, bootstraps the venv, installs requirements, downloads
# WinSW, and registers the Windows Service in one go.
.\install-service.ps1
.\install-service.ps1 -Gpu        # also install the NVIDIA CUDA wheels
.\install-service.ps1 -Gpu -Full  # + heavy extras — the bare-metal equivalent
                                  # of :latest-gpu-full (diarization + music
                                  # separation + translation; -Full works
                                  # without -Gpu too). Re-run with the SAME
                                  # flags to refresh.
```

First server start eagerly preloads the models in `PRELOAD_MODELS` (by default
two: `Systran/faster-whisper-large-v2` and `Systran/faster-whisper-large-v3` —
several GB total) into `WHISPER_DOWNLOAD_ROOT`, which defaults to the models
dir (`/models` in containers and on bare-metal Linux, `<repo>\models` on
Windows); set `WHISPER_DOWNLOAD_ROOT=` empty to fall back to the standard
HuggingFace cache (`~/.cache/huggingface`). Set `WHISPER_PRELOAD_MODELS=large-v2` (or empty) to
download/warm fewer models at startup.

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

CI runs the suite on Linux (Python 3.12/3.13/3.14) and Windows for every pull
request and every push to `main`, and fails the run if coverage drops below the
gate configured in `.coveragerc` (`.forgejo/workflows/ci.yml`). A `v*` tag run
deliberately skips both suites — a tag is only minted for a commit that already
tested green — and goes straight to building and publishing the images.

## Usage

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
with open("audio.wav", "rb") as f:
    r = client.audio.transcriptions.create(
        model="whisper-1", file=f,
        response_format="verbose_json",
        timestamp_granularities=["word"],
    )
print(r.text)
```

## Configuration

**Every** factory default — models, default prompt, server host/port, log paths, faster-whisper transcribe defaults, **and** the post-processing pipeline rules — lives in the committed **`config.json`** at the repo root (single source of truth; `config.py` only loads it and layers the overrides below on top). Edit it directly to change a default for every deployment, then restart the service (`systemctl restart whisper-api` / `Restart-Service WhisperAPI` / the `/settings` restart button) to pick up the changes. The algorithm code in `main.py` doesn't need to be touched.

Layers of overrides, **env wins over file wins over in-repo default**:

1. **`config.json`** — committed in-repo defaults (all scalar settings + pipeline rules).
2. **`config.local.json`** (gitignored) — runtime overrides written by the admin WebUI; or hand-edited (see `config.local.example.json`). Validated against `config_store.AdminConfig`; unknown keys are rejected. Lives in the data dir (default `/data/config.local.json`; follows **`WHISPER_DATA_DIR`** like every runtime-state path, and **`WHISPER_CONFIG_LOCAL`** relocates just this file). All path defaults are container-first on Linux: app state under `/data` (SQLite stores under `/data/db` — move with `WHISPER_DB_DIR`), models under `/models` (move with `WHISPER_MODELS_DIR`). On Windows — always bare metal, the images are Linux — the same tree just roots in the checkout: `.\data` (stores under `.\data\db`) and `.\models`, so a plain `python main.py` needs no path envs. For a bare-metal Linux checkout set `WHISPER_DATA_DIR=./data` to keep state inside the repo dir the same way.
3. **`WHISPER_*` env vars** — per-machine deployment pins; always win. Source them from a `.env` file (auto-loaded at startup), docker-compose `environment:` / `env_file:`, or the service env (`<env>` elements in `WhisperAPI.xml`, regenerated by `install-service.ps1`).

### Environment variables

**Every** editable setting has a matching `WHISPER_<FIELD>` variable — see
[`.env.example`](.env.example) for the complete, grouped list with defaults.
Copy it to get started:

```bash
cp .env.example .env     # auto-loaded on startup; gitignored
```

Notes:
- An env-pinned setting is **read-only in the admin WebUI** (greyed out, badged
  `env: WHISPER_…`). Unset the variable to make it editable in the UI again.
- Booleans accept `1/true/yes/on/y/t/enabled` and `0/false/no/off/n/f/disabled`
  (any other value keeps the current setting and logs a warning at startup);
  lists are comma-separated; an empty value clears/disables nullable settings
  (e.g. `WHISPER_NO_SPEECH_THRESHOLD=`).
- **Secrets** (`WHISPER_BOOTSTRAP_ADMIN_KEY`, `WHISPER_HF_TOKEN`) also
  accept a `*_FILE` form pointing at a mounted secret file, so the value stays
  out of `docker inspect` / the process environment. (`WHISPER_USE_AUTH_TOKEN`,
  the pre-rename spelling of the HF token, is still accepted as an alias.)
- **Server port**: `WHISPER_SERVER_PORT` is honored, but in Docker you must also
  update the compose `ports:` mapping to match.
- **Structured settings** (`PIPELINE_RULES`, `MODEL_OVERRIDES`) accept a JSON
  string; per-model fields can also be set one at a time via
  `WHISPER_MODEL_OVERRIDE__<id>__<FIELD>` (encode `/`→`__SLASH__`, `.`→`__DOT__`).

A few of the most common variables:

| Env var | Maps to setting | Effect |
|---|---|---|
| `WHISPER_DEFAULT_MODEL` | `DEFAULT_MODEL` | Model used when request sends `whisper-1` or omits `model` |
| `WHISPER_ALLOWED_MODELS` | `ALLOWED_MODELS` | Comma-separated allowlist (default: the two official Systran large-v2/large-v3 builds); empty = any well-formed model id passes |
| `WHISPER_MODEL_DEVICE` | `MODEL_DEVICE` | `cuda` (default) or `cpu` |
| `WHISPER_PRELOAD_MODELS` | `PRELOAD_MODELS` | Comma-separated list to load eagerly at startup (no first-request warm-up) |
| `WHISPER_SERVER_PORT` | `SERVER_PORT` | Listen port (also update the Docker `ports:` mapping) |
| `WHISPER_DEFAULT_PROMPT` | `DEFAULT_PROMPT` | Initial prompt when request omits `prompt` (empty string disables) |
| `WHISPER_ADMIN_UI` | `ADMIN_UI_ENABLED` | `0` unregisters `/settings*` + `/quick-config`, `/captures`, `/reports` (on by default) |
| `WHISPER_ADMIN_WEBUI_ALLOWED_HOSTS` | `ADMIN_WEBUI_ALLOWED_HOSTS` | Comma-separated IPs/CIDRs allowed to reach the **admin** pages — `/settings`, `/settings/api-keys`, `/docs` (loopback always implicit; default loopback only) |
| `WHISPER_USER_WEBUI_ALLOWED_HOSTS` | `USER_WEBUI_ALLOWED_HOSTS` | Comma-separated IPs/CIDRs allowed to reach the **user** pages — `/`, `/quick-config`, `/captures`, `/reports`, `/stats`, `/logs`, `/dictate`, `/sev` (loopback always implicit; default open `0.0.0.0/0, ::/0`) |

### Translation (text-to-text, optional)

The settings of the admin WebUI's **Translation** group (all with `WHISPER_*`
env twins), needing `pip install -r requirements-translate.txt`:

- **Capacity** — `TRANSLATION_ENABLED` (master switch, default off; requested
  runs decline softly with a response warning while off and
  `POST /v1/text/translations` returns 403).
- **Models** — `TRANSLATION_DEFAULT_MODEL` (GGUF ref `org/repo[:quant]`, e.g.
  `tencent/HY-MT1.5-7B-GGUF:Q4_K_M`), `TRANSLATION_ALLOWED_MODELS`
  (per-request allowlist; empty = any well-formed ref — ships with the two
  top-ranked dedicated MT models, `tencent/HY-MT1.5-7B-GGUF:Q4_K_M` and
  `mradermacher/MiLMMT-46-12B-v0.1-GGUF:Q4_K_M`; **upgrading** from a release
  where this shipped empty: add your model to the list or set
  `WHISPER_TRANSLATION_ALLOWED_MODELS=` (empty) to keep the old any-ref
  behaviour, otherwise requests for other refs are refused),
  `TRANSLATION_PRELOAD_MODELS` (warmed at startup),
  `TRANSLATION_MAX_LOADED_MODELS` (LRU cap — a 7B Q4 model holds ~5 GB),
  `TRANSLATION_DEVICE` (`auto` follows `MODEL_DEVICE`),
  `TRANSLATION_IDLE_TIMEOUT_S` (idle unload).
- **Prompting** — `TRANSLATION_PROMPT_FAMILY` (`auto` detects from the model
  name: HY-MT/Hunyuan, TranslateGemma, MiLM-MT, Seed-X, generic chatml;
  `custom` renders `TRANSLATION_PROMPT_TEMPLATE`, which must contain `{text}`
  and `{target_language}` — the WebUI previews and test-runs the template),
  `TRANSLATION_BATCH_SEGMENTS` (faithful-mode segments per prompt).
- **Per-request defaults** (per-identity/per-model overridable, lockable) —
  `TRANSLATE_TO` (csv of target codes; empty = translate only when the request
  asks), `TRANSLATION_MODEL`, `TRANSLATION_MODE` (`fluent` merges sentence
  groups for flow, `faithful` keeps 1:1 cue alignment),
  `TRANSLATION_CONTEXT_SEGMENTS`, `TRANSLATION_MAX_TARGETS`,
  `TRANSLATION_GLOSSARY` (`source = target` lines enforced via the prompt).

Model weights are not pip packages — GGUF files download on first use via
huggingface_hub into the models volume (`HF_HOME`).

### Rate limits & concurrency

Seven request budgets, all in the **Concurrency & Request Limits** settings
group, all with `WHISPER_*` env twins, and all **hot** — the limiters re-read
their ceiling on every call, so raising one applies to the next request with
no restart and no bucket reset.

| Setting | Shape | Guards |
|---|---|---|
| `TRANSLATE_MAX_INFLIGHT_PER_USER` | concurrency (default 2) | `POST /v1/text/translations` — how many translations one caller may have *running* |
| `TRANSLATE_RATE_PER_MIN` | 120/min | backstop for a loop that never reaches the gauge because each attempt fails validation first |
| `STREAMING_MAX_SESSIONS_PER_USER` | concurrency (default 4) | live-dictation WebSockets per caller |
| `URL_PREVIEW_RATE_PER_MIN` | 10/min | `POST /v1/audio/url-preview` |
| `CAPTURES_AUDIO_RATE_PER_MIN` | 240/min | capture-audio fetches |
| `REPORTS_SUBMIT_RATE_PER_10MIN` | 20/10 min | user report submissions |
| `LOGIN_FAILURE_RATE` | 10/min | `POST /auth/login` — **failures only** |

**Keying.** Every budget except the login throttle is charged to
`user_id → key_id → client host`, in that order: the user if the key carries
one, else the key itself (machine clients often have no user), else the peer
address. The host rung means several callers behind one NAT share a bucket —
the conservative direction. In **open mode** (no admin key exists yet) every
caller passing the `ADMIN_WEBUI_ALLOWED_HOSTS` gate (loopback by default)
resolves to the same synthetic admin, so per-user budgets behave
**server-wide** among those callers (everyone else still gets 401); that is a
property of open mode, not a bug, and it goes away the moment you create an
admin key.

**Concurrency vs rate.** `TRANSLATE_MAX_INFLIGHT_PER_USER` is a concurrency
cap, not a per-minute ceiling, because a single translation can run for
minutes — what hurts is one client holding the GGUF model while everyone else
waits, and a per-minute counter cannot express that (a request can outlive its
own window). `STREAMING_MAX_SESSIONS_PER_USER` is the per-caller twin of the
server-wide `STREAMING_MAX_SESSIONS`: the per-user check runs first, so one
client cannot fill the whole server-wide pool.

**The login throttle** is keyed by client **host**, not identity — an attempt
has no identity yet, and the key it presents is exactly what must not be
trusted. Only *failures* count and a success clears the window, so fat-fingering
one paste never walks you toward a lockout. It is a **cookie-login** guard
only: bearer API-key auth is never throttled, so an automation client cannot
be locked out by somebody else's browser. `LOGIN_FAILURE_RATE=0` disables it
outright — the escape hatch if you ever throttle yourself out of the WebUI.

**`0` = unlimited** for every setting above, and short-circuits before any
bookkeeping, so a single-user box pays nothing for machinery it doesn't want.

**The 429.** Bodies are OpenAI-shaped and name the field an admin would raise:

```json
{"error": {"message": "you already have 2 translations running — wait for one to finish",
           "type": "rate_limit_exceeded",
           "param": "TRANSLATE_MAX_INFLIGHT_PER_USER",
           "retry_after": 5},
 "detail": "you already have 2 translations running — wait for one to finish"}
```

`error.param` is the setting to raise; `Retry-After` rides as a header too (on
a concurrency cap it is an advisory nudge — there is no honest deadline for
"when somebody else finishes").

**Per-process caveat.** All limiter state is in-process. With
`SERVER_WORKERS > 1` each worker enforces its own copy, so every budget is
effectively multiplied by the worker count. The threat model is "runaway
script / accidental double-click / one client starving the others", not a
motivated attacker spreading load across workers; anything stronger needs
shared state (Redis) and does not belong in-process.

### Model preloading

`POST /v1/models/preload` asks the server to warm the models a job is about to
need, so a stage's load happens *during* the previous stage instead of after
it. User-tier bearer auth, no host allowlist — the same tier as `/v1/models`
and `/v1/me`, which already publish `loaded` flags for these models.

```bash
curl -X POST http://localhost:8000/v1/models/preload \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"models":[{"family":"separation","id":"UVR-MDX-NET-Inst_HQ_4"},
                 {"family":"diarization","id":"pyannote/speaker-diarization-community-1"}]}'
```

Families are `whisper`, `diarization`, `separation` and `translation`. (There
is deliberately no `vad`: Silero ships inside faster-whisper, runs on the CPU
and has neither a registry entry nor an evictor.) Each entry comes back with
one of four states:

- **`resident`** — already loaded; its idle clock was restarted.
- **`loading`** — the worker is loading it now.
- **`queued`** — admitted, waiting behind another load (one worker, so loads
  serialise — which is also what keeps the VRAM measurement clean).
- **`deferred`** — not warmed, with a `reason`: `insufficient_vram`,
  `insufficient_ram`, `vram_unknown`, `size_unknown`, `family_busy`,
  `not_allowed`, `stage_disabled`, `queue_full` or `disabled`.

**The endpoint never errors.** It answers `202` for everything except a
structurally invalid body (`422`). A model your allowlist refuses, a stage
that is switched off and a server with preloading disabled all come back
`202` + `deferred` + a reason, because the client's response to all three is
the same: let the stage load its model in-band, exactly as it did before this
endpoint existed. A `404` therefore keeps its old meaning — "this backend
build is too old" — and never means "preloading is off here".

The server registers the same kind of plan for every batch job once the stage
plan is resolved, and advances it as the job moves between stages, so
preloading works with no client changes at all. A client that already POSTed a
plan can hand its id back on the transcribe request as the `preload_plan` form
field instead of the server duplicating it.

**Warm leases vs job leases.** A running job holds a *job lease*: its model
cannot be freed underneath it. A plan holds a weaker *warm lease*, which does
exactly one thing — makes the model ineligible for idle eviction (and for
preload-driven eviction) while the plan is alive. It never forces a model to
stay resident, never pins memory against a job that needs it, and never gates
a loader, so **a job is never delayed by warmth**. Leases are released as a
cascade: when a plan expires, its keys are dropped unless another live plan
still wants them.

Settings (Models → *Advanced — preload & warm cache*):
`MODEL_PRELOAD_ENABLED` (master switch), `MODEL_PRELOAD_WARM_TTL_S` (plan
lifetime, restamped by a re-POST and by every stage start of the owning job —
so it bounds idle plans, not long ones), `MODEL_PRELOAD_VRAM_RESERVE_MB` /
`MODEL_PRELOAD_RAM_RESERVE_MB` (headroom a preload must leave free; the VRAM
figure is checked against the *driver's* free memory, since other processes on
the card are invisible to our own bookkeeping), and
`MODEL_PRELOAD_EVICT_IDLE_MODELS` (whether a preload may drop an idle,
unleased, unwarmed peer of the same family to make room — off means "never
disturb what is already loaded"). `/stats` carries a `preload` block (worker
alive, plan count, warm count, queue depth) so you can tell the feature apart
from the feature doing nothing.

### Allowed hosts

WebUI access is gated by two IP/CIDR allowlists, bucketed by **privilege tier** — each is the outer (host) layer; an API key is still required on the data layer.

- **`ADMIN_WEBUI_ALLOWED_HOSTS`** — admin pages (`/settings`, `/settings/api-keys`, `/docs`). Default `["127.0.0.1", "::1"]` (loopback only); data also requires an **admin** key.
- **`USER_WEBUI_ALLOWED_HOSTS`** — user pages (`/`, `/quick-config`, `/captures`, `/reports`, `/stats`, `/logs`, `/dictate`, `/sev`). Default `["0.0.0.0/0", "::/0"]` (**open**) — the per-page API key is the real gate; narrow this to restrict which networks may even reach the pages.

Loopback is *always* implicitly allowed regardless of the configured list, so a typo can never lock you out from the box itself.

```bash
# Lock the admin pages to loopback, narrow the user pages to your LAN
# (the API key still gates the data). Set via env, or edit the same
# fields in the /settings UI (→ config.local.json):
WHISPER_ADMIN_WEBUI_ALLOWED_HOSTS=127.0.0.1,::1
WHISPER_USER_WEBUI_ALLOWED_HOSTS=127.0.0.1,::1,192.168.1.0/24
```

CIDR is accepted (`192.168.0.0/16`) and so are bare IPs (`10.0.0.5`). For a dual-stack "any host" allowlist you need both `0.0.0.0/0` (IPv4) and `::/0` (IPv6).

### Transcribe from a URL

`POST /v1/audio/transcriptions` accepts a `source_url` (and `POST /v1/audio/url-preview` shows what it resolves to) when `WHISPER_URL_DOWNLOAD_ENABLED` is on. The link is fetched server-side with **yt-dlp**: `URL_ALLOWED_EXTRACTORS` limits which sites are allowed, `URL_ALLOW_DIRECT_MEDIA` (on by default) additionally accepts plain audio/video file links, and `URL_ALLOW_GENERIC` (off) accepts any page.

**Deployment: outbound network isolation.** A client picks the URL, so the server can be pointed at things it can reach and the client cannot — a LAN service, the cloud metadata endpoint at `169.254.169.254`. The application blocks that on its own: one address policy (`net_policy.py`) is applied to **every** fetch on this path — the direct-media probe, the thumbnail fetch, yt-dlp's metadata probe and the yt-dlp download subprocess — on **hop 0 and every redirect hop**, with the resolved IP **pinned** for the connection (so a second DNS answer cannot move the target) and only `http(s)` spoken. yt-dlp's own opener is covered by the guard in [`ytdlp_plugins/`](ytdlp_plugins/), installed in-process and in the download subprocess; **if it cannot be installed, link downloads are refused** rather than run unguarded (look for `[url-dl] SSRF guard active` at startup).

That is application-level containment. For defence in depth, also deny the private ranges at the network layer, so a bug in the guard is not the only thing standing between a pasted link and your LAN. On the container host, with the backend in its own network namespace:

```nft
# /etc/nftables.d/whisper-egress.nft  —  nft -f this file
table inet whisper {
  chain egress {
    type filter hook forward priority filter; policy accept;
    # Only traffic leaving the backend's subnet (adjust to your compose net).
    ip  saddr != 172.20.0.0/16 return
    ip  daddr { 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 127.0.0.0/8, 169.254.0.0/16 } drop
    ip6 daddr { ::1/128, fc00::/7, fe80::/10 } drop
  }
}
```

Adjust `172.20.0.0/16` to the compose network's subnet, and drop the `172.16.0.0/12` line only if that subnet overlaps it (it does for Docker's defaults — give the service its own network with an explicit subnet outside the ranges you deny). The commented block in `docker-compose.yml` shows the same idea. On a bare-metal install, the equivalent is an `output` chain matched on the service user (`meta skuid whisper`).

### Behind a reverse proxy

Two things change once the app is reached through a proxy:

- **Client IP.** Without `FORWARDED_ALLOW_IPS` the app sees the *proxy's* IP, so the allowlists above gate on that (admin pages 403 for everyone). Set it to the proxy's IP/subnet — it is read by **uvicorn directly**, so it carries no `WHISPER_` prefix — and uvicorn then rewrites the client IP from `X-Forwarded-For`. Never `*` unless the app is unreachable except through the proxy (the header is spoofable).

  This is also the **only** mechanism that makes per-host keying meaningful behind a proxy. Everything that falls back to the client host — the login throttle (always) and any rate limit charged to a caller with no user or key id (see [Rate limits & concurrency](#rate-limits--concurrency)) — otherwise sees one address for the whole internet, collapsing every caller into a single shared bucket. `FORWARDED_ALLOW_IPS` is peer-validated: uvicorn trusts `X-Forwarded-For` only when the *connecting socket* is one of the listed addresses, which is what stops a client from forging its way into somebody else's bucket. Set it to the proxy's address, never `*`.
- **Origin.** Unsafe methods (`POST`/`PUT`/`DELETE`) must come from the same origin as the request's `Host` header. Nginx Proxy Manager, NPMplus, Caddy and Traefik pass `Host` through by default, so the check already succeeds and nothing is needed. A proxy configured to **rewrite `Host` to the upstream** (e.g. `proxy_set_header Host backend:8000`) makes every WebUI mutation fail with `403 Origin not allowed for this host` — list the public origin in `TRUSTED_ORIGINS`. This only widens that check; cross-origin API access is still governed solely by `CORS_ALLOW_ORIGINS`.

```bash
FORWARDED_ALLOW_IPS=172.16.0.0/12                    # proxy IP/subnet (uvicorn, no WHISPER_ prefix)
WHISPER_SESSION_COOKIE_SECURE=1                      # required when the proxy serves HTTPS
WHISPER_TRUSTED_ORIGINS=https://whisper.example.com  # only if the proxy rewrites Host
```

## Endpoints

**Core API** (`/v1`, bearer API-key auth, no host allowlist — always registered):

- `POST /v1/audio/transcriptions` — OpenAI-compatible transcription. Pass `model=<name>` to pick a specific model (any faster-whisper short name or HF repo id).
- `POST /v1/audio/translations` — OpenAI-compatible Whisper translate-to-English (the transcription handler with `task` pinned to `translate`; English is Whisper's only target).
- `POST /v1/text/translations` — **text-to-text** translation of caller-supplied text/segments into arbitrary target languages via local GGUF models (llama.cpp; see the Translation configuration group). Distinct from `/v1/audio/translations`: this translates finished text with a dedicated translation model, not audio with Whisper. 403 while `TRANSLATION_ENABLED` is off.
- `WS   /v1/audio/transcriptions/stream` — live streaming dictation (raw 16 kHz PCM or browser WebM/Opus); see the Features section.
- `GET  /v1/models` — list currently-loaded models, the configured default, and the allowlist (if set). Also carries the server's build identity — `server_name` ("faster-whisper-backend"), `server_version`, and the per-process `boot_id` — non-standard fields clients use to recognize the full backend and display its version. The version resolves via `WHISPER_BUILD_VERSION` (baked into container images by CI as `git describe`) or a runtime `git describe` on bare-metal checkouts (see `build_info.py`).
- `POST /v1/models/preload` — ask the server to warm the models a job is about to need, so a stage's load overlaps the previous stage instead of following it. Always `202` (only a structurally invalid body is a `422`); each entry answers `resident`/`loading`/`queued`/`deferred`. See [Model preloading](#model-preloading).
- `GET  /v1/me` — the caller's effective request-override capabilities (drives client UI).
- `GET  /v1/override-profiles` (+ `/{name}`) — the override profiles this caller may request per-request; name list / single-profile preview.
- `GET/PATCH /v1/pipeline-rules` — the exposed post-processing rules this caller may view/edit (same gating + semantics as `/quick-config`, for API clients).
- `GET  /v1/recent-words` — recently-transcribed word/phrase suggestions (for rule-editor autocomplete).
- `GET  /v1/usage` — the caller's own transcription usage rollup.
- `GET/PUT/DELETE /v1/client-settings` — per-account opaque settings blob for desktop-client sync (all machines authenticating as the same account share one configuration). Optimistic concurrency: PUT echoes `base_version`; a mismatch returns `409` carrying the current `{version, blob, updated_at, device}` so the client can merge and re-PUT without another GET; oversized blobs get `413`, malformed bodies `422`. GET on an empty store returns `200 {version: 0, blob: null}` — a route-level `404` means the backend build predates the endpoint. The blob is stored verbatim and never logged (it may contain the client's own backend API keys). Open-mode caveat: with no admin key configured, only callers passing the `ADMIN_WEBUI_ALLOWED_HOSTS` gate (loopback by default) resolve to the synthetic `(open-mode)` user and share its single row; every other caller gets `401` until an API key exists — remote sync always needs a key. Cookie-authenticated (non-bearer) PUT/DELETE additionally needs `X-CSRF-Token`; the desktop client uses bearer and is exempt. Admin visibility/management lives on `/settings/api-keys`: each account shows a `⇅ vN` chip plus a "Synced settings" drawer (metadata only — version/size/last device/updated; blob contents never render) with Export (file download, two-press guard because it can include the account's saved API keys), Import (accepts a desktop settings export or a previously exported server file; force-writes with a version bump so every device converges on its next sync), and Delete.

**User pages** (host-gated by `USER_WEBUI_ALLOWED_HOSTS`, loopback always allowed; data endpoints additionally need an API key with the page permission):

- `GET  /` — landing hub: the sign-in screen when signed out, a launcher listing the pages the caller's key can reach when signed in.
- `GET  /logs` — live log viewer; `GET /logs/stream` (SSE feed), `GET /logs/older` (pagination).
- `GET  /stats` — system overview dashboard; `GET /stats/snapshot` + `GET /stats/stream` (JSON one-shot + ~1 Hz SSE), `GET /stats/usage` (per-user/key usage chart data). Page scope `own` shows a user only their own jobs and usage plus a coarse server status (the system metrics cards return with `STATS_OWN_SCOPE_SHOW_SYSTEM_METRICS`); `all` shows every user's numbers with identities scrubbed for non-admins.
- `GET  /quick-config` — end-user rule editor (state/recent/stream/usage/reapply-rules sub-endpoints, incl. error-report submission).
- `GET  /captures` — training-data curation UI (`/captures/api/*`: list/export, per-capture CRUD + audio, samples + merge/preview, reprocess jobs).
- `GET  /reports` — error-report triage UI (`/reports/api/*`).
- `GET  /dictate` — browser demo for the streaming endpoint.
- `GET  /sev` — tiny JSON severity counts powering the nav pills.

**Admin pages** (host-gated by `ADMIN_WEBUI_ALLOWED_HOSTS`, loopback always allowed; data endpoints require an **admin** key):

- `GET  /settings` — admin WebUI; `GET/POST /settings/state`, `POST /settings/restart`.
- `GET  /settings/pipeline` — pipeline-rule editor; `GET/POST /settings/factory-rules` (+ `/clear-local-override`), `POST /settings/test-pipeline`.
- `GET  /settings/api-keys` — per-user API key management + per-account synced-settings management (chip, drawer, export/import/delete) (`/settings/api-keys/api/*`).
- `GET  /settings/overrides` — per-identity override editor (`state`, `resolve` explorer, profile rename).
- `GET  /docs`, `GET /redoc`, `GET /openapi.json` — interactive API docs (always registered; `/openapi.json` additionally requires an admin key).

`WHISPER_ADMIN_UI=0` unregisters `/settings*` **and** the WebUI pages that ride the same switch (`/quick-config`, `/captures`, `/reports`); the core API, `/logs`, `/stats`, `/dictate`, `/sev`, and `/docs` stay up.

**Auth** (user-tier host gate):

- `GET  /auth/whoami` — resolve the current credentials to `{open_mode, user_id, username, is_admin}`. The WebUI uses this to render the login modal and the OPEN-mode banner.
- `POST /auth/login`, `POST /auth/logout` — exchange an API key for a browser session cookie / end the session.

### Model selection examples

```python
# Use the configured default (Whisper-1 = OpenAI default name)
client.audio.transcriptions.create(model="whisper-1", file=f)

# Pick a specific faster-whisper short name
client.audio.transcriptions.create(model="large-v3-turbo", file=f)

# Use a German finetune from Hugging Face
client.audio.transcriptions.create(model="primeline/whisper-large-v3-turbo-german", file=f)
```

> **Note:** `ALLOWED_MODELS` ships as a curated 2-model set
> (`Systran/faster-whisper-large-v2`, `Systran/faster-whisper-large-v3`),
> so requests for other ids (e.g. `large-v3-turbo`, `primeline/whisper-large-v3-turbo-german`)
> are rejected until you add them to the allowlist (`WHISPER_ALLOWED_MODELS=...`)
> or clear it (`WHISPER_ALLOWED_MODELS=` → any well-formed model id passes: a
> short name or an `org/name` repo id, plus whatever `DEFAULT_MODEL` is set to;
> filesystem paths sent by a client are refused).

First-use of any new model triggers a one-time download (~600 MB to ~1.5 GB depending on the model) into `%USERPROFILE%\.cache\huggingface\hub\`. Subsequent loads come from cache (~5–10 s into VRAM).

## Service control

Linux (systemd):

```bash
sudo systemctl restart whisper-api       # after editing main.py / config
sudo systemctl stop    whisper-api
systemctl status       whisper-api
./uninstall-service.sh                   # remove the service
```

Windows (service):

```powershell
Restart-Service WhisperAPI               # after editing main.py
Stop-Service    WhisperAPI
Get-Service     WhisperAPI
.\uninstall-service.ps1                  # remove the service
.\uninstall-service.ps1 -RemoveLocal     # also delete logs/, WhisperAPI.exe / .xml, any legacy nssm.exe
```

Docker: `docker compose restart` / `docker compose down`. Any deployment can also
self-restart from the admin UI (`/settings` → **restart**) — it re-execs the process
on Linux/macOS and uses WinSW on Windows.

`Get-Content -Wait logs\whisper.log` to tail logs in a terminal, or open `http://localhost:8000/logs`.

## Post-processing pipeline

A single ordered list of rules — `cfg.PIPELINE_RULES` — is applied to each transcript's joined text. Each row is one of:

- `regex-list` — an ordered batch of find→replace entries (each one `re.sub`), edited as a single card
- `callback:lowercase-wordlist` — strip terminator and lowercase next word if it's in the wordlist
- `callback:map` — auto-built alternation of map keys (longest-first, case-insensitive); look up replacement
- `callback:dedup` — collapse adjacent punctuation runs (last non-comma wins; pure-comma run → single comma)
- `callback:upper` — capitalize after sentence terminator
- `terminal` — final `lstrip(" \t\r") + rstrip(" \t\r")`; always last (preserves leading/trailing `\n`)

The 14 seeded defaults handle orthography normalization (`ß`→`ss`), Whisper noise stripping, dictation (`Punkt`→`.`, `neue Zeile`→`\n`, …), and tidy spacing/newlines/capitalization. They live in the committed **`config.json`** (the `PIPELINE_RULES` array, next to all the scalar defaults); `config.py` loads that file at startup. Each rule carries an optional `note` field documenting its rationale.

**Ordering invariants:** `dictation-map` multi-word phrases must precede their single-word components (the alternation regex is rebuilt longest-first, so the longest phrase wins); the `terminal` trim rule is always last.

**Editing — the dedicated editor at `/settings/pipeline`** (`/settings` keeps every
other section and links there in its place). One rule list
shows the **effective** pipeline (config.json overlaid by config.local.json). Edits
save to `config.local.json` via the page Save (gitignored, per-deployment). Each
rule carries an **origin badge**:

- `● factory` — matches `config.json`.
- `◆ edited` — in `config.json` but locally edited; offers `↺ reset` (discard the
  local edit) and `⇪ promote`.
- `✚ local-only` — not in `config.json`; offers `× delete` and `⇪ promote`.

**Promote** writes a rule (or, via *Promote all changes to config.json*, the whole
list) into the committed **`config.json`** — a diff dialog confirms first. Since
`config.json` is git-tracked, `git commit && git push` then ships the change to
every deployment. After *Promote all* you're offered to **clear the local override**
so `config.json` runs directly on this deployment too (otherwise the local snapshot
keeps shadowing it).

Factory rules cannot be deleted; uncheck `enabled` to disable. `config.json` is
required — if it is missing or malformed the service fails fast at startup; restore
it with `git checkout config.json`.

JSON response notes: `text` is the post-processed joined transcript. `segments[].text` and `words[].word` carry **raw** Whisper output (no post-processing applied to per-segment / per-word fields — multi-word dictation phrases like `"neue Zeile"` only resolve cleanly on the joined text).

## Stats dashboard

`http://localhost:8000/stats` shows a live dashboard updated over Server-Sent Events at ~1 Hz:

- **GPU**: name, util %, VRAM used/total, temp, power draw, SM clock, current performance state.
- **Host**: total CPU%, per-core mini-strip, RAM, free disk on the model cache drive.
- **Process**: PID, RSS, threads, uptime.
- **Loaded models** with per-model VRAM (NVML delta sample taken at construction time), warm/cold badge, and the cold-load history.
- **Request metrics**: in-flight transcriptions, p50/p95/p99 latency, endpoint counters, 5xx counts in 1m/5m/15m windows.
- **Recent transcriptions** ring (last 20) with model, audio length, wall-clock, real-time-factor, words emitted.
- **Usage history** (`/stats/usage`): the v2 document — totals and a headline strip, stacked columns by kind (or lines by user / key / model / stage), a comparison window (`compare=prev|yoy`), pipeline-stage adoption and speed, a weekday × hour grid of GPU seconds, and a leaderboard with sessions, GPU seconds and RTF. The header's scope bar (range presets, custom span, kind and "ran" chips, click-to-filter chips) scopes every usage card and mirrors to the URL.
- **The tail** (`/stats/tail`): queue wait p50 / p95 (the inference semaphore is timed), a turnaround histogram with the queue-wait share, failures by stage and class (`policy_blocked`, `cuda_oom`, `timeout`, `cancelled`, `decode_failed`, `rejected`, `other`), per-model RTF, and deltas against the preceding window.
- **Jobs** (`/stats/jobs`): the recent-jobs table pages beyond the snapshot with kind / status / slow filters, shows the wait column and error class, expands a row into a timeline, and cancels a running job.
- **System-metrics history** (`/stats/history`): a 1 Hz sampler keeps one GPU/CPU/RAM reading every `STATS_SYSTEM_METRICS_SAMPLE_S` seconds in `STATS_SYSTEM_METRICS_DB` for `STATS_SYSTEM_METRICS_RETENTION_DAYS`; the sparklines switch between the live two-minute ring (scrubbable, hover-freeze) and 1 h / 24 h / 7 d history. Layout presets (ops / usage / both) and an explicit edit-layout mode with keyboard move / resize replace always-on dragging.

Sparklines are rendered with [uPlot](https://github.com/leeoniya/uPlot), vendored under `static/` so the page works **fully offline** — no CDN fetch at page-load. To update the bundled version, see `static/VENDOR.md`.

The `/stats` endpoint is user-tier allowlist-gated (`USER_WEBUI_ALLOWED_HOSTS`) plus a `stats` API key on the data endpoints. On a host without an NVIDIA GPU or with `nvidia-ml-py` missing, the GPU panel hides and the rest of the dashboard still works.

The nav row at the top of every page (logs ↔ stats ↔ quick-config ↔ captures ↔ reports ↔ settings) also surfaces three severity pills counting `WARNING` / `ERROR` / `CRITICAL` records since process start (bounded by a 2000-entry ring; restart resets to zero); clicking any pill jumps to `/logs` with that filter prefilled.

## Admin WebUI (optional)

A second WebUI at `/settings` lets you edit every setting from the browser, with hot-reload for safe knobs (transcribe params, dictation map, prompt) and an automatic service restart for cold ones (server port, log file, preload list).

**On by default** (`ADMIN_UI_ENABLED = true`). Set `WHISPER_ADMIN_UI=0` (or flip `ADMIN_UI_ENABLED` in `config.json`) and restart to unregister `/settings*` (plus `/quick-config`, `/captures`, `/reports`, which ride the same switch). Don't put the key in `config.local.json`: that file only accepts `AdminConfig` fields, and an unknown key makes the whole file fail validation — every override in it is then ignored (with a message on stderr at startup). The page opens at `http://localhost:8000/settings` from the server itself or any host in `ADMIN_WEBUI_ALLOWED_HOSTS`. Settings pinned by a `WHISPER_*` env var appear **read-only** (greyed out, badged with the variable name) since the environment takes precedence.

### Authentication: per-user API keys

The transcription endpoint and every WebUI page are gated by **per-user API keys**, not a shared token. Each key looks like `wk_<43-char base64>` (256-bit entropy); raw keys are SHA-256-hashed at rest and shown **once** on creation.

**Bootstrap.** On a fresh install with no admin key in the DB, the server starts in **OPEN mode**: a red banner appears on every WebUI page and a `WARNING` log line fires every 60 s. This is the operator's prompt to generate the first admin key. Two ways:

1. **In the UI** — open `/settings/api-keys`, click "+ add user" with admin=true, then "+ generate key", and copy the raw key from the show-once modal.
2. **Via env var** — set `WHISPER_BOOTSTRAP_ADMIN_KEY=wk_…` on first start. A `bootstrap-admin` user is created (or skipped if the same key hash is already present) with that exact raw key. Subsequent starts no-op. **Recommended for containers** — the server comes up already locked down, so OPEN mode never happens.

OPEN mode is a bootstrap state, not a deployment mode: the synthetic admin is handed out **only to callers in `ADMIN_WEBUI_ALLOWED_HOSTS`** (loopback by default) — the very allowlist that gates `/settings/api-keys`, so it never reaches past the surface you already had to open to create the first key. Everyone else gets the usual 401. Once at least one active admin key exists, the OPEN-mode banner disappears and 401 is returned to every unauthenticated caller.

**Using a key.** API clients and curl send `Authorization: Bearer wk_…` on every request — including the streaming WebSocket handshake. The WebUI instead exchanges the key **once** for an HttpOnly session cookie via `POST /auth/login` (server-side session rows, 30-day TTL by default — `SESSION_TTL_S`; CSRF double-submit cookie on mutating requests). On any 401 the full-page login gate re-prompts; `POST /auth/logout` ends the session. A browser page opening the WebSocket **cross-origin** (where that cookie is not sent) can offer the subprotocol `bearer.wk_…` instead — the only request header the browser WebSocket API lets a page set. The key is **never** accepted as a query parameter: the access log records the full request line.

**Lockout protection.** Revoking the last active admin key (or the last admin user) returns 409. Generate a second admin key first.

**Multi-user.** Each capture is tagged with the originating `user_id`. Non-admin users see only their own captures in `/captures`; admins see all and can filter by user. Merging captures into a training sample is locked to a single speaker — the server rejects any merge whose members span more than one user.

Other layers:
- **Feature flag**: `ADMIN_UI_ENABLED` (on by default; pin with `WHISPER_ADMIN_UI=1`) registers the routes. Set `WHISPER_ADMIN_UI=0` and `/settings*` returns 404.
- **Host allowlist**: `ADMIN_WEBUI_ALLOWED_HOSTS` keeps the admin endpoints reachable only from the configured CIDRs (loopback always implicit). User pages use the separate `USER_WEBUI_ALLOWED_HOSTS` (default open).
- **Server-side validation**: every payload is validated against `config_store.AdminConfig` (Pydantic v2).
- **Auto-restart**: when a "cold" setting changes (server port, log file, preload list, …), a confirmation modal asks whether to restart the service. WinSW relaunches the wrapper; the page polls `/v1/models` until back up.

Edits land in **`config.local.json`** in the data dir (default `/data/config.local.json` — see [Configuration](#configuration) for `WHISPER_DATA_DIR` / `WHISPER_CONFIG_LOCAL`; gitignored). See `config.local.example.json` for the schema. The one exception is the Pipeline section's **promote** action, which writes the committed **`config.json`** instead (see [Post-processing pipeline](#post-processing-pipeline)).

### Per-identity config overrides

Beyond the global and per-model layers, decode / streaming / output / pipeline-rule settings can be overridden **per user, per API key, and per reusable profile** — so many users can share one deployment without re-flashing every device. Managed on the dedicated **`/settings/overrides`** page; bound to users & keys in-context on **`/settings/api-keys`** (a `⚙ overrides` / `⚙ config` drawer per user / key). Load-time model fields (device, compute type, workers…) are **never** per-identity — a model is loaded once for everyone.

- **Profiles** — named, reusable override bundles (e.g. `low-latency`). Assign an *ordered* list to a user or key; **earlier wins** on a conflicting field.
- **Direct overrides** — a per-user or per-key blob layered on top of its profiles for one-offs.
- **Precedence** (most → least specific): `per-key direct → per-key profiles → per-user direct → per-user profiles → per-model → global → library`. The first identity layer that sets a field owns its value **and** its lock.
- **Clearing vs inheriting** — a per-request field that is *absent* inherits the resolved layer; a field that is *present but empty* is an explicit **clear** that overrides it: `language=""` → auto-detect, `translate_to=""` → no translation targets, `translation_glossary=""` → no glossary, `prompt=""` → no initial prompt, and `decode_overrides.suppress_tokens: ""`/`[]` → suppress nothing (a JSON `null` on a boolean override inherits). Profile / per-model text fields behave the same way: `""` means "none", e.g. a blank `PREPEND_PUNCTUATIONS` disables prepend-splitting and a blank `SUPPRESS_TOKENS` keeps only `SUPPRESS_CHARS`. This is what the desktop client's *clear* vs *reset* controls send.
- **Per-field locking** — mark a field 🔒 to forbid the client's per-request `decode_overrides` (and `language`/`prompt`) from changing it; the dropped keys are surfaced in `verbose_json.overrides_ignored` (batch) / the `ready` frame (streaming), never silently ignored. A useful compute cap on a shared server (e.g. lock `BEAM_SIZE`).
- **Effective-config Explorer** — the `/settings/overrides` *Explorer* tab is a what-if simulator: pick a user (+ key, + model, + a simulated client override) and see the full resolution **waterfall** per field — which layer won, what was overridden, what is locked.
- **Pipeline rules** resolve analogously (first layer that force-on/off a rule decides; otherwise per-model, then the global `enabled`). Capture **reprocess** re-runs the pipeline under the capture **owner's** rules.
- **Live changes apply without reconnect** — batch requests resolve identity per request; a live dictation **WebSocket** re-resolves at each utterance boundary whenever the config version changes (any binding / profile / settings edit), so edits land on the next utterance. Session-shaping `STREAMING_*` (chunking / VAD / endpointer timing) and word-timestamp gating stay fixed for the connection — change those, then reconnect. Every batch & stream log block carries an **`Identity`** section (resolved user / key + applied profiles, or `overrides (none — inherits …)`) so a missing binding is obvious at a glance.

The same `OVERRIDE_PROFILES` JSON can be pinned via `WHISPER_OVERRIDE_PROFILES` (see `.env.example`). Per-user bindings live in the user's permissions JSON; per-key bindings in the `api_keys.config` column (added by an idempotent migration on first start).

## Brand

The mark is an **audio waveform skewed forward** — *whisper* (the equalizer bars) meets *faster* (the rightward lean reads as motion / fast-forward) — on a rounded terminal-dark tile, in the WebUI's GitHub-dark palette (cyan `#79c0ff` → green `#7ee787` on `#0d1117`). It is deliberately **this service's own mark** and is untouched by the family alignment below.

The wordmark follows the **brand-family grammar shared with
[faster-whisper-frontend](https://github.com/v3DJG6GL/faster-whisper-frontend)**: light `faster`
(Hubot Sans 430) in ink + bold `whisper` (730) in the product accent, then an accent `>` prompt
before the tracked-caps role label in Geist Mono (`> BACKEND` here, `> FRONTEND` there). Each
product keeps its own accent — terminal green for the backend, amber for the frontend — so the
family reads through structure, the product through colour. Fonts are vendored in `static/`
(see `static/VENDOR.md`); brand assets + regen tooling live in `docs/brand/`.

**Logo variants** — each designed to work on dark *and* light backgrounds:

- **Full logo** — icon + stacked wordmark (`fasterwhisper` over `> BACKEND`); the hero above.
- **Compact** — single line, `[icon] fasterwhisper > backend`, for headers and tight spaces. The admin WebUI's sticky header uses this form.
- **Icon only** — the waveform tile alone (favicon, app tile, anything ≤ ~80 px).
- **Monochrome** — single-colour, tile-less bars for one-colour or print contexts.

**Type spec:** name in Hubot Sans (430/730 weight pair, −0.025em); label in Geist Mono, 500,
tracked caps at 0.14em, ⅔ of the name size; the green `>` and `whisper` are the wordmark's only
accents. The same icon is the favicon and sits in the sticky header of every admin page.

Assets:

| File | Use |
| --- | --- |
| `docs/brand/logo-{dark,light}.svg` | full logo, vector — wordmark converted to paths, renders everywhere without fonts (`gen-logo-svg.py` regenerates) |
| `docs/brand/logo-{dark,light}.png` | full logo, raster @2× (used in this README); rendered from `docs/brand/logo.html` |
| `docs/brand/icon.svg` / `icon.png` | icon only (copy of `static/logo.svg` / 512 px raster) |
| `static/logo.svg` | the canonical icon the WebUI serves |
| `static/favicon.svg` | simplified 3-bar icon for small sizes |
| `static/favicon.ico`, `favicon-16.png`, `favicon-32.png`, `apple-touch-icon.png` | raster fallbacks (Safari / legacy) |

## Files

```
main.py                    FastAPI app + post-processing pipeline + log viewer
config.py                  Loads config.json factory defaults, layers config.local.json + WHISPER_* env on top
config.json                Committed factory defaults for EVERY setting + the pipeline rules (single source of truth)
config_store.py            Admin-WebUI persistence layer (Pydantic schema, atomic writes)
effective_config.py        Layered per-identity config resolution (key/user/profile overrides, locks)
admin_routes.py            Admin /settings + /settings/pipeline pages & endpoints (disable with WHISPER_ADMIN_UI=0)
overrides_routes.py        /settings/overrides admin page & API (profiles, explorer/resolve)
api_keys_store.py          users + api_keys SQLite store (SHA-256 hash, soft revoke, O(1) lookup)
api_keys_routes.py         /settings/api-keys admin UI for per-user key management
sessions_store.py          Durable browser-session store — the cookie layer on top of api_keys_store
auth.py                    Auth deps — get_current_user / require_admin / require_page + OPEN-mode loop
quick_config_routes.py     /quick-config end-user rule editor + the /v1/pipeline-rules client API
quick_config_state.py      Tokenization + SSE broadcast layer for /quick-config recent transcriptions
stats_routes.py            /stats dashboard endpoints + HTML page (always on, allowlist-gated)
metrics.py                 In-process request metrics (counters, latency ring, recent transcriptions)
system_stats.py            GPU + host snapshot (pynvml + psutil; degrades gracefully if NVML missing)
usage_store.py             Durable per-key / per-user usage rollup (/v1/usage, /stats/usage)
client_settings_store.py / client_settings_routes.py
                           Per-account desktop-client settings blob + /v1/client-settings sync API
transcriptions_store.py    Durable store for recent transcription traces (/quick-config recent)
web_common.py              Shared helpers: allowlist gate, nav HTML + severity pills, login gate / OPEN-mode banner
restart_service.py         Detached self-restart helper (os.execv re-exec on Linux/macOS, WinSW on Windows)
streaming_routes.py        WebSocket /v1/audio/transcriptions/stream + /dictate demo page
streaming_session.py       Per-connection streaming dictation state machine
streaming_transport.py     Streaming audio decoders (raw PCM passthrough, ffmpeg WebM/Opus)
streaming_vad.py           Streaming endpointing (two-tier Silero/energy VAD)
streaming_localagreement.py LocalAgreement-2 hypothesis stabilization
translation.py             Text-to-text translation stage: GGUF models via llama.cpp (LRU cache, prompt families, guards)
diarization.py             Speaker diarization via pyannote.audio (optional install: requirements-diarize.txt)
bgm_separation.py          Background-music separation via audio-separator / UVR MDX-Net (optional: requirements-bgm.txt)
url_download.py            Transcribe-from-URL: fetch a client-supplied media link with yt-dlp
url_media_store.py         Short-term retention store for URL-downloaded audio
net_policy.py              The one definition of which outbound addresses the server refuses
ytdlp_plugins/             SSRF guard for yt-dlp: same policy on every hop, pinned DNS, http(s) only
home_routes.py             Root landing hub — GET / serves the WebUI's front door
preload.py                 Model preloading: plan registry, warm leases and the single load worker
preload_routes.py          POST /v1/models/preload — warm the models a job is about to need
jobs.py                    Central registry of running jobs (transcribe / dictate / translate / download / preload)
download_progress.py       Model-download progress (huggingface_hub tqdm shim + capture scope)
receipt_hold.py            Holds a dictation's request receipt open until its translation lands
rate_limit.py              Shared per-identity limiters + the typed 429 envelope they raise
store_common.py            Shared store hardening (0600/0700 chmod) + log_safe control-char screen
model_sizes.py             Persisted ledger of measured model sizes + free-memory fit check
build_info.py              Build + runtime identity — the version string clients and the WebUI display
audio_transcode.py         In-process audio transcoder (PyAV — no ffmpeg-on-PATH needed)
audio_vad_trim.py          Silence-trim WAVs with the bundled Silero VAD
audio_merge.py             stdlib-wave PCM splicer for duration-capped training-sample packing (default ≤29.9 s)
captures_store.py          Capture rows + audio fanout, retention, eviction
capture_samples_store.py   Duration-capped training samples (default ≤29.9 s) built from consecutive same-speaker captures
captures_routes.py         /captures page + samples/merge/reprocess API
captures_merge_proposer.py Auto-merge proposer for /captures curation
captures_reapply.py        Background job: re-run current pipeline rules over existing captures
captures_vad_reprocess.py  Background job: re-merge sample audio with current silence settings
reports_store.py / reports_routes.py
                           User-submitted transcription error reports + admin triage
regex_guard.py             Out-of-process guard for user-authored pipeline regexes
text_corrections.py        Shared schema for word-correction chips
config.local.json          Runtime overrides written by the admin UI (gitignored, optional)
config.local.example.json  Example overrides file
.env.example               Documented list of every WHISPER_* env var + defaults (copy to .env)
test.py                    Manual test client (OpenAI SDK compatibility)
install-service.ps1        Windows Service installer (WinSW-based, self-elevating, auto-bootstraps venv)
uninstall-service.ps1      Windows Service uninstaller
install-service.sh         Linux systemd installer (self-elevating, auto-bootstraps venv); --gpu adds CUDA wheels
uninstall-service.sh       Linux systemd uninstaller
Dockerfile / Dockerfile.gpu / .dockerignore
docker-compose.yml / docker-compose.gpu.yml   CPU base + GPU overlay (NVIDIA)
                           CPU container image + compose (named volumes for state); run on any OS
requirements.txt           Base (CPU, cross-platform) deps; transitive resolved by pip
requirements-heavy.txt     The large compiled wheels, `-r`-included by requirements.txt (own Docker layer)
requirements-gpu.txt       NVIDIA CUDA wheels (opt-in, additive)
requirements-dev.txt       Test deps (pytest)
requirements-convert.txt   Deps for converting HF models to CTranslate2 (opt-in)
requirements-translate.txt Text-to-text translation deps: llama-cpp-python (opt-in; prebuilt wheel indexes documented inside)
requirements-diarize.txt   Speaker-diarization deps: pyannote.audio + torch (opt-in)
requirements-bgm.txt       Background-music separation deps: audio-separator (opt-in; GPU paths swap in the [gpu] extra)
pytest.ini                 Test discovery config (pytest -q from repo root)
.coveragerc                Coverage config; CI runs pytest --cov-fail-under against it
renovate.json              Renovate dependency-update policy (grouping, automerge rules)
.forgejo/workflows/ci.yml  CI: test suite on Linux + Windows; the v* tag run builds and publishes the registry images
.forgejo/workflows/release.yml      Tag-first release flow: mints the next v* tag off a green main push ("[skip release]" opts out)
.forgejo/workflows/mirror-ghcr.yml  Mirrors the published images to ghcr.io; needs the GHCR_USER/GHCR_TOKEN secrets, else it no-ops
static/                    Brand assets (logo.svg, favicon.*) + vendored uPlot/GridStack (offline /stats)
.gitignore / .gitattributes
logs/                      Created at first run; rotates at 10 MB × 10 files
```
