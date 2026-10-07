"""WebSocket endpoint for live (streaming) dictation.

`ws[s]://HOST/v1/audio/transcriptions/stream` — a second entry point alongside the
batch `POST /v1/audio/transcriptions`. It reuses the same model cache
(`transcription.models._get_or_load_model`), per-model config (`cfg_for`), and post-processing pipeline
(`pipeline.engine._postprocess_text`) — none of which are modified — and drives
them through :class:`streaming_session.StreamSession` (LocalAgreement-2
stabilized partials, append-only post-processed finals).

Handshake credentials, tried in order: `Authorization: Bearer <key>` (native
clients), a `bearer.<key>` entry in Sec-WebSocket-Protocol (browser clients —
the only request header they may set on a WebSocket; echoed back on accept),
then the WebUI session cookie. Never a query parameter: the access log records
the full request line.

Protocol (see streaming_session for the emission contract):
  client → server:
    1. first TEXT frame: JSON config
       {"type":"config","model":..,"language":..,"response_format":"json|verbose_json",
        "audio":{"format":"pcm_s16le","sample_rate":16000}}
    2. BINARY frames: raw 16 kHz mono s16le PCM  (encoded formats: phase E)
    3. control TEXT frames: {"type":"flush"} | {"type":"stop"}
  server → client:
    {"type":"loading",model}  (keepalive while a cold model loads — may repeat)
    {"type":"ready",..} / {"type":"partial",committed,pending} /
    {"type":"final",utterance?,committed,tail,forced?,last?,flush?} /
    {"type":"error",code,message}
    (final: both are full strings. Within one document ``committed + tail``
     only grows — every final extends the previous one, so the client may type
     just the difference; ``committed`` is locked, ``tail`` the newest sentence
     and only a display hint (it does not change later either). A trailing
     fragment a dictation rule could still join with what follows ("neue" of
     "neue Zeile", "Komma" before a line break, "120 Schrägstrich") may be
     withheld until the next final, the ``boundary``, a ``flush`` or the close;
     when nothing follows, it arrives in a release final: ``flush: true`` and
     no ``utterance``, sent BEFORE the ``boundary`` it precedes.)
    {"type":"utterance",utterance,state,reason?}  lifecycle of the utterance the
     server is holding: state "open" (once, at min-speech), "decoding" (once,
     before the final decode), "dropped" (reason no_speech|empty|error). Every
     announced utterance ends in exactly one of: a ``final`` with the same
     ordinal, or "dropped". Clients must ignore states they do not know. The
     closing ``last`` final is the document, not an utterance: its ordinal
     belongs to none.
    {"type":"captured",id,utterance}  receipt for a stored capture (after its final)
    {"type":"boundary",utterance,separator}  long-silence hard break: fresh document
    {"type":"closing"}  the server is done; the socket closes next

The handshake's same-origin check is auth/hosts.py's, shared with main's
CSRF middleware; this module never imports main.
"""

import asyncio
import json
import logging
import os
import random
import re
import shutil
import tempfile
import time
import uuid
import wave

import numpy as np
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials

from faster_whisper_backend.auth import dependencies as auth
from faster_whisper_backend.auth import hosts as auth_hosts
from faster_whisper_backend.settings import config as cfg
from faster_whisper_backend.settings import effective_config
from faster_whisper_backend.settings import schema as settings_schema
from faster_whisper_backend.settings import version as settings_version
from faster_whisper_backend.transcription import decode_trace
from faster_whisper_backend.core import jobs
from faster_whisper_backend.stats import metrics
from faster_whisper_backend.auth import rate_limit
from faster_whisper_backend.transcription import receipt_hold
from faster_whisper_backend.transcription import segment_guards
from faster_whisper_backend.core import store_common
from faster_whisper_backend.core import templates
from faster_whisper_backend.core import web_common
from faster_whisper_backend.pipeline import engine as pl_engine
from faster_whisper_backend.transcription import guards as tx_guards
from faster_whisper_backend.transcription import models as tx_models
from faster_whisper_backend.transcription import progress as tx_progress
from faster_whisper_backend.transcription import receipt as tx_receipt
from faster_whisper_backend.streaming.session import CloseAbort, StreamConfig, StreamSession
from faster_whisper_backend.streaming.transport import ENCODED_FORMATS, RAW_FORMATS, make_transport
from faster_whisper_backend.streaming.vad import SAMPLE_RATE, make_endpointer

logger = logging.getLogger(__name__)

# Floor between two backlog/oversize WARNINGs on one session. Both branches are
# entered at a rate the client controls, and the log is a fixed-size rotating
# chain — unthrottled they are a way to erase the audit trail.
_SHED_LOG_INTERVAL_S = 10.0
# Absolute cap on QUEUED items in a session's audio queue. The byte cap bounds
# bytes, not tuples: control items carry no PCM and a raw-PCM client may send
# 2-byte frames, so without this a client could queue millions of tuples under
# the byte cap (each PCM frame may re-arm one queued flush).
_HARD_CAP_ITEMS = 4096

router = APIRouter()


async def _safe_ws_send(ws: WebSocket, message: dict, *, close: bool = False) -> bool:
    """Send a JSON message, swallowing the errors raised when the peer has already
    disconnected (e.g. the page was reloaded mid-dictation). Without this, the
    session-close drain's final send hits a closed socket and uvicorn raises
    ``RuntimeError: Unexpected ASGI message 'websocket.send' after ... close``,
    surfacing as a noisy traceback. Returns False if the send was dropped.
    With ``close=True`` also closes the socket under the same guard (refusal
    paths: the peer may already be gone, and Starlette raises RuntimeError on
    close after a failed send)."""
    try:
        await ws.send_json(message)
        if close:
            await ws.close()
        return True
    except (RuntimeError, WebSocketDisconnect):
        return False


def _write_pcm16_wav(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """Write a float32 mono [-1,1] buffer to a temp 16-bit PCM WAV and return its
    path. Used to hand a streamed utterance's audio to the captures pipeline
    (which re-transcodes any source file to its canonical 16 kHz mono WAV)."""
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    # "whisperup-" is one of the prefixes the hard-restart TMPDIR sweep
    # (admin/restart_service.py) reclaims, so an orphaned capture WAV does not
    # outlive the process.
    fd, path = tempfile.mkstemp(prefix="whisperup-", suffix=".wav")
    os.close(fd)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return path

# Active sessions, for the /stats gauge and the max-session cap (phase D).
_active_sessions: set[str] = set()

# Per-identity slice of that cap. Modelled as an InFlight rather than a second
# hand-rolled dict so it inherits the shared 0 = unlimited path, the typed
# refusal and — the one that matters here — reset_all(), which keeps a leaked
# slot in one test from refusing connections in the next.
_stream_sessions = rate_limit.InFlight(
    config_field="STREAMING_MAX_SESSIONS_PER_USER",
    default_max=4,
    message="you already have {limit} live sessions open",
)

# WebSocket close codes (4000-4999 = application-defined). A refusal must
# ACCEPT before it closes with one of these: per ASGI a `websocket.close` sent
# before `websocket.accept` is a handshake rejection, which uvicorn turns into a
# bare HTTP 403 — the code never reaches the wire and a browser sees 1006. See
# _refuse.
_WS_UNAUTH = 4401
_WS_DISABLED = 4503
_WS_TOO_MANY = 4429
_WS_BAD_ORIGIN = 4403
_WS_IDLE_TIMEOUT = 4408  # client sent no audio for STREAMING_IDLE_TIMEOUT_S

# The client decode_override keys the server actually honors (tx_models._apply_
# decode_overrides consumes the decode ones, _client_stream_values the live-dictation
# ones; every other key is discarded). Bound once at import from the public
# settings_schema.CONFIG_TO_CLIENT_KEY registry so
# the handshake can narrow the client's dict — and without retaining an
# unbounded, connection-lifetime dict of attacker-chosen keys.
_CLIENT_OVERRIDE_KEYS: frozenset[str] = frozenset(
    settings_schema.CONFIG_TO_CLIENT_KEY.values())


# The handshake `language`: a Whisper code (`de`, `haw`) or a BCP-47-ish tag.
_LANGUAGE_RE = re.compile(r"\A[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8}){0,2}\Z")


def _handshake_language(raw: object) -> "str | None":
    """The handshake `language` as the Whisper code the decoder accepts.

    Tri-state, same as batch: not a string → None (inherit DEFAULT_LANGUAGE);
    blank → "" (explicit auto-detect, the client's cleared state). Otherwise
    the value is lower-cased and kept when it, or a locale tag's primary
    subtag (`de-DE` → `de`, `pt-BR` → `pt`), is one of Whisper's codes —
    faster-whisper's tokenizer raises on anything else (`DE`, `en-US`), which
    in streaming fails every partial and final with no error frame, and the
    language-tagged pipeline rules are keyed by these codes too. Anything
    else is unusable, so it is treated as absent (None), as is anything not
    even shaped like a language code (CR/LF would reach the WARNING log raw).
    """
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    if not value:
        return ""
    if not _LANGUAGE_RE.match(value):
        return None
    value = value.lower()
    for cand in (value, value.split("-", 1)[0]):
        if cand in settings_schema.WHISPER_LANGUAGE_CODES:
            return cand
    return None


async def _refuse(ws: WebSocket, code: int, reason: str) -> None:
    """Refuse a handshake so the client learns WHY: accept (echoing the bearer
    subprotocol, or a browser fails the handshake before it sees anything),
    then close with the application code and a short reason. Closing
    pre-accept would discard the code as a plain HTTP 403. No `error` frame
    first: the first receive must raise the close, which is the contract the
    dictate page's onclose branches and the clients' handshake tests rely on.
    Nothing is allocated before this runs, so the accept is cheap; a peer that
    already went away is simply ignored."""
    try:
        await ws.accept(subprotocol=_ws_bearer_subprotocol(ws) or None)
        await ws.close(code=code, reason=reason)
    except Exception:  # noqa: BLE001 — refusal is best effort; the peer may be gone
        pass


async def _receive_idle(ws: WebSocket, timeout_sec: float):
    """`ws.receive()`, but abandoned after `timeout_sec` so an idle / abandoned /
    dead connection can't hold its session slot forever. A value <= 0 disables
    the timeout (plain receive). On expiry raises TimeoutError.

    Callers pass the REMAINING budget, not the full timeout: the deadline the
    route enforces is "no AUDIO for N seconds", so a client that keeps sending
    non-audio frames (empty binary frames, unrecognised control JSON) no longer
    re-arms it. See the producer loop in transcribe_stream."""
    if timeout_sec and timeout_sec > 0:
        return await asyncio.wait_for(ws.receive(), timeout_sec)
    return await ws.receive()


# Sec-WebSocket-Protocol is the ONE request header the browser WebSocket API
# lets a page set, so it is how a cross-origin browser client (where the session
# cookie is not sent) carries its key. Unlike a query parameter it never reaches
# an access log. RFC 6455 requires the server to echo the accepted value — see
# the ws.accept() in transcribe_stream.
# `bearer.<key>` stays a single RFC 7230 token because generate_raw_key() is
# `wk_` + secrets.token_urlsafe (base64URL: alphanumerics, `-`, `_`), and `.`
# is itself a tchar — so no escaping is needed and none is done.
_WS_BEARER_SUBPROTOCOL = "bearer."


class _CredentialRevoked(CloseAbort):
    """Raised out of a streaming decode when the connection's credential no
    longer resolves to the identity that opened it (key/user revoked, key
    rotated, session signed out). Aborts the decode BEFORE the model, the
    captures row, the trace row and the usage row — see _refresh_ident."""


def _ws_bearer_subprotocol(ws: WebSocket) -> str:
    """The `bearer.<raw_key>` entry from the handshake's requested subprotocol
    list, verbatim (so it can be echoed back on accept), or ""."""
    for entry in (ws.headers.get("sec-websocket-protocol") or "").split(","):
        entry = entry.strip()
        if entry.startswith(_WS_BEARER_SUBPROTOCOL) and entry != _WS_BEARER_SUBPROTOCOL:
            return entry
    return ""


def _ws_credentials(ws: WebSocket) -> "HTTPAuthorizationCredentials | None":
    """Build bearer credentials from the WS handshake: the Authorization header
    (native clients), else a `bearer.<key>` subprotocol (browser clients)."""
    creds = auth.bearer_credentials(ws)
    if creds:
        return creds
    sub = _ws_bearer_subprotocol(ws)
    if sub:
        return HTTPAuthorizationCredentials(
            scheme="Bearer", credentials=sub[len(_WS_BEARER_SUBPROTOCOL):])
    return None


def authenticate_ws(ws: WebSocket) -> "dict | None":
    """Resolve the WS caller to a user record (or None). Reuses the canonical
    non-raising auth core — Starlette's WebSocket exposes .cookies/.headers/.state,
    so the cookie + bearer paths work unchanged. Open mode → synthetic admin, and
    only from the admin host allowlist (auth.open_mode_host_ok)."""
    return auth._resolve_user(ws, _ws_credentials(ws))


def _client_stream_values(overrides: dict, ident) -> "tuple[dict, list[str]]":
    """The live-dictation knobs this connection's decode_overrides set
    (settings_schema.STREAM_ONLY_CLIENT_KEYS), as {STREAMING_* field: value} for
    _stream_config / make_endpointer, plus a note per adjustment made.

    Locked keys (ident.locked_client_keys, which also covers the master gate)
    are skipped — the handshake already reports them in overrides_ignored.
    Numbers are clamped to the field's own bounds (client_key_bounds), the
    separator is kept verbatim up to its max length ("\\n" included).

    The silence gates only work as a pair, so when a client value takes part:
    inner must stay below outer (inner >= outer stretches the commit wait to
    the forced-commit cap, session.py) — else inner = outer - 50; and a hard
    break must be off (0) or longer than outer (the idle-silence clock
    survives the finalize, so hard break <= outer would break after every
    utterance) — else hard break = outer + 1000. Server-only values are the
    admin's and are never adjusted; a LOCKED partner is never adjusted either
    — a client outer that conflicts with one is dropped instead."""
    bounds = settings_schema.client_key_bounds()
    field_of = {ck: f for f, ck in settings_schema.CONFIG_TO_CLIENT_KEY.items()}
    values: dict = {}
    for key in settings_schema.STREAM_ONLY_CLIENT_KEYS:
        if key not in overrides or key in ident.locked_client_keys:
            continue
        raw, b = overrides[key], bounds[key]
        if b["kind"] == "int":
            v = tx_models._clamp_int(raw, b["min"], b["max"])
        elif b["kind"] == "float":
            v = tx_models._clamp_float(raw, b["min"], b["max"])
        else:
            v = raw[:b["maxlen"]] if isinstance(raw, str) else None
        if v is not None:
            values[field_of[key]] = v
    notes: list[str] = []
    inner_f, outer_f = "STREAMING_VAD_INNER_SILENCE_MS", "STREAMING_VAD_OUTER_SILENCE_MS"
    hb_f = "STREAMING_HARD_BREAK_SILENCE_MS"

    def eff(field):
        return int(values[field] if field in values
                   else effective_config.cfg_for(None, field, ident))
    locked = {f for f in (inner_f, hb_f)
              if settings_schema.CONFIG_TO_CLIENT_KEY[f] in ident.locked_client_keys}
    if outer_f in values:
        inner, hb = eff(inner_f), eff(hb_f)
        if ((inner_f in locked and inner >= values[outer_f])
                or (hb_f in locked and hb and hb <= values[outer_f])):
            notes.append(f"outer {values.pop(outer_f)} ms dropped "
                         f"(locked inner {inner} / hard break {hb} ms)")
    outer = eff(outer_f)
    if inner_f in values or outer_f in values:
        inner = eff(inner_f)
        if inner >= outer:
            values[inner_f] = max(0, outer - 50)
            notes.append(f"inner {inner}→{values[inner_f]} ms (below outer {outer})")
    if hb_f in values or outer_f in values:
        hb = eff(hb_f)
        if hb and hb <= outer:
            values[hb_f] = outer + 1000
            notes.append(f"hard break {hb}→{values[hb_f]} ms (above outer {outer})")
    return values, notes


def _stream_config(cfg_for, ident=None, client: "dict | None" = None) -> StreamConfig:
    # Per-identity override (ident) > global. STREAMING_* are not per-model, so
    # model_id=None skips cfg_for's per-model layer — one resolver for every
    # STREAMING_* knob (the route resolves its siblings the same way). `client`
    # holds this connection's own values (_client_stream_values), which win.
    client = client or {}

    # No fallback argument: cfg_for always resolves the schema default, so a
    # literal here would never be read and would only drift from settings/config.py.
    def g(name):
        field = "STREAMING_" + name
        return client[field] if field in client else cfg_for(None, field, ident)
    # The trim pair only works with keep < trim: keep >= trim puts the cut at
    # or before the buffer start, so the trim never fires and the buffer grows
    # to MAX_BUFFER_S. AdminConfig's validator only sees the GLOBAL pair; a
    # profile / identity value can still cross it, so the resolved pair is
    # clamped here, like the VAD inner/outer pair in _client_stream_values.
    trim = float(g("BUFFER_TRIM_S"))
    keep = float(g("BUFFER_TRIM_KEEP_S"))
    if keep >= trim:
        keep = trim - 1.0
    return StreamConfig(
        sample_rate=SAMPLE_RATE,
        # Public config keys (the g("…") suffix, after STREAMING_) may differ from
        # the internal StreamConfig field names — this adapter is the seam.
        min_chunk_ms=int(g("PARTIAL_INTERVAL_MS")),
        min_speech_ms=int(g("GATE_MIN_SPEECH_MS")),
        vad_min_silence_ms=int(g("VAD_INNER_SILENCE_MS")),
        commit_silence_ms=int(g("VAD_OUTER_SILENCE_MS")),
        hard_break_silence_ms=int(g("HARD_BREAK_SILENCE_MS")),
        hard_break_separator=str(g("HARD_BREAK_SEPARATOR")),
        forced_commit_sec=float(g("FORCED_COMMIT_S")),
        buffer_trim_sec=trim,
        buffer_trim_keep_sec=keep,
        max_buffer_sec=float(g("MAX_BUFFER_S")),
        rms_gate_dbfs=float(g("GATE_RMS_DBFS")),
        prompt_words=int(g("PROMPT_WORDS")),
    )


def _build_transcribe_kwargs(model_name: str, *, final: bool,
                             prompt: str, want_words: bool,
                             language: "str | None" = None, model_obj=None,
                             overrides=None, ident=None) -> dict:
    """Assemble model.transcribe kwargs for a streaming decode.

    Both partial and final decodes pull the SAME per-model config as the batch
    route (via ``tx_models.assemble_transcribe_kwargs``) — hotwords, suppress_tokens/
    chars, prepend/append_punctuations, penalties, thresholds — so streaming
    output matches batch. The FINAL decode (the full committed utterance) is the
    batch decode's exact analogue and uses the assembler verbatim. The PARTIAL
    decode keeps all those quality knobs (they're ~free) and overrides ONLY the
    handful that must stay streaming-specific for latency/stability (see below).

    ``language`` is the per-connection language from the config handshake; it wins
    over the model's DEFAULT_LANGUAGE. Pinning it avoids faster-whisper auto-
    detecting per (short, growing) partial buffer, which is unstable — a brief
    German chunk can be mis-detected as e.g. Swedish."""
    cfg_for = effective_config.cfg_for
    # Present-but-empty is an explicit "auto-detect" (the client's cleared
    # state); only an ABSENT field inherits DEFAULT_LANGUAGE.
    lang = ((language if language is not None
             else cfg_for(model_name, "DEFAULT_LANGUAGE", ident)) or "").strip()
    _vad_filter = cfg_for(model_name, "VAD_FILTER", ident)
    vad_parameters = dict(
        min_silence_duration_ms=cfg_for(model_name, "VAD_MIN_SILENCE_MS", ident),
        speech_pad_ms=cfg_for(model_name, "VAD_SPEECH_PAD_MS", ident),
        threshold=cfg_for(model_name, "VAD_THRESHOLD", ident),
    ) if _vad_filter else None
    # The caller passes the session's rolling prompt (base_prompt — seeded from the
    # client prompt or DEFAULT_PROMPT — plus recent confirmed words), so use it
    # verbatim. An empty prompt means "no initial_prompt" (the client cleared it),
    # NOT "fall back to DEFAULT_PROMPT": DEFAULT_PROMPT is applied once, at the
    # base_prompt seed (see StreamSession construction / _refresh_ident).
    _prompt = prompt
    kwargs = tx_models.assemble_transcribe_kwargs(
        model_name, model_obj,
        language=lang, temperature=0.0,
        vad_filter=_vad_filter, vad_parameters=vad_parameters,
        want_word_ts=want_words, initial_prompt=(_prompt or None),
        overrides=overrides, ident=ident,
    )
    if final:
        # Full-utterance decode — identical to the batch route, EXCEPT
        # condition_on_previous_text (STREAMING_FINAL_CONDITION_ON_PREVIOUS_TEXT,
        # default off): when trailing non-speech survives into the buffer,
        # Whisper decodes the sub-second leftover after the last word as its
        # own window, and with conditioning ON that window sees the rolling
        # prompt + this utterance's own text — which it then confidently echoes
        # into the transcript. OFF gives the leftover window an empty text
        # context (nothing to echo); the first window still gets initial_prompt,
        # so cross-utterance context is unaffected. The pin overrides any
        # client decode_overrides value too (reported via overrides_ignored).
        kwargs["condition_on_previous_text"] = bool(
            cfg_for(model_name, "STREAMING_FINAL_CONDITION_ON_PREVIOUS_TEXT", ident))
        # best_of → STREAMING_FINAL_BEST_OF (default 1): a sampled fallback
        # rung runs all best_of candidates until the LAST one ends, so one
        # looping sibling made a 17-token answer cost 13.9 s (2026-09-19).
        # With one candidate the rung costs what its own answer costs. Batch
        # keeps BEST_OF; a client decode_overrides value still wins (unless the
        # identity locks that key — the assembler dropped it then). Presence is
        # not application: a null / unparseable value is dropped by the
        # assembler, which leaves the batch BEST_OF in kwargs, so pin then too.
        _locked = ident.locked_client_keys if ident is not None else frozenset()
        _client_best_of = tx_models._clamp_int(
            overrides.get("best_of") if isinstance(overrides, dict) else None,
            *tx_models._DECODE_INT_BOUNDS["best_of"])
        if _client_best_of is None or "best_of" in _locked:
            kwargs["best_of"] = int(
                cfg_for(model_name, "STREAMING_FINAL_BEST_OF", ident))
        return kwargs
    # PARTIAL decode: keep every quality knob the final/batch decode applies
    # (hotwords, suppress_tokens/chars, punctuation, penalties, thresholds — all
    # ~free: logit masks / beam shaping / post-processing). Override ONLY the few
    # knobs that genuinely matter for a fast, stable per-partial pass on a growing
    # buffer:
    #   • beam_size → STREAMING_PARTIAL_BEAM: the one real speed knob — partials
    #     re-decode the growing buffer many times per utterance, so the final's
    #     larger beam would roughly double that work.
    #   • temperature → STREAMING_PARTIAL_TEMPERATURE (default 0.0, no ladder): a
    #     fallback re-decode is a mid-stream latency spike; only the final needs
    #     the per-model TEMPERATURE ladder's robustness.
    #   • condition_on_previous_text → STREAMING_PARTIAL_CONDITION_ON_PREVIOUS_TEXT
    #     (default False): documented to loop on German finetunes, worst on short/
    #     growing buffers; the final uses the per-model CONDITION_ON_PREVIOUS_TEXT.
    #   • vad_filter off: the stream is already gated by our own VAD.
    kwargs["beam_size"] = int(cfg_for(model_name, "STREAMING_PARTIAL_BEAM", ident))
    kwargs["temperature"] = float(cfg_for(model_name, "STREAMING_PARTIAL_TEMPERATURE", ident))
    kwargs["condition_on_previous_text"] = bool(
        cfg_for(model_name, "STREAMING_PARTIAL_CONDITION_ON_PREVIOUS_TEXT", ident))
    kwargs["vad_filter"] = False
    kwargs["vad_parameters"] = None
    kwargs.setdefault("no_repeat_ngram_size", 3)  # greedy-safe loop guard
    return kwargs


def _trim_trailing_nonspeech(audio: "np.ndarray", pad_ms: int,
                             threshold: float, log_tag: str = "") -> "np.ndarray":
    """Cut trailing non-speech off a FINAL decode buffer (STREAMING_TAIL_TRIM_PAD_MS).

    The buffer always ends with >= commit_silence_ms of endpointer silence (that
    silence is what triggered the finalize), plus any noise the endpointer
    latched onto (phone ring, breath). Whisper hallucinates into such tails: the
    leftover audio after the last aligned word is re-decoded as its own
    zero-padded window and echoes the decode's text context. A Silero pass
    (the same VAD the decode-side vad_filter uses) finds the last speech and
    keeps only ``pad_ms`` beyond it — the pad absorbs VAD-vs-word-timestamp
    jitter so a genuine trailing word is never clipped. Only the tail is cut,
    so the decode's segment/word timestamps stay on the buffer timeline (the
    capture row still stores the full untrimmed utterance audio, and after a
    mid-utterance trim on_final re-bases the capture's segments by trimmed_sec
    onto the utterance timeline — captures are NOT on the buffer timeline).

    Returns ``audio`` unchanged when trimming is disabled (pad_ms <= 0), Silero
    is unavailable, no speech is found (the pre-decode gates own that case), or
    nothing lies beyond the pad."""
    if pad_ms <= 0 or getattr(audio, "size", 0) == 0:
        return audio
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except ImportError:
        return audio
    opts = VadOptions(threshold=float(threshold), min_silence_duration_ms=200,
                      speech_pad_ms=0, min_speech_duration_ms=0)
    speeches = get_speech_timestamps(audio, opts, sampling_rate=SAMPLE_RATE)
    if not speeches:
        return audio
    end = int(speeches[-1]["end"]) + (int(pad_ms) * SAMPLE_RATE) // 1000
    if end >= audio.shape[0]:
        return audio
    cut_sec = (audio.shape[0] - end) / SAMPLE_RATE
    # Every normal finalize trims ~a second of outer-gate silence (debug); a
    # multi-second cut means noise held the gate open (phone ring) — worth INFO.
    log = logger.info if cut_sec > 2.0 else logger.debug
    log("[stream %s] tail-trimmed %.2fs of trailing non-speech before final decode",
        log_tag, cut_sec)
    return audio[:end]


def _note_pinned_condition(req_overrides: dict, overrides_ignored: list) -> None:
    """The streaming final decode pins condition_on_previous_text from
    STREAMING_FINAL_CONDITION_ON_PREVIOUS_TEXT (see _build_transcribe_kwargs),
    so a client override for it is applied by the assembler and then
    overwritten. Say so in `overrides_ignored` instead of dropping it silently.
    Called at every site that rebuilds the list (handshake + ident refresh)."""
    # A null value is the client's "inherit" (the assembler skips it), so it
    # overrode nothing and is not reported.
    if (req_overrides.get("condition_on_previous_text") is not None
            and "condition_on_previous_text" not in overrides_ignored):
        overrides_ignored.append("condition_on_previous_text")


def _parse_translate_expect(conf: dict) -> "dict | None":
    """The client's declaration that a translation is coming on a SEPARATE
    request, so each utterance's log receipt is held until it lands and the two
    halves read as one block.

    Defensively typed like `override_profile`: a malformed handshake value
    degrades to "no declaration", never a crash, and every field is bounded
    here because it becomes server-authored output afterwards.

    `per_utterance` says whether a translate request arrives for EVERY
    utterance (live mode) or once for the whole transcript (stop-timing). Only
    the first can claim a per-utterance receipt; the second would leave every
    held receipt to the idle sweep, so those are logged inline instead and its
    one translate call gets a standalone receipt. Absent means a client from
    before the field, which behaved per-utterance — keep that.
    """
    tx = conf.get("translate_expect")
    if not isinstance(tx, dict):
        return None
    targets = tx.get("targets")
    if not isinstance(targets, list):
        return None
    clean = [t.strip()[:16] for t in targets if isinstance(t, str) and t.strip()][:8]
    if not clean:
        return None
    # An explicit null is a client serializing an unset option: absent, too.
    per_utt = tx.get("per_utterance")
    return {
        "targets": clean,
        "include_original": bool(tx.get("include_original")),
        "per_utterance": True if per_utt is None else bool(per_utt),
    }


@router.websocket("/v1/audio/transcriptions/stream")
async def transcribe_stream(ws: WebSocket) -> None:
    if not getattr(cfg, "STREAMING_ENABLED", True):
        await _refuse(ws, _WS_DISABLED, "live dictation is disabled on this server")
        return
    # Same-origin check, matching every unsafe-method HTTP route. main._csrf_mw
    # is registered with @app.middleware("http"), and BaseHTTPMiddleware passes
    # websocket scopes through untouched — so without this the handshake is the
    # one unsafe-side entry point in the tree with no Origin check. That matters
    # most in OPEN mode, where authenticate_ws resolves the synthetic admin from
    # the peer address alone: any page the operator visits could otherwise open
    # a streaming session against their own server. Absent Origin still means
    # allow, so non-browser clients (curl, SDKs, the desktop client) are
    # unaffected; a proxy that rewrites Host is handled by TRUSTED_ORIGINS
    # exactly as on the HTTP side.
    if not auth_hosts._origin_is_allowed(ws):
        auth_hosts._log_origin_rejected(ws)
        await _refuse(ws, _WS_BAD_ORIGIN, "handshake rejected (origin)")
        return
    user = authenticate_ws(ws)
    if user is None:
        await _refuse(ws, _WS_UNAUTH, "no valid credential")
        return
    # Per-identity cap FIRST: one client filling the pool while others are
    # locked out is the failure we actually see, and refusing it here means
    # the global cap below still has headroom for everybody else.
    _stream_key = rate_limit.identity_key(user, ws)
    try:
        _took_slot = _stream_sessions.acquire(_stream_key)
    except rate_limit.RateLimited as rl:
        logger.info(
            "[stream] refused: per-user cap "
            "(STREAMING_MAX_SESSIONS_PER_USER=%d) reached for %s",
            _stream_sessions.limit(), store_common.log_safe(_stream_key))
        await _refuse(ws, _WS_TOO_MANY, rl.message)
        return
    max_sessions = int(getattr(cfg, "STREAMING_MAX_SESSIONS", 10))
    if len(_active_sessions) >= max_sessions:
        # Release the slot just taken — this connection never becomes a
        # session, and the finally below is not reached from here.
        if _took_slot:
            _stream_sessions.release(_stream_key)
        logger.info(
            "[stream] refused: server-wide cap "
            "(STREAMING_MAX_SESSIONS=%d) reached", max_sessions)
        await _refuse(ws, _WS_TOO_MANY,
                      f"the server's {max_sessions} live sessions are all in use")
        return

    # A browser fails the handshake unless the server echoes one of the
    # subprotocols it offered, so hand back the bearer entry when the key
    # arrived that way; None otherwise (header / cookie clients offered none).
    # accept() awaits the client's connect frame and raises when the client
    # already went away — an aborted handshake must not keep the per-user
    # slot taken above (the finally that releases it only starts below).
    session_id = uuid.uuid4().hex
    _active_sessions.add(session_id)
    try:
        await ws.accept(subprotocol=_ws_bearer_subprotocol(ws) or None)
    except BaseException:
        _active_sessions.discard(session_id)
        if _took_slot:
            _stream_sessions.release(_stream_key)
        raise
    # Set only now: from here every exit runs the finally that releases it.
    _stream_held: "str | None" = _stream_key if _took_slot else None
    metrics.in_flight_transcriptions += 1
    # Per-utterance GPU-gate wait: the session's tasks share this context's
    # accumulator; each recorded utterance takes (and zeroes) it.
    metrics.seed_wait()
    # Central running-jobs registry: one "dictate" entry per live session
    # (no progress — a dictation has no defined end until the client stops).
    # `user` is the display name so the running row in /stats reads the same
    # as the finished rows beneath it.
    jobs.job_start("dictate", id=session_id, detail="live",
                   user=user.get("username") or user.get("user_id"),
                   key=user.get("key_id"), user_id=user.get("user_id"))
    session: "StreamSession | None" = None
    # One entry per lease taken by _load_with_keepalive. The final and partial
    # model are often the SAME name — that leases it twice and releases it
    # twice; the refcount is what makes that correct.
    _model_leases_held: "list[str]" = []
    transport = None
    consumer_task: "asyncio.Task | None" = None
    # Serializes EVERY ws send across the concurrent producer (receive loop) and
    # consumer (decode) tasks: two concurrent sends could interleave. Created here
    # (before the try) so the error handler below can guard its send even when
    # setup fails before the main body runs.
    send_lock = asyncio.Lock()
    try:
        # ---- handshake: first message is the JSON config (binary → defaults) ----
        # An idle/abandoned/dead connection must not hold a session slot, so bound
        # the wait for the client's first frame by the global idle timeout (the
        # per-identity value is resolved once we know the model + identity, below).
        idle_timeout = float(getattr(cfg, "STREAMING_IDLE_TIMEOUT_S", 0.0) or 0.0)
        try:
            first = await _receive_idle(ws, idle_timeout)
        except asyncio.TimeoutError:
            await ws.close(code=_WS_IDLE_TIMEOUT)
            return
        if first.get("type") == "websocket.disconnect":
            return
        conf = {}
        pending_audio: "bytes | None" = None
        if first.get("text") is not None:
            try:
                conf = json.loads(first["text"])
            except (ValueError, TypeError, RecursionError):
                # RecursionError is what json actually raises on a deeply
                # nested document, and it is a RuntimeError, not a ValueError.
                # A ~400 KB "[[[[..." frame (well inside ws_max_size) used to
                # escape to the handler's blanket except Exception, which logs
                # a full traceback carrying absolute paths, unthrottled, at a
                # rate the client picks.
                conf = {}
            if not isinstance(conf, dict):
                conf = {}
        elif first.get("bytes") is not None:
            pending_audio = first["bytes"]

        # A non-string `model` (a list, a number) would otherwise ride into
        # jobs.job_update and the model load before failing; treat it as absent.
        _req_model = conf.get("model")
        model_req = (_req_model if isinstance(_req_model, str) and _req_model.strip()
                     else "whisper-1")
        _req_language = conf.get("language")
        # Tri-state, same as batch: key ABSENT → None → inherit DEFAULT_LANGUAGE;
        # present-but-empty → "" → explicit auto-detect (the client's cleared state).
        # A value that is not (or does not reduce to) a Whisper code is treated
        # as absent and reported in the ready frame's overrides_ignored.
        req_language = _handshake_language(_req_language)
        _language_dropped = (req_language is None and isinstance(_req_language, str)
                             and bool(_req_language.strip()))
        response_format = conf.get("response_format", "json")
        # Per-connection initial prompt (the client's "Vocabulary / prompt"). Sentinel,
        # same as batch: key ABSENT → inherit DEFAULT_PROMPT; present (incl. "") →
        # use verbatim, where "" CLEARS the inherited prompt (no initial_prompt).
        _req_prompt = conf.get("prompt")
        prompt_provided = isinstance(_req_prompt, str)
        # Bounded to the same 2048 the admin-set DEFAULT_PROMPT it replaces
        # carries (config_store: Field(max_length=2048)) and the sibling
        # `hotwords` client override gets from tx_models._DECODE_STR_CAPS. Without a
        # cap the handshake frame — up to the websocket library's 16 MiB default
        # message size — is re-tokenised on every partial for the life of the
        # connection. Clamp rather than reject: an over-long prompt is a client
        # bug, not an attack, and the tail was never going to survive
        # faster-whisper's ~224-token prompt window anyway.
        req_prompt = _req_prompt.strip()[:2048] if prompt_provided else ""
        # Optional per-request decode overrides (the client's "decode overrides").
        # Applied to the FINAL decode (the batch analogue); partials keep their
        # streaming-specific beam/temp/condition/vad knobs (see _build_transcribe_kwargs).
        req_overrides = conf.get("decode_overrides")
        if not isinstance(req_overrides, dict):
            req_overrides = {}
        # Narrow to the keys the assembler actually honors. Every other key is
        # already discarded by tx_models._apply_decode_overrides, so no accepted
        # request changes behaviour — but without this the whole dict (up to the
        # 1 MiB ws_max_size frame) is captured by the decode closures and
        # retained for the life of the connection, and is re-walked on every
        # partial by the lock filter (which walks precisely when overrides are
        # NOT allowed, i.e. the hardened configuration).
        elif req_overrides:
            req_overrides = {k: v for k, v in req_overrides.items()
                             if k in _CLIENT_OVERRIDE_KEYS}
        # Optional per-request server override-profile name (the client's "Server
        # override profile"). Applied as the least-specific identity layer; honored
        # only when ALLOW_REQUEST_OVERRIDE_PROFILE is on, ignored if unknown. A
        # non-string handshake value is ignored rather than crashing the handshake.
        _req_profile = conf.get("override_profile")
        req_override_profile = (_req_profile.strip() or None
                                if isinstance(_req_profile, str) else None)
        # The client's own id for this session, so the usage job it reports
        # an outcome for afterwards is the one the utterances landed in. Same
        # alphabet as a batch progress id; malformed → the server's session
        # id, which the client never learns — its outcome then lands as a
        # stub session and the utterances are swept as unreported.
        _req_job = conf.get("client_job")
        usage_job_id = (_req_job if isinstance(_req_job, str)
                        and tx_progress._PROGRESS_ID_RE.match(_req_job) else session_id)
        # The client DECLARES that it will translate this session's utterances
        # on a separate request. Without a declaration the per-utterance
        # receipt is logged immediately, exactly as before — which is what
        # keeps an older client's log identical. With one, the receipt is
        # held until the translation lands so the two halves read as one
        # block. Defensively typed like override_profile above: a malformed
        # handshake value degrades to "no declaration", never a crash.
        translate_expect = _parse_translate_expect(conf)
        if translate_expect and not getattr(cfg, "TRANSLATION_ENABLED", False):
            # /v1/text/translations refuses with 403 before it parses the
            # capture key, so nothing would ever claim a parked receipt: every
            # utterance's log would wait out LOG_RECEIPT_HOLD_S and land late,
            # mis-annotated "no result". Treat the declaration as absent.
            translate_expect = None
        include_words = response_format == "verbose_json"
        audio_obj = conf.get("audio") or {}
        if not isinstance(audio_obj, dict):
            audio_obj = {}
        audio_fmt = audio_obj.get("format", "pcm_s16le")
        # Type-check before the set-membership test: an unhashable value (a
        # list, a dict) raises TypeError there, which escaped to the handler's
        # blanket except Exception — a full traceback and an error usage row
        # instead of the documented unsupported_format frame.
        if not isinstance(audio_fmt, str) or (
                audio_fmt not in RAW_FORMATS and audio_fmt not in ENCODED_FORMATS):
            await _safe_ws_send(ws, {"type": "error", "code": "unsupported_format",
                                    "message": f"audio format {repr(audio_fmt)[:64]} not supported "
                                               f"(raw: {sorted(RAW_FORMATS)}, "
                                               f"encoded via ffmpeg: {sorted(ENCODED_FORMATS)})"},
                                close=True)
            return
        # Human-readable transport label for the per-utterance log block.
        audio_source_label = (
            f"{audio_fmt} @ {SAMPLE_RATE} Hz mono (raw PCM, WebSocket)"
            if audio_fmt in RAW_FORMATS
            else f"{audio_fmt} → {SAMPLE_RATE} Hz mono (ffmpeg decode, WebSocket)")

        final_model = tx_models._resolve_model_name(model_req)
        # Unknown at job_start (it comes from the handshake): fill the running
        # row's model now so /stats reads it like the finished rows.
        jobs.job_update(session_id, model=final_model)
        partial_cfg = getattr(cfg, "STREAMING_PARTIAL_MODEL", "") or ""
        partial_model_name = partial_cfg or final_model
        async def _load_with_keepalive(name: str):
            # A cold large-v3 load takes 15 s+ with NOTHING on the wire, and
            # clients treat that silence as a dead server — the frontend's
            # stream drain used to discard a finished dictation seconds before
            # its transcript arrived. Signal liveness every few seconds; the
            # frame is additive, clients that don't know it ignore it.
            task = asyncio.ensure_future(tx_models._get_or_load_model(name, lease=True))
            try:
                while not task.done():
                    done, _ = await asyncio.wait({task}, timeout=3.0)
                    if done:
                        break
                    try:
                        await ws.send_json({"type": "loading", "model": name})
                    except Exception:  # noqa: BLE001
                        # Client gone mid-load: stop signalling, but still
                        # await the load so the model lands in the cache for
                        # the next connection (and its error, if any, is
                        # consumed here).
                        break
                model = await task
            except BaseException:
                # A cancellation (lifespan shutdown is the reachable one)
                # delivered at one of the awaits above, while the inner load
                # RETURNED: its lease was taken but the append below never
                # ran, so the teardown could not release it. Record it before
                # propagating; a load still in flight releases its own lease
                # once it lands, since the teardown will have run by then.
                if task.done():
                    if not task.cancelled() and task.exception() is None:
                        _model_leases_held.append(name)
                else:
                    task.add_done_callback(
                        lambda t: (not t.cancelled() and t.exception() is None
                                   and tx_models._release_model_lease(name)))
                raise
            # Only a load that RETURNED took a lease; record it for the
            # session teardown. A raising load leaves the refcount untouched.
            _model_leases_held.append(name)
            return model

        # The model the ledger row below names: the one whose load raised.
        _loading = final_model
        try:
            final_model_obj = await _load_with_keepalive(final_model)
            _loading = partial_model_name
            partial_model_obj = (
                final_model_obj if partial_model_name == final_model
                else await _load_with_keepalive(partial_model_name)
            )
        except Exception as exc:  # noqa: BLE001
            # The handshake `model` field lands verbatim in the not-in-allowed-
            # list HTTPException detail, so this text is client-controlled: a
            # bare CR/LF would forge extra records in the /logs viewer.
            logger.warning("[stream %s] model load failed: %s",
                           session_id[:8], store_common.log_safe(str(exc)))
            # This return skips the blanket handler's ledger row, so write the
            # same one here: a CUDA OOM / missing model on a live dictation
            # shows on /stats like the identical batch failure. Best effort.
            try:
                _ec, _es = metrics.classify_error(exc, status="error",
                                                  stage="transcribing")
                metrics.record_transcription(
                    model=str(_loading or ""), audio_dur=0.0, proc_dur=0.0,
                    status="error", words=0, kind="dictate",
                    request_id=session_id, user_id=user.get("user_id"),
                    key_id=user.get("key_id"), username=user.get("username"),
                    key_label=user.get("key_label"), job_id=usage_job_id,
                    error_class=_ec, error_stage=_es)
            except Exception:  # noqa: BLE001 — never mask the refusal
                pass
            # Generic client message — the raw exception text can carry model
            # dir/filesystem paths; the detail is already in the server log above.
            await _safe_ws_send(ws, {"type": "error", "code": "model_load_failed",
                                    "message": "model could not be loaded"}, close=True)
            return

        # Resolve the caller's effective per-identity config ONCE for this
        # connection. ident is built with final_model (per-model rule folding +
        # output wrappers + postprocess use final_model); identity scalar
        # overrides are model-independent, so they apply to the partial decode
        # too via cfg_for's ident layer.
        ident = effective_config.build_ident(user, final_model, request_profile=req_override_profile)
        gate_final_words = bool(effective_config.cfg_for(final_model, "WORD_TIMESTAMPS_ENABLED", ident))
        gate_partial_words = bool(effective_config.cfg_for(partial_model_name, "WORD_TIMESTAMPS_ENABLED", ident))

        # ident is resolved ONCE here, then re-resolved on the next decode
        # (partial or final) ONLY when the config version changes (see
        # _refresh_ident) — so admin edits to a binding/profile/setting apply
        # mid-session without a reconnect. Snapshot
        # the version and the client's ORIGINAL (pre-lock) handshake values so the
        # lock re-application stays idempotent across refreshes.
        _ident_version = settings_version.config_version()
        _client_language = req_language
        _client_prompt = req_prompt
        _client_prompt_provided = prompt_provided
        # Set by _refresh_ident when the handshake credential stops resolving to
        # the identity that opened the connection. Latching (never cleared): a
        # revoked session is closed, not resumed.
        _auth_revoked = False

        # Locked language / prompt: the admin value stands; the client's
        # handshake value is ignored (and surfaced in the ready frame). Locked
        # decode_overrides keys are dropped in the assembler; record them here.
        overrides_ignored = sorted(k for k in req_overrides
                                   if k in ident.locked_client_keys)
        _note_pinned_condition(req_overrides, overrides_ignored)
        if _language_dropped:
            overrides_ignored.append("language")
        if "DEFAULT_LANGUAGE" in ident.locked:
            _locked_lang = effective_config.cfg_for(final_model, "DEFAULT_LANGUAGE", ident) or ""
            if req_language and req_language != _locked_lang:
                overrides_ignored.append("language")
            req_language = _locked_lang
        tx_models._note_auto_detect_only(
            req_overrides, req_language if req_language is not None
            else effective_config.cfg_for(final_model, "DEFAULT_LANGUAGE", ident),
            overrides_ignored)
        tx_models._note_word_ts_only(req_overrides, gate_final_words, overrides_ignored)
        if "DEFAULT_PROMPT" in ident.locked:
            _locked_prompt = effective_config.cfg_for(final_model, "DEFAULT_PROMPT", ident) or ""
            if prompt_provided and req_prompt != _locked_prompt:
                overrides_ignored.append("prompt")
            req_prompt = _locked_prompt
            prompt_provided = True  # locked admin value is now authoritative

        async def _transcribe(model_obj, audio, kwargs, *, trace: bool = False,
                              skip_residual: bool = False,
                              token_cap_per_s: float = 0.0):
            """Returns (segments, info, trace_dict). `trace` (finals only)
            records what faster-whisper did inside the call — windows, rungs,
            tokens — for the receipt's Decode trace section. Partials skip
            it: they run many times per utterance and nothing reads it.
            `skip_residual` (DECODE_SKIP_RESIDUAL_WINDOWS, finals only) stops
            the decode after the window that reached the end of the audio;
            partials run without the temperature ladder, so a residual window
            costs them well under a second and needs no rule.
            `token_cap_per_s` (DECODE_TOKEN_CAP_PER_SECOND, finals only) bounds
            each rung's token count by the window length, so a looping decode
            cannot run to the model's hard limit."""
            loop = asyncio.get_running_loop()

            def work():
                if not trace:
                    segs, info = model_obj.transcribe(audio, **kwargs)
                    return list(segs), info, None
                # The lazy generator must be consumed INSIDE the capture: the
                # windows after the first are decoded there (thread-local).
                with decode_trace.capture(kwargs, skip_residual=skip_residual,
                                          token_cap_per_s=token_cap_per_s) as tr:
                    segs, info = model_obj.transcribe(audio, **kwargs)
                    segs = decode_trace.consume(segs)
                    return segs, info, decode_trace.finish(tr, segs, info)

            # Shared GPU limiter (same object the batch route uses).
            async with tx_models.get_inference_semaphore():
                return await loop.run_in_executor(None, work)

        # The language the document is FORMATTED in (language-tagged rules run
        # only for theirs). It has to be known before an utterance is formatted:
        # it used to be updated in on_final, i.e. AFTER the utterance's own
        # final had been formatted and sent — so an auto-language session's
        # first sentence ran every language's rules (a German question came out
        # with a Spanish '¿') and the next final formatted the same sentence
        # differently. decode_final now sets it from the decode it just ran;
        # until a final has, the partials' guess stands in.
        _detected_lang = [req_language or None]
        _partial_lang: list = [None]

        def _fmt_lang():
            """Detected (final) → guessed (partial) → requested → "und": an
            unknown language skips every language-tagged rule instead of
            running them all."""
            return _detected_lang[0] or _partial_lang[0] or req_language or "und"

        async def decode_partial(audio, prompt):
            _refresh_ident()
            if _auth_revoked:
                raise _CredentialRevoked("credential revoked mid-session")
            kwargs = _build_transcribe_kwargs(
                partial_model_name, final=False, prompt=prompt,
                want_words=gate_partial_words, language=req_language,
                model_obj=partial_model_obj, overrides=req_overrides, ident=ident)
            # A preview's GPU-gate wait is not the utterance's: only the
            # final's own acquire may land on the dictation row's wait_s.
            _wait_tok = metrics.WAIT_ACC.set(None)
            try:
                segs, _info, _ = await _transcribe(partial_model_obj, audio, kwargs)
            finally:
                metrics.WAIT_ACC.reset(_wait_tok)
            if not req_language:
                _partial_lang[0] = getattr(_info, "language", None) or _partial_lang[0]
            # Live previews get the same tail cuts as the final (core/
            # segment_guards.py): a made-up tail that shows up in two previews in
            # a row would otherwise be committed by LocalAgreement and banked as
            # confirmed text. Per segment, no cascade — a multi-window buffer can
            # hold real speech after a cut. A trimmed hypothesis is just a
            # shorter one, so commits still only ever extend. DEBUG only, not
            # counted: previews decode about once a second.
            # A segment cut to nothing is dropped, as the final and the batch
            # route do: handed on as (start, end, "") two previews would agree
            # on "" and LocalAgreement would commit an empty word at that time.
            # The prompt-echo head cut runs on the first segment, before its
            # tail cuts — the preview's prompt is the same rolling context the
            # final decodes with, so it echoes the same way.
            _limits = tx_guards.tail_guard_limits(partial_model_name, ident)
            _head_min = tx_guards.head_echo_min_words(partial_model_name, ident)
            _head_prompt = segment_guards.prompt_tail_text(kwargs)
            _kept_segs = []
            for _si, seg in enumerate(segs):
                if _si == 0 and _head_min and _head_prompt:
                    _hcut = segment_guards.apply_head_echo_guard(
                        seg, _head_prompt, _head_min)
                    if _hcut:
                        logger.debug("[stream %s] preview: cut prompt echo (%d words): %r",
                                     session_id[:8], _hcut["n"], _hcut["text"])
                        if not (getattr(seg, "text", "") or "").strip():
                            continue
                _cut = segment_guards.apply_tail_guards(seg, **_limits)
                if _cut:
                    logger.debug("[stream %s] preview: cut made-up tail (%s): %r",
                                 session_id[:8], "+".join(_cut["rules"]), _cut["text"])
                    if not (getattr(seg, "text", "") or "").strip():
                        continue
                _kept_segs.append(seg)
            segs = _kept_segs
            if gate_partial_words:
                words = [(w.start, w.end, w.word)
                         for seg in segs for w in (getattr(seg, "words", None) or [])]
                if words:
                    return words
            # fallback: segment-level units (coarser LocalAgreement granularity)
            return [(seg.start, seg.end, seg.text) for seg in segs]

        # Captures are eligible only when the model allows the DTW word path
        # (per-model WORD_TIMESTAMPS_ENABLED) — same gate as the batch route.
        # CAPTURES_RECORDING_ENABLED itself is read per utterance in on_final:
        # it is a privacy switch, and turning it off must stop recording on a
        # long-lived socket too, not only on the next connection.
        # The final decode stashes its faster-whisper info / segment diagnostics /
        # word list here so on_final (serialized right after, under the session
        # lock) can build the rich log block + the capture row without re-decoding.
        last_decode: dict = {}

        def _is_failed_segment(seg) -> bool:
            """Anti-hallucination: True when the final decode of this segment clearly
            FAILED — both very low confidence AND fell through the temperature ladder.
            faster-whisper only *retries* on low avg_logprob (never drops), and its
            silence skip needs no_speech_prob > NO_SPEECH_THRESHOLD, so a low-energy
            clip can emit a fabricated segment; this drops it. Requiring BOTH signals
            avoids discarding genuine quiet speech."""
            alp = getattr(seg, "avg_logprob", 0.0)
            temp = getattr(seg, "temperature", 0.0)
            # Resolve through the per-identity layer like every sibling STREAMING_*
            # decode knob here (cfg_for honours ident > per-model > global).
            floor = float(effective_config.cfg_for(final_model, "STREAMING_FINAL_DROP_MIN_AVG_LOGPROB", ident))
            ceil = float(effective_config.cfg_for(final_model, "STREAMING_FINAL_DROP_TEMPERATURE", ident))
            return alp < floor and temp >= ceil

        async def decode_final(audio, prompt):
            _refresh_ident()
            if _auth_revoked:
                # Raised BEFORE the model call, so on_final never runs: no
                # captures row, no quick_config trace, no usage row, no GPU work
                # attributed to an identity that no longer exists.
                raise _CredentialRevoked("credential revoked mid-session")
            # Each final starts from a clean accumulator: a final whose decode
            # raised after the gate charged its wait never reaches on_final
            # (the only take), and that wait would otherwise land on the NEXT
            # recorded utterance's wait_s.
            metrics.take_wait()
            tail_pad_ms = int(effective_config.cfg_for(final_model, "STREAMING_TAIL_TRIM_PAD_MS", ident))
            # Off the event loop: this is a full Silero sweep over the whole
            # final buffer (up to max_buffer_sec), and decode_final is awaited
            # from the pump task — run inline it stalls the producer's
            # ws.receive() drain (the wedge documented at the queue-sizing
            # note). The pump is the sole session mutator, so the extra
            # suspension adds no new interleaving.
            _untrimmed_s = audio.shape[0] / SAMPLE_RATE
            audio = await asyncio.to_thread(
                _trim_trailing_nonspeech,
                audio,
                tail_pad_ms,
                float(effective_config.cfg_for(final_model, "VAD_THRESHOLD", ident)),
                session_id[:8])
            _trimmed_s = audio.shape[0] / SAMPLE_RATE
            kwargs = _build_transcribe_kwargs(
                final_model, final=True, prompt=prompt,
                want_words=gate_final_words, language=req_language,
                model_obj=final_model_obj, overrides=req_overrides, ident=ident)
            skip_residual = bool(effective_config.cfg_for(
                final_model, "DECODE_SKIP_RESIDUAL_WINDOWS", ident))
            token_cap = float(effective_config.cfg_for(
                final_model, "DECODE_TOKEN_CAP_PER_SECOND", ident) or 0.0)
            segs, info, trace = await _transcribe(
                final_model_obj, audio, kwargs, trace=True,
                skip_residual=skip_residual, token_cap_per_s=token_cap)
            max_wps = float(effective_config.cfg_for(final_model, "SEGMENT_MAX_WORDS_PER_S", ident) or 0)
            # Tail cuts inside a segment — AFTER the whole-segment verdicts, on
            # the survivors (a segment made up from start to end is still
            # dropped whole). See transcription/segment_guards.py.
            tail_limits = tx_guards.tail_guard_limits(final_model, ident)
            tail_cuts: list[dict] = []
            # Head cut (SEGMENT_HEAD_ECHO_MIN_WORDS): the decode repeating the
            # last words of its prompt — the previous utterances — before the
            # new speech, which would type the end of the last sentence twice.
            # First surviving segment only, before its tail cuts.
            head_min = tx_guards.head_echo_min_words(final_model, ident)
            head_prompt = segment_guards.prompt_tail_text(kwargs)
            head_pending = bool(head_min and head_prompt)
            head_cut: "dict | None" = None
            words_out: list[dict] = []
            seg_diag: list[dict] = []
            kept: list[str] = []
            for i, seg in enumerate(segs):
                dropped_conf = _is_failed_segment(seg)
                dropped_rate = tx_guards.segment_exceeds_word_rate(seg, max_wps)
                dropped = dropped_conf or dropped_rate
                cut = None
                hcut = None
                if not dropped and head_pending:
                    head_pending = False
                    hcut = segment_guards.apply_head_echo_guard(seg, head_prompt, head_min)
                    if hcut is None:
                        _hw = segment_guards.head_words_diag(seg, head_prompt, head_min)
                        if _hw:
                            logger.info("[stream %s] head_words (prompt repeated, looks "
                                        "spoken, nothing cut): %s", session_id[:8], _hw)
                    head_cut = hcut
                if not dropped and (hcut is None or (seg.text or "").strip()):
                    # IN PLACE: raw, the capture's words and the rolling prompt
                    # all inherit the cut, so a made-up tail can no longer feed
                    # the next utterance's prompt.
                    cut = segment_guards.apply_tail_guards(seg, **tail_limits)
                emptied = ((cut is not None or hcut is not None)
                           and not (seg.text or "").strip())
                dropped = dropped or emptied
                seg_diag.append({
                    "id": i, "start": seg.start, "end": seg.end,
                    "alp": getattr(seg, "avg_logprob", 0.0),
                    "nsp": getattr(seg, "no_speech_prob", 0.0),
                    "cr": getattr(seg, "compression_ratio", 1.0),
                    "temp": getattr(seg, "temperature", 0.0),
                    # An emptied row shows what was removed, not "".
                    "text": ((hcut or {}).get("text", "") + (cut or {}).get("text", "")
                             if emptied else seg.text),
                    "dropped": dropped,
                    **({"cut": cut} if cut else {}),
                    **({"head_cut": hcut} if hcut else {}),
                })
                if hcut:
                    # "emptied" once per segment: here only when the head cut
                    # alone left nothing (the tail cuts then never ran).
                    tx_guards.record_tail_cut(hcut, emptied=emptied and cut is None)
                    logger.info("[stream %s] cut prompt echo from the start of the final "
                                "(%d words %.2f-%.2fs): %r", session_id[:8],
                                hcut["n"], hcut["from"], hcut["to"], hcut["text"])
                    if emptied and cut is None:
                        continue
                if cut:
                    tail_cuts.append(cut)
                    tx_guards.record_tail_cut(cut, emptied=emptied)
                    logger.info("[stream %s] cut made-up tail of final segment "
                                "(%s, %d words%s): %r", session_id[:8],
                                "+".join(cut["rules"]), cut["n"],
                                "" if cut.get("from") is None
                                else f" from {cut['from']:.2f}s", cut["text"])
                    if emptied:
                        continue
                elif not dropped:
                    _tw = segment_guards.tail_words_diag(seg)
                    if _tw:
                        logger.info("[stream %s] tail_words (zero-length word, "
                                    "nothing cut): %s", session_id[:8], _tw)
                if dropped_conf:
                    metrics.record_guard_hit("low_conf")
                    logger.info("[stream %s] dropped low-confidence final segment "
                                "(alp=%.2f temp=%.2f): %r", session_id[:8],
                                getattr(seg, "avg_logprob", 0.0),
                                getattr(seg, "temperature", 0.0), seg.text)
                    continue
                if dropped_rate:
                    metrics.record_guard_hit("word_rate")
                    _dur = float(seg.end) - float(seg.start)
                    _n = (len(getattr(seg, "words", None) or [])
                          or len((seg.text or "").split()))
                    logger.info("[stream %s] dropped word-rate-anomalous final "
                                "segment (%.2f-%.2fs, %.1f w/s > %.1f): %r",
                                session_id[:8], seg.start, seg.end,
                                _n / max(_dur, 1e-6), max_wps, seg.text)
                    continue
                kept.append(seg.text)
                for w in (getattr(seg, "words", None) or []):
                    words_out.append({"word": w.word, "start": w.start, "end": w.end})
            raw = "".join(kept)
            # Tell the session whether this (possibly empty) result is authoritative:
            # when the decode produced segments but dropped them ALL as hallucinations,
            # the empty text must NOT be replaced by the partial-built transcript
            # (partials run at fixed temperature and so never trip _is_failed_segment —
            # they would still carry the hallucination).
            dropped_all = bool(segs) and not kept
            # The formatting language, from THIS decode, before the session
            # formats its text (see _fmt_lang). Only when something was kept:
            # a decode of dropped noise says nothing about the speaker.
            if kept:
                _detected_lang[0] = (getattr(info, "language", None) or req_language
                                     or _detected_lang[0])
            last_decode.clear()
            last_decode.update(info=info, seg_diag=seg_diag, kwargs=kwargs, trace=trace, guards={
                # Post-decode guard settings as applied to THIS decode — rendered
                # in the log block's guards section (they are not transcribe
                # kwargs, so the Decode params section can't show them).
                "segment_max_words_per_sec": max_wps,
                **tx_guards.tail_guard_rows(tail_limits),
                **tx_guards.tail_cut_rows(tail_cuts),
                **tx_guards.head_echo_rows(head_min, head_cut),
                "skip_residual_windows": decode_trace.residual_stop_active(
                    kwargs, skip_residual),
                "token_cap_per_second": token_cap,
                "tail_trim_pad_ms": tail_pad_ms,
                # What the trim actually removed from THIS buffer (the block's
                # `duration` row is the post-trim length; the utterance label
                # is the pre-trim one — this row is the bridge between them).
                "tail_trim_cut": tx_receipt.PlainText(
                    f"{_untrimmed_s - _trimmed_s:.2f}s  "
                    f"({_untrimmed_s:.2f}s → {_trimmed_s:.2f}s)"),
                "final_drop_min_avg_logprob": float(effective_config.cfg_for(
                    final_model, "STREAMING_FINAL_DROP_MIN_AVG_LOGPROB", ident)),
                "final_drop_temperature": float(effective_config.cfg_for(
                    final_model, "STREAMING_FINAL_DROP_TEMPERATURE", ident)),
            })
            return raw, words_out, dropped_all

        def postprocess(raw_text):
            return pl_engine._postprocess_text(raw_text, model_name=final_model, ident=ident, language=_fmt_lang())

        # Output wrappers: the prefix sits at the very start of the document, the
        # suffix at its end. committed/tail are full authoritative strings (the
        # client replaces each region), so re-applying the prefix on every final
        # is correct — it never accumulates. A document ends on the closing
        # flush (``last``) or at a hard break: a ``boundary`` starts a fresh
        # document that gets the prefix again, so the one before it is closed by
        # putting the suffix in front of the boundary's separator (clients type
        # the separator verbatim) — else every document but the last opened a
        # wrapper that never closed. The client's output_prefix / output_suffix
        # win unless locked.
        out_prefix, out_suffix = tx_models._output_wrappers(final_model, ident, req_overrides)
        # Whether a final of the current document has gone out (with the
        # prefix) and its suffix is still owed.
        _doc_open = [False]

        async def emit(message):
            kind = message.get("type")
            if kind == "final":
                if not include_words:
                    message.pop("words", None)   # word timestamps only for verbose_json
                if message.get("committed") or message.get("tail"):
                    _doc_open[0] = True
                if out_prefix:
                    if message.get("committed"):
                        message["committed"] = out_prefix + message["committed"]
                    elif message.get("tail"):
                        message["tail"] = out_prefix + message["tail"]
                if out_suffix and message.get("last"):
                    message["committed"] = (message.get("committed") or "") + out_suffix
                    _doc_open[0] = False
            elif kind == "boundary":
                if out_suffix and _doc_open[0]:
                    message["separator"] = out_suffix + (message.get("separator") or "")
                _doc_open[0] = False
            # Peer may have vanished mid-drain (page reload during dictation): the
            # socket is already closed, so swallow the send. Side-effects
            # (metrics/trace/captures) still ran in on_final.
            async with send_lock:
                await _safe_ws_send(ws, message)

        def _maybe_capture(rid, info, raw_text, final_text, words, fw_info,
                           segments=()):
            """Persist a fine-tuning capture for this utterance, mirroring the batch
            route's eligibility gate (sampling / size / duration / disk). No
            CAPTURES_MAX gate: a full store is create_capture's _evict_to_cap's
            job, which rotates the oldest rows out in its documented priority
            order — refusing here at the cap meant that eviction never ran and
            every later utterance silently recorded nothing. The one cap term
            is a store whose "ready" rows alone fill it: the new row would be
            evicted by its own insert, so skip before the WAV write."""
            try:
                from faster_whisper_backend.captures import store as captures_store
                audio = info["audio"]
                pcm_bytes = int(getattr(audio, "size", 0)) * 2
                hard_lim = int(getattr(cfg, "CAPTURES_RECORDING_AUDIO_BYTES_HARD_LIMIT", 100_000_000))
                sample = float(getattr(cfg, "CAPTURES_RECORDING_SAMPLE_RATE", 1.0))
                if not (pcm_bytes < hard_lim and random.random() < sample):
                    return None
                dur = float(info["audio_dur"])
                min_s = float(getattr(cfg, "CAPTURES_RECORDING_MIN_DURATION_S", 0.5))
                max_s = float(getattr(cfg, "CAPTURES_RECORDING_MAX_DURATION_S", 600.0))
                if not (min_s <= dur <= max_s):
                    logger.info("[stream %s] capture skipped duration %.1fs (window %.1f-%.1f)",
                                session_id[:8], dur, min_s, max_s)
                    return None
                try:
                    free = shutil.disk_usage(cfg.CAPTURES_DIR).free
                except OSError:
                    free = 1 << 40
                if free <= 1_000_000_000:
                    logger.warning("[stream %s] capture skipped: low disk (%.0f MB free)",
                                   session_id[:8], free / (1024 * 1024))
                    return None
                if captures_store.ready_fills_cap():
                    return None
                training_text = pl_engine._postprocess_text(
                    raw_text, model_name=final_model, trace=None,
                    extra_excludes=getattr(cfg, "CAPTURES_PIPELINE_RULES_EXCLUDE", None),
                    ident=ident, language=_fmt_lang())
                wav_path = _write_pcm16_wav(audio)
                try:
                    return captures_store.create_capture(
                        audio_src_path=wav_path, request_id=rid, model=final_model,
                        language=(getattr(fw_info, "language", None) or req_language or ""),
                        audio_s=dur, raw=raw_text, final=final_text,
                        text_for_training=training_text, words=words,
                        segments=list(segments),
                        user_id=user.get("user_id"))
                finally:
                    try:
                        os.unlink(wav_path)
                    except OSError:
                        pass
            except Exception as _ce:  # noqa: BLE001 — never let a capture failure break dictation
                logger.warning("[stream %s] capture failed: %s", session_id[:8], _ce)
                return None

        async def on_final(info):
            # One finalized utterance == one mini-transcription: replicate the batch
            # route's per-request side-effects (rich log block, durable trace for
            # /quick-config + /reports, capture, metrics) so streaming has parity.
            #
            # Same guard as decode_final: on_final is reachable WITHOUT a decode
            # (the session's near-silence gate skips the decoder but still
            # reports the committed text), so the pre-model raise there no
            # longer covers every side-effect sink. A revoked identity gets no
            # captures row, no trace, no usage row from this path either.
            _refresh_ident()
            if _auth_revoked:
                # Raise (not return) so the gate path — where no decode ran and
                # this is the only latch site — still reaches the pump's
                # _CredentialRevoked branch that closes the socket.
                raise _CredentialRevoked("credential revoked mid-session")
            rid = uuid.uuid4().hex
            raw_text = info["raw_text"] or ""
            words = info.get("words") or []
            decoded = bool(info.get("decoded", True))
            # SNAPSHOT, not an alias: last_decode lives for the whole session
            # and decode_final clears/refills it for every utterance. on_final
            # awaits below (the capture thread, the `captured` frame), and the
            # log block is assembled after those awaits — reading through the
            # live dict there would be one refactor away from printing the NEXT
            # utterance's guards under this one's text. The values are all
            # rebuilt per decode, so a shallow copy is enough. When NO decode
            # ran for this utterance the dict still holds the PREVIOUS
            # utterance's info/diagnostics/guards — those must not be
            # attributed to this text, so the snapshot is empty then (fw_info
            # None: the block, the capture and the trace all tolerate it).
            dec = dict(last_decode) if decoded else {}
            fw_info = dec.get("info")
            seg_diag = dec.get("seg_diag", [])
            kwargs = dec.get("kwargs", {})

            steps: "list | None" = [] if getattr(cfg, "TRACE_ENABLED", False) else None
            final_text = pl_engine._postprocess_text(raw_text, model_name=final_model, trace=steps, ident=ident, language=_fmt_lang())
            if info.get("decode_failed"):
                # The session already logged the failure (type only). Say here
                # what the text IS, so the row below isn't read as a decode.
                logger.info(
                    "[stream %s] utt#%s: the final decode failed — the text is "
                    "the partial-committed transcript",
                    session_id[:8], info["utterance"])
            elif not decoded:
                _trimmed = info.get("trimmed_sec") or 0.0
                logger.info(
                    "[stream %s] utt#%s: near-silence gate skipped the final "
                    "decode — the text is the partial-committed transcript%s",
                    session_id[:8], info["utterance"],
                    (f" ({_trimmed:.2f}s banked by a mid-utterance trim)"
                     if _trimmed else ""))
            elif info.get("trimmed_sec"):
                # The session banked the trimmed audio + committed words, so
                # raw_text / words / info["audio"] all span the WHOLE utterance;
                # only the final DECODE ran on the shortened buffer.
                logger.info(
                    "[stream %s] utt#%s: buffer was trimmed mid-utterance — final "
                    "decode heard the last %.2fs; the first %.2fs carry the "
                    "partial-committed text",
                    session_id[:8], info["utterance"],
                    info["audio_dur"] - info["trimmed_sec"], info["trimmed_sec"])

            captured_id = None
            if (gate_final_words and getattr(cfg, "CAPTURES_RECORDING_ENABLED", False)
                    and raw_text.strip()):
                # seg_diag is on the decode BUFFER's timeline; after a trim the
                # capture's audio and words span the whole utterance (the
                # session re-bases the decode's words by the same offset, which
                # equals trimmed_sec), so shift the stored segments to match.
                # seg_diag itself stays buffer-relative for the log and trace.
                cap_segs = seg_diag
                _off = info.get("trimmed_sec") or 0.0
                if decoded and _off:
                    cap_segs = [{**sd, "start": sd["start"] + _off, "end": sd["end"] + _off}
                                for sd in seg_diag]
                # OFF the loop: _maybe_capture writes a WAV and then runs
                # captures_store.create_capture, which re-transcodes it through
                # PyAV — the same blocking work the batch route offloads. This
                # fires per finalized utterance on every live session.
                captured_id = await asyncio.to_thread(
                    _maybe_capture, rid, info, raw_text, final_text, words,
                    fw_info, cap_segs)

            # One `transcribing` stage per live utterance — but none at all when
            # the gate skipped the decode: a 0.00 s stage that never ran would
            # skew the /stats dictation timing aggregates.
            stages = ([{"name": "transcribing",
                        "secs": round(float(info["proc_dur"] or 0.0), 2),
                        "model": final_model}]
                      if decoded else None)

            # Rich diagnostic block — same formatter the batch route uses, so the
            # VAD-ate-audio / empty-output / pipeline-step diagnostics show up for
            # streaming too. file_label marks it as a streamed utterance.
            try:
                _block_kwargs = dict(
                    file_label=f"stream {session_id[:8]} utt#{info['utterance']}  "
                               f"({info['audio_dur']:.2f}s, "
                               f"{store_common.log_safe(str(response_format))})"
                               # The client's session id, when it sent one: a
                               # stop-timing translate names the same id on its
                               # standalone receipt, so the two grep together.
                               + (f"  job={usage_job_id[:8]}"
                                  if usage_job_id != session_id else ""),
                    model_name=final_model, info=fw_info, kwargs=kwargs,
                    seg_diag=seg_diag, raw=raw_text, final=final_text,
                    steps=steps, request_id=rid, captured_id=captured_id,
                    endpoint="/v1/audio/transcriptions/stream",
                    audio_source=audio_source_label,
                    ident=ident, overrides_ignored=overrides_ignored,
                    user_id=user.get("user_id"), key_id=user.get("key_id"),
                    username=user.get("username"), key_label=user.get("key_label"),
                    guards=dec.get("guards"),
                    decode_trace=dec.get("trace"),
                    stages=stages)
                # Hold only when the client said a per-utterance translation
                # is coming AND there is a capture id to key it on — that id is
                # the only handle the separate translate request can name us
                # by. A stop-timing session declares per_utterance=False and
                # logs each utterance immediately, because its one translate
                # call names no capture and would leave every held receipt to
                # the idle sweep.
                if translate_expect and translate_expect["per_utterance"] and captured_id:
                    _block_kwargs["translation"] = {
                        "targets": list(translate_expect["targets"]),
                        "include_original": translate_expect["include_original"],
                        "model": None, "mode": None,
                    }
                    receipt_hold.park(
                        captured_id, _block_kwargs,
                        hold_s=float(getattr(cfg, "LOG_RECEIPT_HOLD_S", 90)))
                    # Tell the client which capture to claim. The `final`
                    # frame has already gone out by the time on_final runs
                    # (streaming_session emits it first), so this rides its
                    # own follow-up frame.
                    try:
                        await emit({"type": "captured", "id": captured_id,
                                    "utterance": info["utterance"]})
                    except Exception:  # noqa: BLE001 — best effort
                        pass
                else:
                    logger.info(tx_receipt._format_request_block(**_block_kwargs))
            except Exception as _le:  # noqa: BLE001
                logger.warning("[stream %s] log block failed: %s", session_id[:8], _le)

            # Durable trace → /quick-config recent-transcriptions + autocomplete + SSE.
            # source='stream' tags the row so /quick-config can chip it as live
            # dictation vs a file-upload (batch) transcription.
            try:
                from faster_whisper_backend.quick_config import recent_feed as qc_recent_feed
                qc_recent_feed.record_trace(
                    request_id=rid, model=final_model, raw=raw_text,
                    steps=steps if steps is not None else [], final=final_text,
                    language=(getattr(fw_info, "language", None) or req_language or None),
                    source="stream", user_id=user.get("user_id"))
            except Exception as _qe:  # noqa: BLE001
                logger.error("[stream %s] record_trace failed: %s", session_id[:8], _qe)

            # Timing/usage half — UPSERTs onto the same request_id row as record_trace.
            # kind + stages are both supported here and were both omitted,
            # which is why every dictation row on /stats read "no per-stage
            # timings recorded" and fell back to the synthetic pipeline glyph.
            # A live utterance has exactly one stage, but saying so beats
            # saying nothing and makes dictation rows comparable with batch.
            metrics.record_transcription(
                model=final_model, audio_dur=info["audio_dur"],
                proc_dur=info["proc_dur"], status="ok",
                words=len(final_text.split()), kind="dictate",
                stages=([{**stages[0], "detail": f"utt#{info['utterance']}"}]
                        if stages else None),
                request_id=rid, user_id=user.get("user_id"), key_id=user.get("key_id"),
                username=user.get("username"), key_label=user.get("key_label"),
                job_id=usage_job_id,
                language=(getattr(fw_info, "language", None) or req_language or None),
                wait_s=metrics.take_wait())

        # The client's own live-dictation knobs (decode_overrides), fixed for
        # the connection like the rest of the session shape.
        client_knobs, knob_notes = _client_stream_values(req_overrides, ident)
        if client_knobs:
            logger.info("[stream %s] client knobs %s%s", session_id[:8],
                        ", ".join(f"{k.removeprefix('STREAMING_').lower()}={v!r}"
                                  for k, v in sorted(client_knobs.items())),
                        f" (adjusted: {'; '.join(knob_notes)})" if knob_notes else "")
        session = StreamSession(
            config=_stream_config(effective_config.cfg_for, ident, client=client_knobs),
            endpointer=make_endpointer(
                effective_config.cfg_for(final_model, "STREAMING_VAD_BACKEND", ident),
                threshold=float(client_knobs.get(
                    "STREAMING_VAD_THRESHOLD",
                    effective_config.cfg_for(final_model, "STREAMING_VAD_THRESHOLD", ident))),
                energy_dbfs=float(effective_config.cfg_for(final_model, "STREAMING_GATE_RMS_DBFS", ident)),
            ),
            decode_partial=decode_partial,
            decode_final=decode_final,
            postprocess=postprocess,
            emit=emit,
            base_prompt=(req_prompt if prompt_provided
                         else (effective_config.cfg_for(final_model, "DEFAULT_PROMPT", ident) or "")),
            on_final=on_final,
            session_id=session_id,
            # Seam hooks (streaming/session.py). Lambdas, not bound values: they
            # read `ident` (re-resolved by _refresh_ident) and the language at
            # call time.
            holdback=lambda raw: pl_engine.holdback_start(
                raw, model_name=final_model, ident=ident, language=_fmt_lang()),
            format_key=_fmt_lang,
            diagnose=lambda sent_raw, raw: pl_engine.seam_culprit(
                sent_raw, raw, model_name=final_model, ident=ident, language=_fmt_lang()),
        )

        def _refresh_ident():
            """Re-resolve this connection's per-identity config when the global
            config version changed since we last resolved — so an admin editing a
            binding / profile / setting takes effect on the next decode instead
            of requiring the client to reconnect. Called by every decode,
            partials included (about once a second): a revocation stops the
            partials too, and one utterance's partials may straddle a change.
            The no-change case is an integer compare plus, at most every 0.25 s
            per process, a sibling data_version PRAGMA (see
            settings_version._KEYS_PROBE_MIN_INTERVAL_S); a real change costs a
            couple of indexed SQLite reads. Session-shaping
            STREAMING_*/endpointer params (the client's own knobs included) and
            the word-timestamp gates stay fixed for the connection: a lock added
            mid-session applies to them from the next connection. Never raises —
            a refresh must not break dictation.

            The same bump ALSO revalidates the credential. A WebSocket has no
            request boundary, so `authenticate_ws` used to run exactly once, at
            the handshake, and the connection then carried that captured `user`
            dict for its whole life: revoking the key (or signing out) left the
            revoked identity decoding under the shared GPU semaphore and writing
            captures / trace / usage rows indefinitely, since an actively
            speaking client re-arms the audio-anchored idle deadline forever.
            api_keys_store.revoke_user / revoke_key already bump the config
            version for exactly this ("revoked identity's live idents
            re-resolve"), as does sessions_store.revoke_session (/auth/logout),
            and this is that bump's only consumer on this path.
            `_ws_credentials` reads only ws.headers/ws.cookies, both of which
            outlive the handshake, so re-invoking it here is enough."""
            nonlocal ident, _ident_version, out_prefix, out_suffix
            nonlocal req_language, req_prompt, overrides_ignored
            nonlocal user, _auth_revoked
            try:
                v = settings_version.config_version()
                if v == _ident_version:
                    return
                _ident_version = v
                # authenticate_ws is sync and non-raising BY DESIGN (auth._resolve_user
                # returns None on an unresolvable credential) — but it reaches SQLite,
                # so treat an unexpected raise as "cannot prove still-valid" rather
                # than letting the blanket handler below log it and carry on decoding.
                try:
                    fresh = authenticate_ws(ws)
                except Exception as _ae:  # noqa: BLE001
                    logger.warning("[stream %s] re-auth failed: %s",
                                   session_id[:8], store_common.log_safe(str(_ae)))
                    fresh = None
                if fresh is None or fresh.get("user_id") != user.get("user_id"):
                    # Revoked, rotated, signed out, or now resolving to somebody
                    # else. Latch and stop: the decode guards abort every sink on
                    # the spot and the producer closes the socket with 4401. The
                    # in-flight utterance is deliberately lost — revocation takes
                    # effect immediately, it does not wait for the sentence to end.
                    _auth_revoked = True
                    logger.warning("[stream %s] credential no longer valid — closing",
                                   session_id[:8])
                    return
                # Adopt the fresh record so a permissions DOWNGRADE (not just a
                # revocation) takes effect too, instead of the connection running
                # on the permissions it captured at the handshake.
                user = fresh
                ident = effective_config.build_ident(user, final_model, request_profile=req_override_profile)
                out_prefix, out_suffix = tx_models._output_wrappers(
                    final_model, ident, req_overrides)
                overrides_ignored = sorted(k for k in req_overrides
                                           if k in ident.locked_client_keys)
                _note_pinned_condition(req_overrides, overrides_ignored)
                if _language_dropped:
                    overrides_ignored.append("language")
                req_language = _client_language
                if "DEFAULT_LANGUAGE" in ident.locked:
                    _ll = effective_config.cfg_for(final_model, "DEFAULT_LANGUAGE", ident) or ""
                    if _client_language and _client_language != _ll:
                        overrides_ignored.append("language")
                    req_language = _ll
                tx_models._note_auto_detect_only(
                    req_overrides, req_language if req_language is not None
                    else effective_config.cfg_for(final_model, "DEFAULT_LANGUAGE", ident),
                    overrides_ignored)
                tx_models._note_word_ts_only(req_overrides, gate_final_words,
                                        overrides_ignored)
                req_prompt = _client_prompt
                _provided = _client_prompt_provided
                if "DEFAULT_PROMPT" in ident.locked:
                    _lp = effective_config.cfg_for(final_model, "DEFAULT_PROMPT", ident) or ""
                    if _client_prompt_provided and _client_prompt != _lp:
                        overrides_ignored.append("prompt")
                    req_prompt = _lp
                    _provided = True
                # The rolling prompt seed lives on the session; updating it makes a
                # changed DEFAULT_PROMPT take effect from the next utterance's
                # _make_prompt() (one-utterance convergence). An explicitly cleared
                # client prompt (_provided + "") keeps the seed empty (no fallback).
                session.base_prompt = (req_prompt if _provided
                                       else (effective_config.cfg_for(final_model, "DEFAULT_PROMPT", ident) or ""))
            except Exception as _re:  # noqa: BLE001 — never break dictation on refresh
                logger.warning("[stream %s] ident refresh failed: %s", session_id[:8], _re)

        # Producer→consumer hand-off. The receive loop (producer) must NEVER block
        # on a decode: if it does, it stops draining ws.receive(), the inbound WS
        # message queue fills, websockets stops reading frames, the client's PONGs
        # go unread, and the keepalive ping times out → the socket dies mid-
        # utterance. So sink() only ENQUEUES; a dedicated consumer task (_pump,
        # below) runs the CPU-bound feed_pcm/decode out of the receive loop's way.
        # Items: ("pcm", bytes) | ("flush", None) | ("stop", None).
        audio_q: "asyncio.Queue[tuple[str, bytes | None]]" = asyncio.Queue()
        _bytes_per_sec = SAMPLE_RATE * 2          # 16 kHz mono s16le
        # Skip the partial decode when the backlog exceeds ~one partial interval of
        # audio (we're behind realtime); audio is still fed so VAD/endpointing stays
        # intact and finals still run, letting us catch up without dropping audio.
        _behind_bytes = int(_bytes_per_sec * max(0.25, session.cfg.min_chunk_ms / 1000.0))
        _hard_cap_bytes = _bytes_per_sec * 60     # absolute backlog cap (drop oldest, logged)
        _qbytes = 0                                # PCM bytes currently queued
        # Control items carry no PCM, so the cap below cannot bound them: while the
        # pump is blocked in a decode, a client spamming {"type":"flush"} would grow
        # the queue without bound. Coalesce instead — two flushes with no audio
        # between them mean the same as one, so at most one is ever queued.
        _flush_pending = False                     # a flush is queued or in flight
        # Backlog-shedding is client-paced, so its WARNING is rate-limited and
        # aggregated (see below). Counters carry the suppressed totals.
        _shed_logged_at = 0.0
        _pending_shed_items = 0
        _pending_shed_bytes = 0
        _oversize_logged_at = 0.0

        async def sink(pcm: bytes):
            nonlocal _qbytes, _flush_pending
            nonlocal _shed_logged_at, _pending_shed_items, _pending_shed_bytes
            nonlocal _oversize_logged_at
            if not pcm:
                return
            # A single frame larger than the whole backlog cap would sail past
            # the loop below (which can only drop what is ALREADY queued) and
            # land in one put. Real clients send ~32 KB frames; the websocket
            # library's default message ceiling is 16 MiB, so one crafted binary
            # frame would otherwise expand to ~8 minutes of audio, float32'd and
            # swept through VAD frame-by-frame in a single uninterrupted pass.
            # Feed it in cap-sized pieces so the backlog logic still governs.
            if len(pcm) > _hard_cap_bytes:
                # Throttled for the same reason as the shed summary below: the
                # client chooses how often this fires.
                _now_o = time.monotonic()
                if _now_o - _oversize_logged_at >= _SHED_LOG_INTERVAL_S:
                    _oversize_logged_at = _now_o
                    logger.warning(
                        "[stream %s] oversized audio frame (%d bytes) — feeding "
                        "in %d-byte pieces (at most one line every %.0f s)",
                        session_id[:8], len(pcm), _hard_cap_bytes,
                        _SHED_LOG_INTERVAL_S,
                    )
                for off in range(0, len(pcm), _hard_cap_bytes):
                    await sink(pcm[off:off + _hard_cap_bytes])
                return
            # Absolute backlog cap: if the consumer has fallen catastrophically
            # behind, drop the oldest queued PCM (never silently) so memory can't
            # grow without bound. Skip-if-behind normally prevents reaching this.
            _shed_items = 0
            _shed_bytes = 0
            while (_qbytes + len(pcm) > _hard_cap_bytes
                   or audio_q.qsize() >= _HARD_CAP_ITEMS):
                try:
                    kind, old = audio_q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if kind == "stop":
                    # _pump's ONLY exit is this sentinel, and teardown awaits the
                    # pump task with no timeout — discarding it would wedge the
                    # consumer forever, so the outer finally never runs and the
                    # session slot leaks for the process lifetime. Put it back
                    # and stop draining, exactly as the flush case re-arms.
                    audio_q.put_nowait((kind, old))
                    break
                dropped = len(old) if (kind == "pcm" and old) else 0
                if kind == "flush":
                    # Dropping the queued flush re-arms coalescing, so the client
                    # can ask for another one instead of being stuck.
                    _flush_pending = False
                _qbytes -= dropped
                _shed_items += 1
                _shed_bytes += dropped
            # One throttled summary rather than a line per popped item. A client
            # uploading faster than the pump consumes keeps this branch hot, and
            # an unthrottled line per item let it roll the whole retained log
            # history (LOG_MAX_BYTES x LOG_BACKUP_COUNT) off disk — taking the
            # origin-rejection and auth records with it. Same reasoning as
            # auth_hosts._ORIGIN_REJECT_LOG_INTERVAL_S.
            if _shed_items:
                _now_m = time.monotonic()
                _pending_shed_items += _shed_items
                _pending_shed_bytes += _shed_bytes
                if _now_m - _shed_logged_at >= _SHED_LOG_INTERVAL_S:
                    _shed_logged_at = _now_m
                    logger.warning(
                        "[stream %s] audio backlog over cap — shed %d queued "
                        "items / %d bytes (at most one line every %.0f s)",
                        session_id[:8], _pending_shed_items, _pending_shed_bytes,
                        _SHED_LOG_INTERVAL_S,
                    )
                    _pending_shed_items = 0
                    _pending_shed_bytes = 0
            audio_q.put_nowait(("pcm", pcm))
            _qbytes += len(pcm)
            # New audio behind a queued flush means that flush is no longer
            # authoritative for the whole stream ("two flushes with no audio
            # between them mean the same as one" — see the control handler):
            # re-arm so a flush AFTER this audio queues its own item instead of
            # being silently coalesced into the earlier one. Flush spam with no
            # audio in between still collapses to a single queued item.
            _flush_pending = False

        async def _pump() -> None:
            """Consume queued audio/control out of the receive loop's way. Sole
            session mutator while the connection is live, so session access here
            needs no lock; session.close() runs only after this task has finished
            or been cancelled (see teardown)."""
            nonlocal _qbytes, _flush_pending
            while True:
                kind, data = await audio_q.get()
                if kind == "stop":
                    break
                try:
                    if kind == "pcm":
                        _qbytes -= len(data)
                        session._skip_partials = _qbytes > _behind_bytes
                        await session.feed_pcm(data)
                    elif kind == "flush":
                        try:
                            await session.flush_utterance()
                        finally:
                            # Re-arm on every exit (done, failed, cancelled) so a
                            # client is never left unable to flush again.
                            _flush_pending = False
                except _CredentialRevoked:
                    # NOT a decode error: stop consuming entirely, and close the
                    # socket HERE — the producer is parked in ws.receive() and
                    # would otherwise only notice the latched flag at the next
                    # inbound frame (a silent-but-connected client keeps the
                    # revoked session open until the idle timeout). Closing
                    # makes that receive() return a disconnect immediately; the
                    # producer's own _auth_revoked branch stays as a backstop
                    # (its send/close already tolerate a closed socket).
                    async with send_lock:
                        try:
                            await ws.send_json({"type": "error", "code": "unauthorized",
                                                "message": "credential no longer valid; closing"})
                            await ws.close(code=_WS_UNAUTH)
                        except Exception:  # noqa: BLE001 — peer may be gone
                            pass
                    break
                except Exception as exc:  # noqa: BLE001 — a decode error must not kill the pump
                    # An invalid handshake `language` makes the tokenizer raise
                    # a ValueError naming the client's own string — screen it
                    # before it reaches the line-oriented log.
                    logger.warning("[stream %s] pump error: %s",
                                   session_id[:8], store_common.log_safe(str(exc)))

        transport = make_transport(audio_fmt, sink, sample_rate=SAMPLE_RATE)
        await transport.start()

        ready_msg = {
            "type": "ready", "session": session_id, "model": final_model,
            "partial_model": partial_model_name, "sample_rate": SAMPLE_RATE,
            "response_format": response_format, "audio_format": audio_fmt,
            # This server announces each utterance's lifecycle (open / decoding /
            # dropped — see the session module). Informational: a client that
            # predates the frame ignores both this key and the frames.
            "utterance_frames": True,
        }
        # Surface (never silently drop) any handshake override the admin config
        # locked out, so the client can see why it had no effect.
        if overrides_ignored:
            ready_msg["overrides_ignored"] = overrides_ignored
        if req_override_profile:
            ready_msg["profile_applied"] = ident.request_profile_applied
        async with send_lock:
            await _safe_ws_send(ws, ready_msg)

        # Start the consumer before any audio is queued (the handshake byte, pending
        # audio, or the receive loop) so nothing waits on a not-yet-running pump.
        consumer_task = asyncio.create_task(_pump())
        if pending_audio:
            await transport.feed(pending_audio)

        # ---- main receive loop (PRODUCER) ----
        # Pure producer: drain ws.receive() and hand audio/control to the consumer.
        # It must never await a decode (that wedged the socket) — enqueue and move on.
        # Per-identity idle timeout (a trusted profile may allow a longer silence
        # grace); resolved now that the model + identity are known.
        idle_timeout = float(effective_config.cfg_for(final_model, "STREAMING_IDLE_TIMEOUT_S", ident) or 0.0)
        # The deadline is anchored to the last AUDIO byte, not the last frame.
        # Wrapping each receive() in the FULL timeout made the control defeatable
        # by any inbound frame — an empty binary frame (a no-op sink write) or an
        # unrecognised control JSON re-armed it forever, so a handful of sockets
        # could pin every STREAMING_MAX_SESSIONS slot (plus their ffmpeg
        # subprocesses) while decoding nothing. Both docstrings already describe
        # this as bounding a connection that "sends no audio". The dictation page
        # sends no keepalive of any kind, and a live mic streams PCM continuously
        # even through silence, so a genuine session is never cut short.
        _loop = asyncio.get_running_loop()
        _last_audio = _loop.time()
        try:
            while True:
                try:
                    if idle_timeout > 0:
                        budget = idle_timeout - (_loop.time() - _last_audio)
                        if budget <= 0:
                            raise asyncio.TimeoutError
                    else:
                        budget = 0.0  # <= 0 keeps the "disabled" semantics
                    msg = await _receive_idle(ws, budget)
                except asyncio.TimeoutError:
                    logger.info("[stream %s] idle %.0fs — closing", session_id[:8], idle_timeout)
                    async with send_lock:
                        try:
                            await ws.send_json({"type": "error", "code": "idle_timeout",
                                                "message": f"no audio for {idle_timeout:.0f}s; closing"})
                        except Exception:  # noqa: BLE001 — peer may be gone
                            pass
                    break
                if msg.get("type") == "websocket.disconnect":
                    break
                if _auth_revoked:
                    # Latched by _refresh_ident on the consumer side. Every sink
                    # is already cut (the decode guards); this closes the socket
                    # too, at the first frame after the revocation.
                    async with send_lock:
                        try:
                            await ws.send_json({"type": "error", "code": "unauthorized",
                                                "message": "credential no longer valid; closing"})
                        except Exception:  # noqa: BLE001 — peer may be gone
                            pass
                    break
                if msg.get("bytes") is not None:
                    # Only a NON-EMPTY audio payload re-arms the idle deadline.
                    if msg["bytes"]:
                        _last_audio = _loop.time()
                    await transport.feed(msg["bytes"])
                    if getattr(transport, "dead", False):
                        # The ffmpeg decoder exited (corrupt container bytes, a
                        # crash): every later frame would be discarded while the
                        # audio keeps re-arming the idle deadline. Say so and
                        # end the session; the close still commits what decoded.
                        logger.info("[stream %s] audio decoder stopped — closing",
                                    session_id[:8])
                        async with send_lock:
                            try:
                                await ws.send_json({"type": "error", "code": "decoder_failed",
                                                    "message": "audio decoder stopped; closing"})
                            except Exception:  # noqa: BLE001 — peer may be gone
                                pass
                        break
                elif msg.get("text") is not None:
                    try:
                        ctrl = json.loads(msg["text"])
                    except (ValueError, TypeError, RecursionError):
                        # See the handshake guard above — deep nesting raises
                        # RecursionError, which slipped past and tore down the
                        # whole dictation session instead of being ignored.
                        continue
                    if not isinstance(ctrl, dict):
                        continue
                    kind = ctrl.get("type")
                    if kind == "flush":
                        if not _flush_pending:
                            _flush_pending = True
                            audio_q.put_nowait(("flush", None))
                    elif kind == "stop":
                        break
        finally:
            # Flush the encoded tail into the session (→ sink → queue), then signal
            # end-of-stream and let the consumer drain everything still queued
            # (incl. that tail) before we finalize/close the session.
            await transport.aclose()
            audio_q.put_nowait(("stop", None))
            try:
                await consumer_task
            except Exception:  # noqa: BLE001
                pass
        if not _auth_revoked:
            try:
                await session.close()
            except _CredentialRevoked:
                # Revoked between the pump's last decode and the stop: the
                # drain's final decode re-auths and raises HERE, not in the
                # pump — without this it reached the blanket handler below
                # (traceback, an error ledger row, "internal error", close 1000)
                # instead of the 4401 every other revocation path sends.
                _auth_revoked = True
                async with send_lock:
                    try:
                        await ws.send_json({"type": "error", "code": "unauthorized",
                                            "message": "credential no longer valid; closing"})
                    except Exception:  # noqa: BLE001 — peer may be gone
                        pass
        if _auth_revoked:
            # No session.close() here: its drain finalizes the in-flight
            # utterance and emits the closing document, i.e. it would hand a
            # last transcript to an identity that no longer exists. The outer
            # finally still calls it (guarded) so the per-session VAD executor
            # and the slot are released; the decode guard makes that drain a
            # no-op raise it swallows.
            async with send_lock:
                try:
                    await ws.close(code=_WS_UNAUTH)
                except (RuntimeError, WebSocketDisconnect):
                    pass
        else:
            async with send_lock:
                try:
                    await ws.send_json({"type": "closing"})
                    await ws.close()
                except (RuntimeError, WebSocketDisconnect):
                    pass
    except WebSocketDisconnect:
        pass  # peer gone; teardown happens in the finally
    except Exception as exc:  # noqa: BLE001
        # Client-chosen handshake strings (model / language / response_format)
        # land verbatim in exception messages, and any exception raised outside
        # the two inner blocks that already screen theirs reaches this line — so
        # screen it too before it hits the line-oriented rotating log.
        logger.exception("[stream %s] error: %s",
                         session_id[:8], store_common.log_safe(str(exc)))
        # The ledger otherwise never hears of a failed session (utterances
        # record only on success): one error row under the session's job id,
        # classified like a batch failure. Best effort — locals bound only
        # after the handshake may be missing.
        try:
            _ec, _es = metrics.classify_error(exc, status="error",
                                              stage="transcribing")
            _loc = locals()
            metrics.record_transcription(
                model=str(_loc.get("final_model") or ""), audio_dur=0.0,
                proc_dur=0.0, status="error", words=0, kind="dictate",
                request_id=session_id, user_id=user.get("user_id"),
                key_id=user.get("key_id"), username=user.get("username"),
                key_label=user.get("key_label"),
                job_id=_loc.get("usage_job_id") or session_id,
                error_class=_ec, error_stage=_es)
        except Exception:  # noqa: BLE001 — never mask the real error
            pass
        try:
            # Lock like every other send: on a setup-window failure the consumer
            # task may still be mid-emit (it is cancelled later, in the finally),
            # so an unguarded send here could interleave with it.
            async with send_lock:
                # Generic client message — never forward the raw exception text
                # (paths/internals); the full traceback is logged just above.
                await ws.send_json({"type": "error", "code": "internal", "message": "internal error"})
                await ws.close()
        except Exception:  # noqa: BLE001
            pass
    finally:
        # Release the slots and the gauge FIRST. They used to sit after three
        # awaits guarded by `except Exception`, and CancelledError is a
        # BaseException — a cancellation delivered anywhere in the teardown
        # (lifespan shutdown is the reachable one) skipped both, permanently
        # burning one of STREAMING_MAX_SESSIONS. The per-user slot joins them
        # for the same reason. All three are idempotent and touch nothing the
        # teardown below needs.
        metrics.in_flight_transcriptions -= 1
        _active_sessions.discard(session_id)
        if _stream_held is not None:
            _stream_sessions.release(_stream_held)
            _stream_held = None
        # Same reason as the slots above: released before any await, so a
        # cancellation in the teardown cannot pin a model in VRAM forever.
        while _model_leases_held:
            tx_models._release_model_lease(_model_leases_held.pop())
        jobs.job_end(session_id)
        # Idempotent backstop so every exit path (normal, disconnect, error)
        # converges here and the ffmpeg subprocess + stdout-reader task are
        # always torn down. On the normal path the inner `finally` already
        # aclose'd the transport (to flush the encoded tail into the session
        # before session.close), so these are no-ops there; on an error mid-
        # setup they are the only cleanup that runs.
        if transport is not None:
            try:
                await transport.aclose()
            except Exception:  # noqa: BLE001
                pass
        # Stop the consumer before closing the session so the two never touch the
        # session concurrently. On the normal path it has already finished (awaited
        # above); on an error/disconnect path cancel + await it here.
        if consumer_task is not None and not consumer_task.done():
            consumer_task.cancel()
            try:
                await consumer_task
            except (Exception, asyncio.CancelledError):  # noqa: BLE001
                pass
        if session is not None:
            try:
                await session.close()
            except Exception:  # noqa: BLE001 — peer already gone
                pass


# --- Dictation page -----------------------------------------------------------


@router.get("/dictate", response_class=HTMLResponse,
            dependencies=[Depends(web_common.require_user_webui_host)])
async def dictate_page() -> HTMLResponse:
    """Live dictation page: mic → 16 kHz PCM → WebSocket, rendering stabilized
    partials + append-only finals, plus a batch mode that POSTs the whole clip.

    Rendered through `render_page`, so it carries the same header, nav, scale
    picker, severity pills, sign-out and login gate as every other WebUI page —
    including the shared gate a signed-out visitor sees. Auth is the ordinary
    session cookie: the page has no API-key field and talks only to its own
    origin, so the browser attaches the cookie to both the WebSocket handshake
    and the batch POST. Shell auth matches the other user-tier pages (host
    allowlist only, nothing sensitive rendered server-side); the WebSocket and
    the transcription endpoint each enforce their own credential.

    The template is templates/dictate.html next to this module (not under
    static/, which is served raw and publicly — this is an un-substituted
    server-side template). Unlike the other page templates it is read PER
    REQUEST, deliberately not cached at import: it is large and mostly client
    JS, and re-reading it means an edit shows up on a browser reload without a
    restart. templates.load resolves the file from this module's own
    directory, which is the right anchor here.
    """
    try:
        template = templates.load(__file__, "dictate.html")
    except (OSError, ValueError) as exc:   # ValueError: CRLF / not UTF-8
        logger.error("[dictate] cannot read page template: %s", exc)
        return HTMLResponse("<h1>dictate unavailable</h1>", status_code=500)
    return HTMLResponse(
        web_common.render_page(template, current="dictate"),
        headers={"Cache-Control": "no-store"},
    )


def _reset_for_tests() -> None:
    """Test-only: a socket torn down without the route's finally would strand
    an id and make the cap tests' exact-count pins order-dependent."""
    _active_sessions.clear()
