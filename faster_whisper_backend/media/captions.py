"""Embedded broadcast captions (ATSC A/53 CEA-608/708 in H.264 SEI) that
carry no text.

TV-sourced H.264 (RTVE, many broadcasters) ships caption side data on every
frame even when no caption is ever sent: players list four empty "Closed
captions 1–4" tracks next to the subtitles we mux. `embedded_captions_empty`
reads the video bitstream (stream copy, no decode — seconds for a feature
film) and reports whether such captions exist AND are all padding, so the
packager can drop them while real captions survive.

Same stance as media/subtitle_mux.py: the source is a media-store path the
probe already accepted (no playlist-type demuxer), ffmpeg runs with
`-protocol_whitelist file`, nothing client-supplied becomes an option.
"""

from __future__ import annotations

import dataclasses
import logging
import subprocess
import time

from faster_whisper_backend.audio import ffmpeg as audio_ffmpeg
from faster_whisper_backend.core.store_common import log_safe

logger = logging.getLogger("whisper-api")

# SEI user_data_registered_itu_t_t35 → ATSC identifier "GA94", then
# user_data_type_code 0x03 = cc_data().
_GA94 = b"GA94\x03"
_CHUNK = 1 << 20
# Longest cc_data(): flags + em_data + 31 triples, with room for emulation-
# prevention bytes. A marker found closer than this to the end of the buffer
# waits for the next chunk.
_TAIL = 2 + 3 * 31 + 48


@dataclasses.dataclass
class A53Scan:
    blocks: int = 0          # cc_data() blocks seen
    pairs: int = 0           # valid byte pairs
    content_pairs: int = 0   # pairs carrying anything but padding

    @property
    def present(self) -> bool:
        return self.blocks > 0

    @property
    def empty(self) -> bool:
        return self.present and self.content_pairs == 0


def _unescape(b: bytes) -> bytes:
    """Drop H.264 emulation-prevention bytes (00 00 03 → 00 00)."""
    return b.replace(b"\x00\x00\x03", b"\x00\x00")


def _scan_block(block: bytes, scan: A53Scan) -> None:
    """One cc_data() body, starting at the flags byte after `GA94 03`."""
    if len(block) < 2:
        return
    count = block[0] & 0x1F
    scan.blocks += 1
    k = 2  # flags, em_data
    for _ in range(count):
        if k + 3 > len(block):
            return
        marker, b1, b2 = block[k], block[k + 1], block[k + 2]
        k += 3
        if not marker & 0x04:  # cc_valid
            continue
        scan.pairs += 1
        cc_type = marker & 0x03
        if cc_type in (0, 1):
            # CEA-608: 7-bit data with odd parity; 0x80 0x80 is padding.
            if (b1 & 0x7F) or (b2 & 0x7F):
                scan.content_pairs += 1
        elif b1 or b2:
            # CEA-708 DTVCC packet bytes: any non-zero byte is real data.
            scan.content_pairs += 1


def scan_a53(chunks, scan: "A53Scan | None" = None, *, stop_on_content: bool = True) -> A53Scan:
    """Scan an Annex B H.264 byte stream (an iterable of chunks)."""
    scan = scan or A53Scan()
    buf = b""
    for chunk in chunks:
        buf += chunk
        i = 0
        while True:
            j = buf.find(_GA94, i)
            if j < 0:
                break
            if len(buf) - j < len(_GA94) + _TAIL:
                break  # possibly cut short: finish with the next chunk
            _scan_block(_unescape(buf[j + len(_GA94): j + len(_GA94) + _TAIL]), scan)
            if stop_on_content and scan.content_pairs:
                return scan
            i = j + len(_GA94)
        # Keep only what may still hold an unfinished block.
        keep = len(_GA94) + _TAIL
        buf = buf[i:] if i else buf
        if len(buf) > keep:
            buf = buf[-keep:]
    # The stream's end: scan what is left, however short.
    i = 0
    while (j := buf.find(_GA94, i)) >= 0:
        _scan_block(_unescape(buf[j + len(_GA94):]), scan)
        i = j + len(_GA94)
    return scan


def embedded_captions_empty(src: str, *, video_index: int = 0, timeout: float = 300.0) -> bool:
    """True only when the H.264 video carries A/53 captions and none of them
    holds text. Any doubt (no ffmpeg, a read error, the time limit, real
    caption bytes) answers False: the captions are kept."""
    argv = [audio_ffmpeg.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin",
            "-protocol_whitelist", "file", "-i", src,
            "-map", f"0:v:{max(0, int(video_index))}", "-c", "copy",
            "-bsf:v", "h264_mp4toannexb", "-f", "h264", "-"]
    t0 = time.monotonic()
    try:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL)
    except OSError:
        return False
    scan = A53Scan()
    state = {"eof": False, "timed_out": False}

    def _chunks():
        while True:
            if time.monotonic() - t0 > timeout:
                state["timed_out"] = True
                return
            data = proc.stdout.read(_CHUNK)
            if not data:
                state["eof"] = True
                return
            yield data
    rc = None
    try:
        scan_a53(_chunks(), scan)
    finally:
        if state["eof"]:
            try:
                rc = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        else:  # stopped early: real captions found, or out of time
            proc.kill()
            proc.wait()
        proc.stdout.close()
    if scan.content_pairs:
        return False
    if state["timed_out"] or rc != 0:
        logger.info("[captions] scan inconclusive (rc=%s, timed out=%s) for %s",
                    rc, state["timed_out"], log_safe(src[-80:]))
        return False
    logger.info("[captions] %d empty caption block(s) in %.1fs",
                scan.blocks, time.monotonic() - t0)
    return scan.empty
