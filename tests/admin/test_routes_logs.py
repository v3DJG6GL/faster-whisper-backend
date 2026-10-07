"""Source pins for the /logs viewer script (admin/logs_routes.py serves it)."""


def test_logs_page_search_skips_fold_controls_and_button_labels(client):
    """The search filter matched el.textContent, which also holds the fold
    controls' "▸ N more segment rows / show all" and a clamped line's
    "Show all · N chars" button, so "show" or "rows" hit every control and
    clipped line, and hide mode hid the controls that reveal folded rows."""
    html = client.get("/logs").text
    f = html[html.index("function applyFilter(el)"):]
    f = f[:f.index("\n  }\n")]
    assert "el.classList.contains('fold-ctl')" in f
    assert "_lineText.get(el)" in f
    assert "el.textContent.toLowerCase()" not in f
    # makeLine records the line's own text for the filter to read.
    m = html[html.index("function makeLine(line, st)"):]
    assert "_lineText.set(el, txt);" in m[:m.index("\n  }\n")]


def test_logs_page_unfold_moves_the_control_past_revealed_rows(client):
    """"show 50" revealed rows in place, so the control was followed by an
    unfolded row; _foldRemaining stopped there, returned 0 and the control
    was removed, stranding every row past the first 50 with no way back."""
    html = client.get("/logs").text
    f = html[html.index("function _unfold(ctl, n)"):]
    f = f[:f.index("\n  }\n")]
    assert "last = e;" in f
    assert f.index("if (last) last.after(ctl);") \
        < f.index("const left = _foldRemaining(ctl);")


def test_logs_page_resets_render_state_when_the_log_is_emptied(client):
    """Clear and the reconnect replay empty #log; a stale inSeg/pipe state
    would fold the next rows with no control in the DOM to reveal them."""
    html = client.get("/logs").text
    clear = html[html.index("clearBtn.addEventListener('click'"):]
    assert "_resetLiveDim();" in clear[:clear.index("});")]
    reopen = html[html.index("function openLogStream()"):]
    reopen = reopen[:reopen.index("es = new EventSource")]
    assert "log.innerHTML = '';" in reopen and "_resetLiveDim();" in reopen
    assert "_liveStarted" not in html


def test_logs_page_trimmed_fold_control_ends_the_live_block(client):
    """The live-tail trim unfolded and removed a fold control while its
    Segments / PIPELINE block was still arriving; _liveDim kept ctl/inSeg/
    pipe, so decorate() folded every later row of the block behind a control
    no longer in the DOM."""
    html = client.get("/logs").text
    trim = html[html.index("while (log.childElementCount > _LOG_DOM_MAX + _olderInDom)"):]
    trim = trim[:trim.index("_unfold(first, 1e9);")]
    assert "if (first === _liveDim.ctl) {" in trim
    for reset in ("_liveDim.ctl = null;", "_liveDim.inSeg = false;",
                  "_liveDim.pipe = false;"):
        assert reset in trim


def test_logs_page_load_older_batch_survives_the_live_trim(client):
    """The next live line trimmed a just-loaded "Load older" batch straight
    back out (the trim bound ignored it) and rewound the cursor, so the next
    click fetched the same batch again."""
    html = client.get("/logs").text
    assert "while (log.childElementCount > _LOG_DOM_MAX + _olderInDom)" in html
    # The bound counts elements: the batch's fold controls count too, read
    # off the fragment BEFORE the insert empties it.
    count = html.index("const added = frag.childElementCount;")
    insert = html.index("log.insertBefore(frag, log.firstChild);")
    assert count < insert
    older = html[insert:]
    older = older[:older.index("next_skip")]
    assert "_olderInDom += added;" in older
    assert "_olderInDom += lines.length;" not in older
    assert "_logsSkip += lines.length;" in older     # the cursor counts lines
    clear = html[html.index("clearBtn.addEventListener('click'"):]
    assert "_olderInDom = 0;" in clear[:clear.index("});")]
    reopen = html[html.index("function openLogStream()"):]
    assert "_olderInDom = 0;" in reopen[:reopen.index("es = new EventSource")]


def test_logs_page_load_older_drops_a_batch_from_before_a_reconnect(client):
    """A "Load older" fetch still in flight when openLogStream() reset the
    DOM and cursors landed its old-offset batch above the replayed backlog
    and added its length to the fresh counters."""
    html = client.get("/logs").text
    reopen = html[html.index("function openLogStream()"):]
    assert "_logsGen++;" in reopen[:reopen.index("es = new EventSource")]
    click = html[html.index("loadOlderBtn.addEventListener('click'"):]
    assert "const gen = _logsGen;" in click[:click.index("await fetch(url)")]
    after = click[click.index("await r.json();"):]
    assert after.index("if (gen !== _logsGen) return;") \
        < after.index("log.insertBefore(frag, log.firstChild);")


def test_logs_page_search_counter_follows_load_older_and_clear(client):
    """Load older and Clear changed the DOM without refreshing the n/m
    counter; Clear also left matchCurEl on a detached node."""
    html = client.get("/logs").text
    older = html[html.index("log.insertBefore(frag, log.firstChild);"):]
    assert "queueNav();" in older[:older.index("next_skip")]
    clear = html[html.index("clearBtn.addEventListener('click'"):]
    clear = clear[:clear.index("});")]
    assert "matchCurEl = null;" in clear and "updateNav();" in clear


def test_logs_page_reconnect_while_paused_shows_paused(client):
    """onerror sets "reconnecting…" regardless; onopen restored the pill only
    when not paused, so a paused tab kept "reconnecting…" on a live stream."""
    html = client.get("/logs").text
    onopen = html[html.index("es.onopen = () => {"):]
    onopen = onopen[:onopen.index("\n    };")]
    assert "statusEl.textContent = paused ? 'paused' : 'live';" in onopen
    assert "if (!paused) {" not in onopen


def test_logs_stream_delivers_a_line_logged_during_the_backlog(
        app_module, monkeypatch, tmp_path):
    """A line logged after the backlog read but before the tail took its
    offset was in neither; the offset is now sampled before the read."""
    import asyncio

    from faster_whisper_backend.admin import logs_routes

    log = tmp_path / "live.log"
    log.write_text("first\n", encoding="utf-8")
    monkeypatch.setattr(app_module.cfg, "LOG_FILE", str(log), raising=False)
    real = logs_routes._read_chain_window

    def _read_then_log(*a, **k):
        out = real(*a, **k)
        with open(log, "a", encoding="utf-8") as f:
            f.write("during\n")
        return out

    monkeypatch.setattr(logs_routes, "_read_chain_window", _read_then_log)
    monkeypatch.setattr(logs_routes, "_logs_stream_reauth",
                        lambda request, seen: seen)

    async def drive():
        gen = logs_routes._stream_log_lines(None)
        try:
            assert await gen.__anext__() == "data: first\n\n"
            assert await gen.__anext__() == "data: __LIVE_TAIL__\n\n"
            # Bounded: on a regression "during" is in neither the backlog
            # nor the tail and the generator polls forever — fail, not hang.
            assert await asyncio.wait_for(gen.__anext__(), timeout=5) \
                == "data: during\n\n"
        finally:
            await gen.aclose()

    asyncio.run(drive())
