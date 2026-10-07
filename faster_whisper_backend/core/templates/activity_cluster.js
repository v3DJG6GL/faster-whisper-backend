
<script>(function(){
  var btn = document.getElementById('hact');
  // A headerless shell (the hub's status strip is a plain div) can never
  // satisfy syncAllowed()'s header-scoped selector, so the button would stay
  // hidden forever — bail before installing the poll timer and listeners.
  if (!btn || !document.querySelector('header')) return;
  var pop = document.getElementById('hact-pop');
  var jobsEl = document.getElementById('hact-jobs');
  var gpuEl = document.getElementById('hact-gpu');
  var gpuvEl = document.getElementById('hact-gpuv');
  var vramEl = document.getElementById('hact-vram');
  var vramvEl = document.getElementById('hact-vramv');
  var es = null, last = null, lastTs = 0, allowed = false;
  // Cancel-in-flight pids. Lives OUTSIDE the DOM because renderPop rebuilds
  // the popover wholesale on every 1 Hz frame -- a disabled attribute set on
  // the live button node is wiped by the very next rebuild.
  var cancelling = {};
  var retryTimer = null, delay = 1500;

  function esc(s){ return String(s == null ? '' : s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
  function onStats(){ return window.__current_page === 'stats'; }
  function gb(mb){ return ((mb || 0) / 1024).toFixed(1); }
  // Job-kind → accent class (mockup palette: transcribe cyan, translate
  // magenta, dictate green, download yellow, preload muted). Keep in step
  // with jobs.KINDS and the /stats kindchip CSS (stats/templates/stats.html).
  function kindCls(k){
    return {transcribe:'tr', stream:'tr', translate:'tl',
            dictate:'dc', download:'dl', preload:'pl'}[k] || '';
  }

  function setBar(el, pct){
    if (!el) return;
    if (pct == null) { el.style.width = '0'; el.className = ''; return; }
    el.style.width = Math.max(0, Math.min(100, pct)).toFixed(0) + '%';
    el.className = pct >= 92 ? 'crit' : (pct >= 75 ? 'warn' : '');
  }

  function feed(snap){
    if (!snap) return;
    last = snap; lastTs = Date.now();
    btn.classList.remove('stale');
    btn.title = 'server activity — click for running jobs';
    delay = 1500;
    var jobs = snap.jobs || [];
    jobsEl.textContent = jobs.length;
    btn.classList.toggle('idle', jobs.length === 0);
    var gpu = snap.gpu || null;
    setBar(gpuEl, gpu ? (gpu.util_pct != null ? gpu.util_pct : null) : null);
    setBar(vramEl, gpu && gpu.mem_total_mb
      ? gpu.mem_used_mb / gpu.mem_total_mb * 100 : null);
    // Own-scope viewers get a coarse gpu dict {busy, mem_*}: no util_pct,
    // so the readout says busy/idle instead of a percentage.
    gpuvEl.textContent = gpu && gpu.util_pct != null
      ? Math.round(gpu.util_pct) + '%'
      : (gpu && gpu.busy != null ? (gpu.busy ? 'busy' : 'idle') : '\u2013');
    vramvEl.textContent = gpu && gpu.mem_used_mb != null
      ? gb(gpu.mem_used_mb) + 'G' : '\u2013';
    if (!pop.hidden) renderPop();
  }
  // The /stats page renderer calls this with its own (full) snapshot so the
  // cluster stays live there without a second SSE stream.
  window._fwFeedActivity = feed;

  function renderPop(){
    var s = last;
    if (!s) {
      pop.innerHTML = '<div class="sec"><div class="empty">no data yet</div></div>';
      return;
    }
    // progress_id is emitted for admins AND the job's own owner (jobs_snapshot),
    // and the cancel endpoint re-checks ownership — same policy as /stats.
    var jobs = s.jobs || [];
    // Prune marks whose pid left the snapshot so the map cannot grow.
    Object.keys(cancelling).forEach(function(pid){
      if (!jobs.some(function(j){ return j.progress_id === pid; }))
        delete cancelling[pid];
    });
    var h = '<div class="sec"><div class="sec-t">Running now · '
      + jobs.length + '</div>';
    if (!jobs.length) h += '<div class="empty">— none —</div>';
    jobs.forEach(function(j){
      var kc = kindCls(j.kind);
      var pct = j.progress != null ? Math.round(j.progress * 100) : null;
      var sub = [j.detail, j.user].filter(Boolean).map(esc).join(' · ');
      h += '<div class="hact-job">'
        + '<span class="hact-kind ' + kc + '">' + esc(j.kind) + '</span>'
        + '<span class="m">' + esc(j.model || '')
        + (sub ? ' <small>· ' + sub + '</small>' : '') + '</span>'
        + (pct != null
            ? '<span class="hact-jbar"><i class="' + kc + '" style="width:'
              + pct + '%"></i></span>'
              + '<span class="p">' + esc(j.step || (pct + '%')) + '</span>'
            : '<span class="p">' + esc(j.step || j.stage || '…') + '</span>')
        + (j.progress_id
            ? '<button class="hact-cancel"'
              + (cancelling[j.progress_id] ? ' disabled' : '')
              + ' data-pid="' + esc(j.progress_id)
              + '" title="cancel this job">✕</button>'
            : '')
        + '</div>';
    });
    h += '</div><div class="sec"><div class="sec-t">Server</div>';
    function row(lbl, pct, colorCls, val){
      h += '<div class="resline"><span class="rl">' + lbl + '</span>'
        + '<span class="rbar"><i class="' + colorCls + '" style="width:'
        + Math.max(0, Math.min(100, pct || 0)).toFixed(0) + '%"></i></span>'
        + '<span class="rv">' + val + '</span></div>';
    }
    var gpu = s.gpu, host = s.host || {};
    if (gpu && gpu.util_pct == null && gpu.busy != null) {
      row('GPU', gpu.busy ? 100 : 0, 'c-cyan', gpu.busy ? 'busy' : 'idle');
      if (gpu.mem_total_mb)
        row('VRAM', gpu.mem_used_mb / gpu.mem_total_mb * 100, 'c-mag',
            gb(gpu.mem_used_mb) + ' / ' + gb(gpu.mem_total_mb) + ' G');
    } else if (gpu) {
      row('GPU', gpu.util_pct || 0, 'c-cyan',
          (gpu.util_pct != null ? Math.round(gpu.util_pct) : '—') + '%'
          + (gpu.temp_c != null ? ' · ' + Math.round(gpu.temp_c) + '°C' : ''));
      if (gpu.mem_total_mb)
        row('VRAM', gpu.mem_used_mb / gpu.mem_total_mb * 100, 'c-mag',
            gb(gpu.mem_used_mb) + ' / ' + gb(gpu.mem_total_mb) + ' G');
    }
    if (host.ram_pct != null)
      row('RAM', host.ram_pct, 'c-green',
          host.ram_total_mb
            ? gb(host.ram_used_mb) + ' / ' + gb(host.ram_total_mb) + ' G'
            : gb(host.ram_used_mb) + ' G');
    if (host.cpu_pct != null)
      row('CPU', host.cpu_pct, 'c-cyan', host.cpu_pct.toFixed(0) + '%');
    h += '</div>';
    var models = s.models || [];
    if (models.length) {
      h += '<div class="sec"><div class="sec-t">Loaded models · '
        + models.length + '</div>';
      models.forEach(function(m){
        // A falsy vram_mb (null, or a cuda NVML delta of exactly 0) is no
        // reading, as in the /stats models table -- not "0.0G".
        var bits = [m.device, m.compute_type,
                    m.vram_mb ? gb(m.vram_mb) + 'G' : null]
          .filter(Boolean).map(esc).join(' · ');
        h += '<div class="modline"><span class="mn">' + esc(m.name)
          + '</span><span>' + bits + '</span></div>';
      });
      h += '</div>';
    }
    h += '<div class="hact-foot"><a href="/stats">Open stats →</a></div>';
    // Not while a pointer is down inside the pop: swapping innerHTML between
    // mousedown and mouseup detaches the pressed .hact-cancel, the click then
    // lands on the pop itself and the cancel is silently dropped. _lastH
    // stays put, so the release re-render below applies the latest frame.
    if (pop._held) return;
    if (h !== pop._lastH) { pop.innerHTML = h; pop._lastH = h; }
  }
  pop.addEventListener('pointerdown', function(){ pop._held = true; });
  function releaseHold(){
    if (!pop._held) return;
    pop._held = false;
    // After the click this pointerup belongs to has been dispatched.
    setTimeout(function(){ if (!pop.hidden) renderPop(); }, 0);
  }
  document.addEventListener('pointerup', releaseHold);
  document.addEventListener('pointercancel', releaseHold);

  // Shared placement ladder (POPOVER_JS): end-aligned under the button,
  // flipped to start-align before it may leave the header canvas, clamped
  // into the viewport; re-placed on scroll / resize while open.
  var ctl = null;
  function placer(){
    if (!ctl && window._anchorPopover) {
      ctl = window._anchorPopover(pop, btn, {
        align: 'end',
        boundary: function(){ return document.querySelector('.header-inner'); }
      });
    } else if (!ctl && pop.hasAttribute('popover')) {
      // No positioner: a manual popover never shows without showPopover(),
      // so degrade to a plain [hidden]-toggled layer.
      pop.removeAttribute('popover'); pop.style.position = 'absolute';
    }
    return ctl;
  }
  function openPop(){
    pop.hidden = false; btn.setAttribute('aria-expanded', 'true');
    renderPop();
    var c = placer(); if (c) c.show();
  }
  function closePop(){
    var c = placer(); if (c) c.hide();
    pop.hidden = true; btn.setAttribute('aria-expanded', 'false');
  }
  btn.addEventListener('click', function(e){
    e.stopPropagation();
    if (pop.hidden) openPop(); else closePop();
  });
  document.addEventListener('click', function(e){
    if (!pop.hidden && !pop.contains(e.target)) closePop();
  });
  document.addEventListener('keydown', function(e){
    if (e.key === 'Escape' && !pop.hidden) closePop();
  });
  // One cancel path for the popover and the /stats jobs table. Cookie-
  // authenticated pages go through _csrf_mw, which 403s any unsafe-method
  // request without the token (same as _signOut). The endpoint re-checks
  // that the caller owns the job (or is an admin).
  window._fwCancelJob = function(pid, c){
    if (!pid) return;
    if (c) c.disabled = true;
    cancelling[pid] = 1;
    fetch('/v1/audio/transcriptions/cancel/' + pid,
          { method: 'POST',
            headers: { 'X-CSRF-Token': window._csrfToken() } })
      .then(function(r){ if (!r.ok) { if (c) c.disabled = false; delete cancelling[pid]; } },
            function(){ if (c) c.disabled = false; delete cancelling[pid]; });
  };
  // The /stats jobs table keeps a cancel it sent disabled across its 1 Hz
  // re-renders by asking the same in-flight set.
  window._fwIsCancelling = function(pid){ return !!cancelling[pid]; };
  pop.addEventListener('click', function(e){
    var c = e.target.closest('.hact-cancel');
    if (!c) return;
    window._fwCancelJob(c.dataset.pid, c);
  });

  function openStream(){
    if (es || onStats() || !allowed) return;
    if (document.visibilityState === 'hidden') return;
    try {
      es = new EventSource('/stats/stream?lite=1');
      es.onmessage = function(ev){
        try { feed(JSON.parse(ev.data)); } catch(_) {}
      };
      es.onerror = function(){
        // A transient drop keeps readyState CONNECTING and the browser
        // retries on its own; an HTTP error response leaves it CLOSED (2)
        // with no retry, so reopen ourselves with backoff (mirrors /stats).
        if (es && es.readyState === 2) {
          closeStream();
          retryTimer = setTimeout(openStream, delay);
          delay = Math.min(delay * 1.7, 30000);
        }
      };
    } catch(_) {}
  }
  function closeStream(){
    if (retryTimer) { clearTimeout(retryTimer); retryTimer = null; }
    if (es) { try { es.close(); } catch(_) {} es = null; }
  }

  // Stale detection: the lite stream ticks at 1 Hz; after >5 s of silence
  // (3+ missed intervals) grey the bars with an explanatory title (the
  // data-tip CSS tooltip is scoped to the .vtag chip, not this button).
  setInterval(function(){
    if (!allowed || !lastTs) return;
    if (Date.now() - lastTs > 5000) {
      btn.classList.add('stale');
      btn.title = 'activity feed stale — reconnecting';
    }
  }, 2500);

  // Visibility: mirror the /stats page — no hidden-tab streams (browser
  // 6-connections-per-origin cap).
  document.addEventListener('visibilitychange', function(){
    if (document.visibilityState === 'hidden') closeStream();
    else openStream();
  });

  // Gate: mirror the stats nav-link's `.allowed` (set async by
  // _refreshAuthChrome after /auth/whoami resolves) onto the button. A
  // MutationObserver catches the async class flip; an initial sync covers
  // the already-resolved case.
  function syncAllowed(){
    var link = document.querySelector('header a.page-link[data-page="stats"]');
    var ok = !!(link && link.classList.contains('allowed'));
    if (ok === allowed) return;
    allowed = ok;
    btn.classList.toggle('allowed', ok);
    if (ok) { btn.hidden = false; openStream(); }
    else {
      btn.hidden = true; closeStream();
      // closePop, not a bare hidden flip: the placement controller must
      // hide() too, or the native popover stays open in the top layer and
      // its scroll / resize listeners keep re-placing it.
      closePop();
    }
  }
  var header = document.querySelector('header');
  if (header && window.MutationObserver) {
    new MutationObserver(syncAllowed).observe(
      header, { attributes: true, subtree: true,
                attributeFilter: ['class'] });
  }
  window.addEventListener('whisper:auth-changed', function(){
    setTimeout(syncAllowed, 0);
  });
  syncAllowed();
})();</script>
