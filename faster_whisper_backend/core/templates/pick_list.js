
<script>(function(){
  var _open = null;
  function esc(s) {
    return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function _renderPickList(opts) {
    opts = opts || {};
    var multi = opts.multi !== false;
    var word = opts.wordPlural || 'items';
    var picked = Array.isArray(opts.picked) ? opts.picked.slice() : [];
    var rows = [], labels = {}, fetchGen = 0;
    var el = opts.mount || document.createElement('span');
    el.classList.add('picker');
    el.innerHTML = '';
    var btn = document.createElement('button');
    btn.type = 'button';
    btn.setAttribute('aria-haspopup', 'listbox');
    btn.setAttribute('aria-expanded', 'false');
    if (opts.title) btn.title = opts.title;
    btn.appendChild(document.createTextNode(word + ' '));
    var n = document.createElement('span'); n.className = 'n'; btn.appendChild(n);
    btn.appendChild(document.createTextNode(' ▾'));
    var pop = document.createElement('div');
    pop.className = 'pick-pop'; pop.setAttribute('role', 'dialog');
    pop.setAttribute('aria-label', opts.ariaLabel || ('pick ' + word)); pop.hidden = true;
    // Top-layer popover placed by the shared ladder: start-aligned under the
    // button, flipped to end-align before it may cross the toolbar (the page
    // canvas), then clamped into the viewport. Resolved lazily: POPOVER_JS
    // may be injected after this block.
    pop.setAttribute('popover', 'manual');
    var ctl = null;
    function placer() {
      if (!ctl && window._anchorPopover) {
        ctl = window._anchorPopover(pop, btn, {
          align: 'start',
          boundary: function() { return btn.closest('.subbar') || btn.closest('main') || document.body; }
        });
      } else if (!ctl && pop.hasAttribute('popover')) {
        // No positioner: a manual popover never shows without showPopover(),
        // so degrade to a plain [hidden]-toggled layer.
        pop.removeAttribute('popover'); pop.style.position = 'absolute';
      }
      return ctl;
    }
    var q = document.createElement('input');
    q.type = 'search'; q.placeholder = opts.placeholder || ('search ' + word);
    q.setAttribute('aria-label', q.placeholder);
    var list = document.createElement('div');
    list.className = 'pick-list'; list.setAttribute('role', 'listbox');
    if (multi) list.setAttribute('aria-multiselectable', 'true');
    var foot = document.createElement('div'); foot.className = 'pick-foot';
    var fs = document.createElement('span'); foot.appendChild(fs);
    var clr = document.createElement('button');
    clr.type = 'button'; clr.className = 'pick-clear'; clr.textContent = 'clear'; foot.appendChild(clr);
    pop.appendChild(q); pop.appendChild(list); pop.appendChild(foot);
    el.appendChild(btn); el.appendChild(pop);

    function label() {
      n.textContent = picked.length
        ? (multi ? picked.length + ' picked' : (labels[picked[0]] || picked[0]))
        : (opts.anyLabel || 'any');
      btn.setAttribute('aria-expanded', _open === inst ? 'true' : 'false');
    }
    function footer() { fs.textContent = picked.length + ' of ' + rows.length + ' picked'; }
    function draw(needle) {
      var max = 1;
      rows.forEach(function(r) { max = Math.max(max, Number(r.value) || 0); });
      var vis = rows.filter(function(r) {
        return !needle || ((r.label || '') + ' ' + (r.sub || '')).toLowerCase().indexOf(needle) !== -1; });
      list.innerHTML = vis.length ? vis.map(function(r) {
        var on = picked.indexOf(r.id) !== -1;
        return '<label class="pick-opt' + (r.stale ? ' stale' : '') + '">'
          + '<input type="' + (multi ? 'checkbox' : 'radio') + '" data-id="' + esc(r.id) + '"' + (on ? ' checked' : '') + '>'
          + '<span class="name">' + esc(r.label)
          + (r.me ? '<span class="me">you</span>' : '')
          + (r.sub ? '<span class="sub">' + esc(r.sub) + '</span>' : '') + '</span>'
          + (r.value != null
              ? '<span class="bar"><i style="width:' + ((Number(r.value) || 0) / max * 100).toFixed(0) + '%"></i></span>'
                + '<span class="v">' + esc(opts.fmt ? opts.fmt(r.value) : String(r.value)) + '</span>'
              : '')
          + '</label>';
      }).join('') : '<div class="pick-note">nothing matches</div>';
      footer();
    }
    function open() {
      if (_open === inst) { close(); return; }
      if (_open) _open.close();
      _open = inst;
      pop.hidden = false; label();
      var c = placer(); if (c) c.show();
      q.value = '';
      list.innerHTML = '<div class="pick-note">loading…</div>';
      // Per-open generation: after a close + reopen, the first open's late
      // resolve (stale rows) or reject ("not available") must not overwrite
      // what the newer open drew.
      var gen = ++fetchGen;
      function current() { return _open === inst && gen === fetchGen; }
      Promise.resolve().then(function() { return opts.fetchRows ? opts.fetchRows() : rows; })
        .then(function(rs) {
          if (!current()) return;
          rows = Array.isArray(rs) ? rs : [];
          rows.forEach(function(r) { labels[r.id] = r.label; });
          draw(''); label();
          try { q.focus(); } catch (_) {}
        })
        .catch(function() {
          if (!current()) return;
          list.innerHTML = '<div class="pick-note">' + esc(opts.errorNote || 'not available') + '</div>';
        });
    }
    function close() {
      if (ctl) ctl.hide();
      pop.hidden = true; if (_open === inst) _open = null; label();
    }
    btn.addEventListener('click', function(e) { e.stopPropagation(); open(); });
    q.addEventListener('input', function() { draw(q.value.trim().toLowerCase()); });
    list.addEventListener('change', function(e) {
      var id = e.target && e.target.dataset.id; if (!id) return;
      if (multi) picked = e.target.checked ? picked.concat([id]) : picked.filter(function(x) { return x !== id; });
      else picked = [id];
      footer(); label();
      if (opts.onChange) opts.onChange(picked.slice());
      if (!multi) close();
    });
    clr.addEventListener('click', function() {
      if (!picked.length) return;
      picked = [];
      list.querySelectorAll('input').forEach(function(i) { i.checked = false; });
      footer(); label();
      if (opts.onChange) opts.onChange([]);
    });
    var inst = {
      el: el,
      getPicked: function() { return picked.slice(); },
      setPicked: function(ids) {
        var next = Array.isArray(ids) ? ids.slice() : [];
        // A page echoing the widget's own ids back must not redraw the open
        // list: that rebuilds it under a click that is still bubbling.
        if (next.length === picked.length &&
            next.every(function(x, i) { return x === picked[i]; })) return;
        picked = next;
        label();
        if (!pop.hidden) draw(q.value.trim().toLowerCase());
      },
      setLabels: function(m) { Object.keys(m || {}).forEach(function(k) { labels[k] = m[k]; }); label(); },
      close: close,
      contains: function(node) { return el.contains(node) || pop.contains(node); },
    };
    label();
    return inst;
  }
  document.addEventListener('click', function(e) {
    if (_open && e.target.isConnected && !_open.contains(e.target)) _open.close();
  });
  document.addEventListener('keydown', function(e) { if (e.key === 'Escape' && _open) _open.close(); });
  window._renderPickList = _renderPickList;
})();</script>
