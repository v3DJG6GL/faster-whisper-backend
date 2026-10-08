
<script>(function(){

  function _renderLanguagePicker(opts) {
    opts = opts || {};
    var codes = Array.isArray(opts.initial) ? opts.initial.slice() : [];
    var allLangs = Array.isArray(opts.allLangs) ? opts.allLangs : [];
    var groups = opts.groups || {};
    var disabled = !!opts.disabled;
    var readOnly = !!opts.readOnly;

    var el = document.createElement('div');
    el.className = 'lang-picker' + (disabled ? ' disabled' : '');

    var pillsWrap = document.createElement('span');
    pillsWrap.style.cssText = 'display:inline-flex;flex-wrap:wrap;gap:0.25rem;align-items:center;';
    el.appendChild(pillsWrap);

    var addBtn = null;
    var dropEl = null;
    var dropContent = null;
    var searchInp = null;
    var popCtl = null;

    if (!readOnly) {
      addBtn = document.createElement('button');
      addBtn.type = 'button';
      addBtn.className = 'lang-picker-add';
      addBtn.textContent = '+ add';
      addBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        if (popCtl && popCtl.isOpen()) { _close(); return; }
        _renderDropdown('');
        if (searchInp) { searchInp.value = ''; }
        popCtl.show();
        if (searchInp) searchInp.focus();
        document.addEventListener('pointerdown', _onDocDown, true);
      });
      el.appendChild(addBtn);

      dropEl = document.createElement('div');
      dropEl.className = 'lang-picker-dropdown';
      dropEl.setAttribute('popover', 'manual');
      el.appendChild(dropEl);

      var searchWrap = document.createElement('div');
      searchWrap.className = 'lang-picker-search';
      searchInp = document.createElement('input');
      searchInp.type = 'text';
      searchInp.placeholder = 'Search languages…';
      searchInp.autocomplete = 'off';
      searchInp.addEventListener('input', function() { _renderDropdown(searchInp.value); });
      searchInp.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') { e.preventDefault(); _close(); addBtn.focus(); }
      });
      searchWrap.appendChild(searchInp);
      dropEl.appendChild(searchWrap);

      dropContent = document.createElement('div');
      dropEl.appendChild(dropContent);

      popCtl = window._anchorPopover(dropEl, addBtn, { align: 'left' });
      popCtl.hide();
    }

    function _onDocDown(ev) {
      if (dropEl && !dropEl.contains(ev.target) && addBtn && !addBtn.contains(ev.target)) _close();
    }
    function _close() {
      document.removeEventListener('pointerdown', _onDocDown, true);
      if (popCtl) popCtl.hide();
    }

    function _fire() { if (opts.onChange) opts.onChange(codes.slice()); }

    function _addCode(c) {
      if (codes.indexOf(c) >= 0) return;
      codes.push(c);
      codes.sort();
      _renderPills();
      _fire();
    }
    function _removeCode(c) {
      var i = codes.indexOf(c);
      if (i >= 0) codes.splice(i, 1);
      _renderPills();
      _fire();
    }
    function _toggleCode(c) {
      if (codes.indexOf(c) >= 0) _removeCode(c); else _addCode(c);
      _renderDropdown(searchInp ? searchInp.value : '');
    }

    function _renderPills() {
      pillsWrap.innerHTML = '';
      if (!codes.length) {
        var ghost = document.createElement('span');
        ghost.className = 'lang-picker-ghost';
        ghost.textContent = 'all languages';
        pillsWrap.appendChild(ghost);
        return;
      }
      codes.forEach(function(c) {
        var pill = document.createElement('span');
        pill.className = 'lang-pill';
        pill.textContent = c;
        if (!readOnly) {
          var x = document.createElement('button');
          x.type = 'button';
          x.className = 'lang-pill-x';
          x.textContent = '×';
          x.addEventListener('click', function(e) { e.stopPropagation(); _removeCode(c); });
          pill.appendChild(x);
        }
        pillsWrap.appendChild(pill);
      });
    }

    function _renderDropdown(query) {
      if (!dropContent) return;
      dropContent.innerHTML = '';
      var q = (query || '').toLowerCase();
      var filtered = allLangs.filter(function(l) {
        if (!q) return true;
        return l.code.indexOf(q) >= 0 || l.name.toLowerCase().indexOf(q) >= 0;
      });
      filtered.forEach(function(l) {
        var opt = document.createElement('div');
        opt.className = 'lang-picker-opt' + (codes.indexOf(l.code) >= 0 ? ' selected' : '');
        opt.innerHTML = '<span class="lp-code">' + l.code + '</span>'
          + '<span class="lp-name">' + l.name + '</span>'
          + '<span class="lp-check">✓</span>';
        opt.addEventListener('click', function() { _toggleCode(l.code); });
        dropContent.appendChild(opt);
      });
      // Family shortcuts at the bottom
      var groupKeys = Object.keys(groups).sort();
      var matchingGroups = groupKeys.filter(function(g) {
        return !q || g.toLowerCase().indexOf(q) >= 0;
      });
      if (matchingGroups.length) {
        var hdr = document.createElement('div');
        hdr.className = 'lang-picker-group-hdr';
        hdr.textContent = 'Add by family';
        dropContent.appendChild(hdr);
        matchingGroups.forEach(function(g) {
          var gc = groups[g];
          var allIn = gc.every(function(c) { return codes.indexOf(c) >= 0; });
          var opt = document.createElement('div');
          opt.className = 'lang-picker-opt' + (allIn ? ' selected' : '');
          opt.innerHTML = '<span class="lp-name" style="color:var(--amber,#e0a040);font-weight:500">'
            + (allIn ? '✓ ' : '') + g + '</span>'
            + '<span class="lp-code">' + gc.join(', ') + '</span>';
          opt.addEventListener('click', function() {
            // One commit for the whole family: a per-code _addCode /
            // _removeCode loop fired onChange once per member, so the page
            // saw (and committed) every half-applied list on the way.
            if (allIn) {
              codes = codes.filter(function(c) { return gc.indexOf(c) < 0; });
            } else {
              gc.forEach(function(c) { if (codes.indexOf(c) < 0) codes.push(c); });
              codes.sort();
            }
            _renderPills();
            _fire();
            _renderDropdown(searchInp ? searchInp.value : '');
          });
          dropContent.appendChild(opt);
        });
      }
      // The layer was placed (flip-up, clamp) at its old height; a search
      // keystroke or a toggle changes it — and a toggle can re-wrap the pills
      // and move the "+ add" anchor — so re-place it while open.
      if (popCtl && popCtl.isOpen()) popCtl.place();
    }

    _renderPills();
    return {
      el: el,
      getLangs: function() { return codes.slice(); },
      setLangs: function(c) {
        codes = Array.isArray(c) ? c.slice() : [];
        codes.sort();
        _renderPills();
      },
    };
  }

  window._renderLanguagePicker = _renderLanguagePicker;
})();</script>
