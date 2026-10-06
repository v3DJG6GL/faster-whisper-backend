
<script>(function(){
  var TAG_RE = /^[a-z0-9][a-z0-9-]{0,31}$/;
  function _norm(s) { return String(s == null ? '' : s).trim().toLowerCase(); }

  // The top-layer popover positioner (_anchorPopover) is shared from
  // POPOVER_JS, which ships inside the SCALE_PICKER_JS block. It is looked up
  // when a picker is BUILT, not when this block loads — but unguarded, so a
  // page embedding this block must also embed SCALE_PICKER_JS and call
  // _renderTagPicker only after it ran (settings / api-keys do both).
  var _POPOVER_OK = typeof HTMLElement !== 'undefined'
    && typeof HTMLElement.prototype.showPopover === 'function';

  var _SUG_SEQ = 0;        // unique-id sequence for combobox <option> ids

  function _renderTagPicker(opts) {
    opts = opts || {};
    var tags = Array.isArray(opts.initial) ? opts.initial.slice() : [];
    var available = Array.isArray(opts.available) ? opts.available.slice() : [];
    var disabled = !!opts.disabled;
    var sugId = 'tag-suggest-' + (++_SUG_SEQ);

    var el = document.createElement('div');
    el.className = 'tag-picker' + (disabled ? ' disabled' : '');

    var pills = document.createElement('div');
    pills.className = 'tag-picker-pills';
    el.appendChild(pills);

    var input = document.createElement('input');
    input.type = 'text';
    input.className = 'tag-picker-input';
    input.placeholder = opts.placeholder || '+ tag';
    input.autocomplete = 'off';
    // WAI-ARIA combobox (editable, with listbox popup). aria-activedescendant
    // tracks the highlighted suggestion while focus stays in the input.
    input.setAttribute('role', 'combobox');
    input.setAttribute('aria-autocomplete', 'list');
    input.setAttribute('aria-expanded', 'false');
    input.setAttribute('aria-controls', sugId);
    input.setAttribute('aria-haspopup', 'listbox');
    if (disabled) input.disabled = true;

    var suggest = document.createElement('div');
    suggest.className = 'tag-picker-suggest';
    suggest.id = sugId;
    suggest.setAttribute('role', 'listbox');
    // Top-layer popover so the card's overflow:hidden can't clip the list.
    suggest.setAttribute('popover', 'manual');
    if (!_POPOVER_OK) suggest.hidden = true;     // fallback path manages display
    el.appendChild(suggest);
    var sugPop = window._anchorPopover(suggest, pills, { gap: 2 });

    // Suggestion-list state for keyboard navigation.
    var matchVals = [];     // string values currently shown
    var matchEls = [];      // their <div role=option> elements
    var activeIdx = -1;     // highlighted index, -1 = none
    function _setActive(i) {
      if (i < 0 || i >= matchEls.length) i = -1;
      activeIdx = i;
      matchEls.forEach(function(elm, j) {
        var on = j === i;
        elm.classList.toggle('active', on);
        elm.setAttribute('aria-selected', on ? 'true' : 'false');
      });
      if (i >= 0) {
        input.setAttribute('aria-activedescendant', matchEls[i].id);
        matchEls[i].scrollIntoView({ block: 'nearest' });
      } else {
        input.removeAttribute('aria-activedescendant');
      }
    }

    function _hasTag(s) { return tags.indexOf(s) !== -1; }
    function _notify() {
      if (typeof opts.onChange === 'function') opts.onChange(tags.slice());
    }

    function _render() {
      if (!input.parentNode || input.parentNode !== pills)
        pills.appendChild(input);
      while (pills.firstChild && pills.firstChild !== input)
        pills.removeChild(pills.firstChild);
      var frag = document.createDocumentFragment();
      tags.forEach(function(t) {
        var pill = document.createElement('span');
        pill.className = 'tag-pill';
        pill.textContent = t;
        if (!disabled) {
          var x = document.createElement('button');
          x.type = 'button';
          x.className = 'tag-pill-x';
          x.textContent = '×';
          x.title = 'Remove "' + t + '"';
          x.addEventListener('click', function(e) {
            e.preventDefault();
            var i = tags.indexOf(t);
            if (i >= 0) {
              tags.splice(i, 1);
              _render();
              _notify();
            }
          });
          pill.appendChild(x);
        }
        frag.appendChild(pill);
      });
      pills.insertBefore(frag, input);
    }

    function _tryAdd(raw) {
      var t = _norm(raw);
      if (!t) return false;
      if (!TAG_RE.test(t)) {
        input.classList.add('invalid');
        input.title = 'Invalid tag — lowercase a-z0-9- only, max 32 chars, no leading hyphen';
        return false;
      }
      input.classList.remove('invalid');
      input.title = '';
      if (_hasTag(t)) { input.value = ''; return false; }
      tags.push(t);
      tags.sort();
      input.value = '';
      _render();
      _notify();
      return true;
    }

    function _hideSuggest() {
      sugPop.hide();
      suggest.innerHTML = '';
      matchVals = []; matchEls = []; activeIdx = -1;
      input.setAttribute('aria-expanded', 'false');
      input.removeAttribute('aria-activedescendant');
    }
    function _updateSuggest(prefix) {
      var matches = available.filter(function(a) {
        return a !== prefix && a.indexOf(prefix) === 0 && !_hasTag(a);
      }).slice(0, 8);
      if (!matches.length) { _hideSuggest(); return; }
      suggest.innerHTML = '';
      matchVals = matches.slice();
      matchEls = [];
      matches.forEach(function(a, idx) {
        var item = document.createElement('div');
        item.className = 'tag-picker-suggest-item';
        item.id = sugId + '-opt' + idx;
        item.setAttribute('role', 'option');
        item.setAttribute('aria-selected', 'false');
        item.textContent = a;
        // mousedown not click so input's blur doesn't fire first and
        // hide the suggest before the click registers.
        item.addEventListener('mousedown', function(e) {
          e.preventDefault();
          _tryAdd(a);
          _hideSuggest();
          input.focus();
        });
        // Hover and keyboard share one highlight.
        item.addEventListener('mouseenter', function() { _setActive(idx); });
        suggest.appendChild(item);
        matchEls.push(item);
      });
      _setActive(-1);   // reset highlight + clear any stale aria-activedescendant
      input.setAttribute('aria-expanded', 'true');
      sugPop.show();      // promote to top layer + position under the field
    }

    input.addEventListener('keydown', function(e) {
      var n = matchEls.length;
      if (e.key === 'ArrowDown') {
        if (!n) return;
        e.preventDefault();
        _setActive(activeIdx + 1 >= n ? 0 : activeIdx + 1);
      } else if (e.key === 'ArrowUp') {
        if (!n) return;
        e.preventDefault();
        _setActive(activeIdx - 1 < 0 ? n - 1 : activeIdx - 1);
      } else if (e.key === 'Home' && n) {
        e.preventDefault(); _setActive(0);
      } else if (e.key === 'End' && n) {
        e.preventDefault(); _setActive(n - 1);
      } else if (e.key === 'Enter' || e.key === ',') {
        // Enter on a highlighted suggestion picks it; otherwise (and for
        // comma) commit the typed text — preserving the old behaviour.
        e.preventDefault();
        if (e.key === 'Enter' && activeIdx >= 0) _tryAdd(matchVals[activeIdx]);
        else _tryAdd(input.value);
        _hideSuggest();
      } else if (e.key === 'Backspace' && !input.value && tags.length) {
        tags.pop();
        _render();
        _notify();
      } else if (e.key === 'Escape') {
        // First Escape closes the list; if already closed, clear the input.
        if (sugPop.isOpen()) { e.preventDefault(); _hideSuggest(); }
        else { input.value = ''; input.classList.remove('invalid'); }
      }
    });
    input.addEventListener('input', function() {
      var v = _norm(input.value);
      if (!v) { input.classList.remove('invalid'); _hideSuggest(); return; }
      input.classList.toggle('invalid', !TAG_RE.test(v));
      _updateSuggest(v);
    });
    input.addEventListener('blur', function() {
      // Auto-commit a trailing typed value on blur — easy to forget Enter.
      if (input.value.trim()) _tryAdd(input.value);
      setTimeout(_hideSuggest, 150);
    });

    _render();

    return {
      el: el,
      getTags: function() { return tags.slice(); },
      setTags: function(t) {
        tags = Array.isArray(t) ? t.slice() : [];
        _render();
      },
      setAvailable: function(a) {
        available = Array.isArray(a) ? a.slice() : [];
      },
    };
  }

  window._renderTagPicker = _renderTagPicker;
})();</script>
