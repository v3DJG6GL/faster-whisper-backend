
<script>(function(){
  // ---- Shared floating-layer positioner: window._anchorPopover ---------
  // One placement ladder for every dropdown a toolbar or header opens
  // (pick-lists, the header activity popover, tag / language suggest
  // lists, the pipeline colour palette):
  //   1. start-align below the anchor (or end-align when opts.align is
  //      'end' / 'right');
  //   2. if that crosses opts.boundary (an element or a function returning
  //      one — the page canvas: .subbar, .header-inner, a card) flip to the
  //      anchor's other edge (flip-inline);
  //   3. shift into the viewport with an 8px margin (a canvas narrower than
  //      the layer on a phone);
  //   4. flip above the anchor when there is no room below.
  // The native Popover API promotes the element into the top layer so no
  // ancestor overflow/clip/z-index can cut it; without it the node is
  // portaled onto <body> with position:fixed. Re-placed on scroll + resize
  // while open.
  var _POPOVER_OK = typeof HTMLElement !== 'undefined'
    && typeof HTMLElement.prototype.showPopover === 'function';

  function _anchorPopover(popEl, anchorEl, opts) {
    opts = opts || {};
    var align = (opts.align === 'right' || opts.align === 'end') ? 'end' : 'start';
    var gap = opts.gap == null ? 4 : opts.gap;
    var pad = opts.padding == null ? 8 : opts.padding;
    var nativePop = _POPOVER_OK && popEl.hasAttribute('popover');
    var listening = false;

    function boundaryRect() {
      var b = typeof opts.boundary === 'function' ? opts.boundary() : opts.boundary;
      var br = b && b.getBoundingClientRect ? b.getBoundingClientRect() : null;
      if (!br || br.width <= 0) return null;
      return br;
    }
    function place() {
      var r = anchorEl.getBoundingClientRect();
      var pw = popEl.offsetWidth, ph = popEl.offsetHeight;
      var vw = document.documentElement.clientWidth;
      var vh = document.documentElement.clientHeight;
      var br = boundaryRect();
      var left = align === 'end' ? (r.right - pw) : r.left;
      if (br) {                                          // 2: flip at the canvas edge
        if (align === 'start' && left + pw > br.right && r.right - pw >= br.left) left = r.right - pw;
        if (align === 'end' && left < br.left && r.left + pw <= br.right) left = r.left;
      }
      var top = r.bottom + gap;                          // prefer below the anchor
      if (top + ph > vh && r.top - gap - ph >= 0) top = r.top - gap - ph;  // 4: flip up
      left = Math.max(pad, Math.min(left, vw - pw - pad));   // 3: shift into the viewport
      top = Math.max(4, Math.min(top, vh - ph - 4));
      // Override the UA [popover] defaults (inset:0; margin:auto) with explicit
      // fixed coords; right/bottom:auto so a leftover inset can't stretch it.
      popEl.style.position = 'fixed';
      popEl.style.margin = '0';
      popEl.style.left = left + 'px';
      popEl.style.top = top + 'px';
      popEl.style.right = 'auto';
      popEl.style.bottom = 'auto';
    }
    function startListening() {
      if (listening) return;
      listening = true;
      window.addEventListener('scroll', place, true);    // capture: catch inner scrollers
      window.addEventListener('resize', place);
    }
    function stopListening() {
      if (!listening) return;
      listening = false;
      window.removeEventListener('scroll', place, true);
      window.removeEventListener('resize', place);
    }
    function isOpen() {
      if (nativePop) { try { return popEl.matches(':popover-open'); } catch (e) { return false; } }
      return !popEl.hidden && popEl.style.display !== 'none';
    }
    function show() {
      if (!isOpen()) {
        if (nativePop) { try { popEl.showPopover(); } catch (e) {} }
        else {
          if (popEl.parentNode !== document.body) document.body.appendChild(popEl);
          popEl.hidden = false; popEl.style.display = '';
        }
      }
      place(); startListening();
    }
    function hide() {
      if (nativePop) { try { if (isOpen()) popEl.hidePopover(); } catch (e) {} }
      else { popEl.hidden = true; popEl.style.display = 'none'; }
      stopListening();
    }
    // Keep listeners in sync when the browser opens/closes the popover for us
    // (auto popovers via an invoker button, Esc, or outside-click light-dismiss).
    if (nativePop) {
      popEl.addEventListener('toggle', function (ev) {
        if (ev.newState === 'open') { place(); startListening(); }
        else stopListening();
      });
    }
    return { show: show, hide: hide, place: place, isOpen: isOpen };
  }
  window._anchorPopover = _anchorPopover;
})();</script>
